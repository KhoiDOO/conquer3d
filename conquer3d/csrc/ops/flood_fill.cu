#include "flood_fill.h"
#include "../constants.h"
#include "../primitive/ray.h"
#include "../primitive/triangle.h"
#include "../maths/maths.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace ops {

    __device__ int8_t atomicCAS_int8(int8_t* address, int8_t compare, int8_t val) {
        int32_t* address_as_int = (int32_t*)((uintptr_t)address & ~3);
        int shift = (((uintptr_t)address & 3) * 8);
        int32_t old = *address_as_int;
        int32_t assumed;
        do {
            assumed = old;
            int8_t current_val = (int8_t)((assumed >> shift) & 0xff);
            if (current_val != compare) {
                break;
            }
            int32_t new_val = (assumed & ~(0xff << shift)) | ((int32_t)(uint8_t)val << shift);
            old = atomicCAS(address_as_int, assumed, new_val);
        } while (assumed != old);
        return (int8_t)((old >> shift) & 0xff);
    }

    __device__ __forceinline__ bool test_segment_intersect_bvh(
        const float3& p0, const float3& p1,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects)
    {
        float3 dir = p1 - p0;
        float len = maths::norm(dir);
        if (len < 1e-8f) return false;
        float3 norm_dir = dir / len;
        
        Ray ray(p0, norm_dir, 0.0f, len);
        
        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;
        
        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];
            
            float t_hit_aabb;
            float3 box_min = bvh_aabb_mins[node_idx] - make_float3(1e-4f, 1e-4f, 1e-4f);
            float3 box_max = bvh_aabb_maxs[node_idx] + make_float3(1e-4f, 1e-4f, 1e-4f);
            if (!ray.is_intersect_aabb(box_min, box_max, t_hit_aabb)) {
                continue;
            }
            
            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                float3 v0 = vertices[tri.x];
                float3 v1 = vertices[tri.y];
                float3 v2 = vertices[tri.z];
                
                float3 edge1 = v1 - v0;
                float3 edge2 = v2 - v0;
                float3 h = maths::cross(ray.direction, edge2);
                float a = maths::dot(edge1, h);

                if (a > -1e-8f && a < 1e-8f) continue; 

                float f = 1.0f / a;
                float3 s = ray.origin - v0;
                float u = f * maths::dot(s, h);

                if (u < -1e-4f || u > 1.0f + 1e-4f) continue;

                float3 q = maths::cross(s, edge1);
                float v = f * maths::dot(ray.direction, q);

                if (v < -1e-4f || u + v > 1.0f + 1e-4f) continue;

                float t = f * maths::dot(edge2, q);
                if (t >= -1e-4f && t <= len + 1e-4f)
                {
                    return true;
                }
            }
            else
            {
                if (stack_ptr + 2 < BVH_STACK_SIZE)
                {
                    int2 children = bvh_children[node_idx];
                    if (children.y != -1) stack[++stack_ptr] = children.y;
                    if (children.x != -1) stack[++stack_ptr] = children.x;
                }
            }
        }
        return false;
    }

    __global__ void init_perimeter_kernel(
        int8_t* __restrict__ mask,
        int* __restrict__ frontier,
        int* __restrict__ frontier_size,
        int RX, int RY, int RZ)
    {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t num_vertices = (int64_t)RX * RY * RZ;
        if (idx >= num_vertices) return;

        int vi = idx / (RY * RZ);
        int rem = idx % (RY * RZ);
        int vj = rem / RZ;
        int vk = rem % RZ;

        if (vi == 0 || vi == RX - 1 || vj == 0 || vj == RY - 1 || vk == 0 || vk == RZ - 1)
        {
            if (mask[idx] == -2)
            {
                mask[idx] = 2; // Water (Open Sea)
                int pos = atomicAdd(frontier_size, 1);
                frontier[pos] = idx;
            }
        }
    }

    __global__ void flood_fill_step_kernel(
        int8_t* __restrict__ mask,
        const int* __restrict__ current_frontier,
        int frontier_size,
        int* __restrict__ next_frontier,
        int* __restrict__ next_frontier_size,
        int RX, int RY, int RZ,
        int connectivity,
        float3 grid_min,
        float3 grid_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= frontier_size) return;

        int vertex_idx = current_frontier[idx];
        int vi = vertex_idx / (RY * RZ);
        int rem = vertex_idx % (RY * RZ);
        int vj = rem / RZ;
        int vk = rem % RZ;
        
        float3 pA = make_float3(
            grid_min.x + vi * grid_spacing.x,
            grid_min.y + vj * grid_spacing.y,
            grid_min.z + vk * grid_spacing.z
        );

        for (int di = -1; di <= 1; di++)
        {
            for (int dj = -1; dj <= 1; dj++)
            {
                for (int dk = -1; dk <= 1; dk++)
                {
                    if (di == 0 && dj == 0 && dk == 0) continue;

                    int dist = abs(di) + abs(dj) + abs(dk);
                    if (connectivity == 6 && dist > 1) continue;
                    if (connectivity == 18 && dist > 2) continue;

                    int ni = vi + di;
                    int nj = vj + dj;
                    int nk = vk + dk;

                    if (ni < 0 || ni >= RX || nj < 0 || nj >= RY || nk < 0 || nk >= RZ) continue;

                    int n_idx = ni * (RY * RZ) + nj * RZ + nk;
                    if (mask[n_idx] == -2)
                    {
                        float3 pB = make_float3(
                            grid_min.x + ni * grid_spacing.x,
                            grid_min.y + nj * grid_spacing.y,
                            grid_min.z + nk * grid_spacing.z
                        );
                        
                        if (!test_segment_intersect_bvh(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects))
                        {
                            int8_t old_val = atomicCAS_int8(&mask[n_idx], -2, 2);
                            if (old_val == -2)
                            {
                                int pos = atomicAdd(next_frontier_size, 1);
                                next_frontier[pos] = n_idx;
                            }
                        }
                    }
                }
            }
        }
    }

    torch::Tensor compute_flood_fill(
        const torch::Tensor& vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& aabb_mins,
        const torch::Tensor& aabb_maxs,
        const torch::Tensor& bvh_children,
        const torch::Tensor& object_ids,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> grid_res,
        int connectivity
    ) {
        int64_t rx = grid_res[0];
        int64_t ry = grid_res[1];
        int64_t rz = grid_res[2];
        int64_t num_vertices = rx * ry * rz;
        auto options = torch::TensorOptions().device(vertices.device()).dtype(torch::kInt32);

        auto mask = torch::full({num_vertices}, -2, options.dtype(torch::kInt8));

        auto current_frontier = torch::empty({num_vertices}, options);
        auto next_frontier = torch::empty({num_vertices}, options);
        auto frontier_size = torch::zeros({1}, options);
        auto next_frontier_size = torch::zeros({1}, options);

        int threads = NTHREADS;
        int blocks = (num_vertices + threads - 1) / threads;

        init_perimeter_kernel<<<blocks, threads>>>(
            mask.data_ptr<int8_t>(),
            current_frontier.data_ptr<int>(),
            frontier_size.data_ptr<int>(),
            static_cast<int>(rx),
            static_cast<int>(ry),
            static_cast<int>(rz)
        );

        float3 f_min = make_float3(grid_min[0], grid_min[1], grid_min[2]);
        float3 f_spacing = make_float3(
            (rx > 1) ? (grid_max[0] - grid_min[0]) / (rx - 1) : 1.0f,
            (ry > 1) ? (grid_max[1] - grid_min[1]) / (ry - 1) : 1.0f,
            (rz > 1) ? (grid_max[2] - grid_min[2]) / (rz - 1) : 1.0f
        );

        int curr_size = frontier_size.item<int>();

        while (curr_size > 0)
        {
            next_frontier_size.zero_();
            int step_blocks = (curr_size + threads - 1) / threads;

            flood_fill_step_kernel<<<step_blocks, threads>>>(
                mask.data_ptr<int8_t>(),
                current_frontier.data_ptr<int>(),
                curr_size,
                next_frontier.data_ptr<int>(),
                next_frontier_size.data_ptr<int>(),
                static_cast<int>(rx),
                static_cast<int>(ry),
                static_cast<int>(rz),
                connectivity,
                f_min,
                f_spacing,
                (const float3*)aabb_mins.data_ptr<float>(),
                (const float3*)aabb_maxs.data_ptr<float>(),
                (const int2*)bvh_children.data_ptr<int>(),
                object_ids.data_ptr<int>(),
                (const float3*)vertices.data_ptr<float>(),
                (const int3*)triangles.data_ptr<int>(),
                static_cast<int>(object_ids.size(0))
            );

            curr_size = next_frontier_size.item<int>();
            std::swap(current_frontier, next_frontier);
        }

        return mask.view({rx, ry, rz});
    }

} // namespace ops
