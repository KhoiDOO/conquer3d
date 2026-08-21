/**
 * @file flood_fill_dilated.cu
 * @brief CUDA kernel implementations for 2-Level Hierarchical Boundary-Aware Dilated Flood Fill.
 */

#include "flood_fill_dilated.h"
#include "../constants.h"
#include "../primitive/ray.h"
#include "../primitive/triangle.h"
#include "../maths/maths.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace ops {

    __device__ static int8_t atomicCAS_int8_dilated(int8_t* address, int8_t compare, int8_t val) {
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

    __device__ __forceinline__ bool test_segment_intersect_bvh_dilated(
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

    __device__ __forceinline__ bool test_box_overlap_bvh_dilated(
        const float3& box_min, const float3& box_max,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects)
    {
        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];
            float3 node_min = bvh_aabb_mins[node_idx];
            float3 node_max = bvh_aabb_maxs[node_idx];

            if (box_max.x < node_min.x || box_min.x > node_max.x ||
                box_max.y < node_min.y || box_min.y > node_max.y ||
                box_max.z < node_min.z || box_min.z > node_max.z)
            {
                continue;
            }

            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
                if (T.is_voxel_intersect(box_min, box_max)) {
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

    // -------------------------------------------------------------
    // Stage 1: Coarse & Fine Band Initialization with Dilation
    // -------------------------------------------------------------

    __global__ void init_coarse_grid_dilated_kernel(
        int8_t* __restrict__ coarse_mask,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        int dilation_radius,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects)
    {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total_coarse = (int64_t)CX * CY * CZ;
        if (idx >= total_coarse) return;

        int ci = idx / (CY * CZ);
        int rem = idx % (CY * CZ);
        int cj = rem / CZ;
        int ck = rem % CZ;

        float3 margin = make_float3(
            (float)dilation_radius * fine_spacing.x,
            (float)dilation_radius * fine_spacing.y,
            (float)dilation_radius * fine_spacing.z
        );

        float3 box_min = make_float3(
            grid_min.x + ci * BX * fine_spacing.x - margin.x,
            grid_min.y + cj * BY * fine_spacing.y - margin.y,
            grid_min.z + ck * BZ * fine_spacing.z - margin.z
        );
        float3 box_max = make_float3(
            grid_min.x + (ci + 1) * BX * fine_spacing.x + margin.x,
            grid_min.y + (cj + 1) * BY * fine_spacing.y + margin.y,
            grid_min.z + (ck + 1) * BZ * fine_spacing.z + margin.z
        );

        bool intersects = test_box_overlap_bvh_dilated(
            box_min, box_max,
            bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids,
            vertices, triangles, num_objects
        );

        coarse_mask[idx] = intersects ? (int8_t)1 : (int8_t)-2;
    }

    __global__ void init_perimeter_seeds_dilated_kernel(
        int8_t* __restrict__ coarse_mask,
        int CX, int CY, int CZ)
    {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total_coarse = (int64_t)CX * CY * CZ;
        if (idx >= total_coarse) return;

        int ci = idx / (CY * CZ);
        int rem = idx % (CY * CZ);
        int cj = rem / CZ;
        int ck = rem % CZ;

        if (ci == 0 || ci == CX - 1 || cj == 0 || cj == CY - 1 || ck == 0 || ck == CZ - 1) {
            if (coarse_mask[idx] == -2) {
                coarse_mask[idx] = 2; // Exterior Water
            }
        }
    }

    __global__ void init_fine_boundary_blocks_dilated_kernel(
        const int* __restrict__ boundary_coords,
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        int dilation_radius,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int ci = boundary_coords[block_id * 3 + 0];
        int cj = boundary_coords[block_id * 3 + 1];
        int ck = boundary_coords[block_id * 3 + 2];
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        float3 margin = make_float3(
            (float)dilation_radius * fine_spacing.x,
            (float)dilation_radius * fine_spacing.y,
            (float)dilation_radius * fine_spacing.z
        );

        float3 v_min = make_float3(
            grid_min.x + (ci * BX + fi) * fine_spacing.x - margin.x,
            grid_min.y + (cj * BY + fj) * fine_spacing.y - margin.y,
            grid_min.z + (ck * BZ + fk) * fine_spacing.z - margin.z
        );
        float3 v_max = make_float3(
            grid_min.x + (ci * BX + fi + 1) * fine_spacing.x + margin.x,
            grid_min.y + (cj * BY + fj + 1) * fine_spacing.y + margin.y,
            grid_min.z + (ck * BZ + fk + 1) * fine_spacing.z + margin.z
        );

        bool in_dilated_band = test_box_overlap_bvh_dilated(
            v_min, v_max,
            bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids,
            vertices, triangles, num_objects
        );

        // 0: Undetermined Dilated Band (barrier), -2: Dry/Unvisited
        my_fine_mask[local_idx] = in_dilated_band ? (int8_t)0 : (int8_t)-2;
    }

    __global__ void init_boundary_perimeter_faces_dilated_kernel(
        const int* __restrict__ boundary_coords,
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int ci = boundary_coords[block_id * 3 + 0];
        int cj = boundary_coords[block_id * 3 + 1];
        int ck = boundary_coords[block_id * 3 + 2];
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        bool on_outer_perimeter = false;
        if (ci == 0 && fi == 0) on_outer_perimeter = true;
        if (ci == CX - 1 && fi == BX - 1) on_outer_perimeter = true;
        if (cj == 0 && fj == 0) on_outer_perimeter = true;
        if (cj == CY - 1 && fj == BY - 1) on_outer_perimeter = true;
        if (ck == 0 && fk == 0) on_outer_perimeter = true;
        if (ck == CZ - 1 && fk == BZ - 1) on_outer_perimeter = true;

        if (on_outer_perimeter) {
            my_fine_mask[local_idx] = 2; // Always force Exterior Water on domain perimeter
        }
    }

    // -------------------------------------------------------------
    // Stage 2: Coarse-to-Fine Wavefront Flood Fill Propagation
    // -------------------------------------------------------------

    __global__ void coarse_to_coarse_dilated_kernel(
        int8_t* __restrict__ coarse_mask,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects,
        int* __restrict__ changed_flag)
    {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t total_coarse = (int64_t)CX * CY * CZ;
        if (idx >= total_coarse) return;

        if (coarse_mask[idx] != -2) return;

        int ci = idx / (CY * CZ);
        int rem = idx % (CY * CZ);
        int cj = rem / CZ;
        int ck = rem % CZ;

        float3 pA = make_float3(
            grid_min.x + (ci + 0.5f) * BX * fine_spacing.x,
            grid_min.y + (cj + 0.5f) * BY * fine_spacing.y,
            grid_min.z + (ck + 0.5f) * BZ * fine_spacing.z
        );

        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};

        for (int k = 0; k < 6; ++k) {
            int ni = ci + di[k];
            int nj = cj + dj[k];
            int nk = ck + dk[k];
            if (ni >= 0 && ni < CX && nj >= 0 && nj < CY && nk >= 0 && nk < CZ) {
                int n_idx = ni * (CY * CZ) + nj * CZ + nk;
                if (coarse_mask[n_idx] == 2) {
                    float3 pB = make_float3(
                        grid_min.x + (ni + 0.5f) * BX * fine_spacing.x,
                        grid_min.y + (nj + 0.5f) * BY * fine_spacing.y,
                        grid_min.z + (nk + 0.5f) * BZ * fine_spacing.z
                    );
                    if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) {
                        coarse_mask[idx] = 2;
                        atomicExch(changed_flag, 1);
                        return;
                    }
                }
            }
        }
    }

    __global__ void coarse_to_fine_dilated_kernel(
        const int* __restrict__ boundary_coords,
        const int8_t* __restrict__ coarse_mask,
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects,
        int* __restrict__ changed_flag)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int ci = boundary_coords[block_id * 3 + 0];
        int cj = boundary_coords[block_id * 3 + 1];
        int ck = boundary_coords[block_id * 3 + 2];
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        if (my_fine_mask[local_idx] != -2) return;

        float3 pA = make_float3(
            grid_min.x + (ci * BX + fi) * fine_spacing.x,
            grid_min.y + (cj * BY + fj) * fine_spacing.y,
            grid_min.z + (ck * BZ + fk) * fine_spacing.z
        );

        bool seed = false;
        if (fi == 0 && ci > 0 && coarse_mask[(ci - 1) * (CY * CZ) + cj * CZ + ck] == 2) {
            float3 pB = make_float3(grid_min.x + ((ci - 1) * BX + (BX - 1)) * fine_spacing.x, pA.y, pA.z);
            if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }
        if (!seed && fi == BX - 1 && ci < CX - 1 && coarse_mask[(ci + 1) * (CY * CZ) + cj * CZ + ck] == 2) {
            float3 pB = make_float3(grid_min.x + ((ci + 1) * BX + 0) * fine_spacing.x, pA.y, pA.z);
            if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }
        if (!seed && fj == 0 && cj > 0 && coarse_mask[ci * (CY * CZ) + (cj - 1) * CZ + ck] == 2) {
            float3 pB = make_float3(pA.x, grid_min.y + ((cj - 1) * BY + (BY - 1)) * fine_spacing.y, pA.z);
            if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }
        if (!seed && fj == BY - 1 && cj < CY - 1 && coarse_mask[ci * (CY * CZ) + (cj + 1) * CZ + ck] == 2) {
            float3 pB = make_float3(pA.x, grid_min.y + ((cj + 1) * BY + 0) * fine_spacing.y, pA.z);
            if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }
        if (!seed && fk == 0 && ck > 0 && coarse_mask[ci * (CY * CZ) + cj * CZ + (ck - 1)] == 2) {
            float3 pB = make_float3(pA.x, pA.y, grid_min.z + ((ck - 1) * BZ + (BZ - 1)) * fine_spacing.z);
            if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }
        if (!seed && fk == BZ - 1 && ck < CZ - 1 && coarse_mask[ci * (CY * CZ) + cj * CZ + (ck + 1)] == 2) {
            float3 pB = make_float3(pA.x, pA.y, grid_min.z + ((ck + 1) * BZ + 0) * fine_spacing.z);
            if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
        }

        if (seed) {
            my_fine_mask[local_idx] = 2;
            atomicExch(changed_flag, 1);
        }
    }

    __global__ void fine_intra_block_bfs_dilated_kernel(
        const int* __restrict__ boundary_coords,
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects,
        int* __restrict__ changed_flag)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int ci = boundary_coords[block_id * 3 + 0];
        int cj = boundary_coords[block_id * 3 + 1];
        int ck = boundary_coords[block_id * 3 + 2];
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        __shared__ int8_t s_mask[8][8][8];
        s_mask[fi][fj][fk] = my_fine_mask[local_idx];
        __syncthreads();

        float3 pA = make_float3(
            grid_min.x + (ci * BX + fi) * fine_spacing.x,
            grid_min.y + (cj * BY + fj) * fine_spacing.y,
            grid_min.z + (ck * BZ + fk) * fine_spacing.z
        );

        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};

        bool any_local_change = false;

        for (int iter = 0; iter < (BX + BY + BZ); ++iter)
        {
            if (s_mask[fi][fj][fk] == 2)
            {
                for (int k = 0; k < 6; ++k)
                {
                    int nfi = fi + di[k];
                    int nfj = fj + dj[k];
                    int nfk = fk + dk[k];

                    if (nfi >= 0 && nfi < BX && nfj >= 0 && nfj < BY && nfk >= 0 && nfk < BZ)
                    {
                        if (s_mask[nfi][nfj][nfk] == -2)
                        {
                            float3 pB = make_float3(
                                grid_min.x + (ci * BX + nfi) * fine_spacing.x,
                                grid_min.y + (cj * BY + nfj) * fine_spacing.y,
                                grid_min.z + (ck * BZ + nfk) * fine_spacing.z
                            );
                            if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects))
                            {
                                s_mask[nfi][nfj][nfk] = 2;
                                any_local_change = true;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (any_local_change) {
            my_fine_mask[local_idx] = s_mask[fi][fj][fk];
            atomicExch(changed_flag, 1);
        }
    }

    __global__ void fine_to_fine_halo_dilated_kernel(
        const int* __restrict__ boundary_coords,
        const int32_t* __restrict__ boundary_lookup,
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ,
        float3 grid_min,
        float3 fine_spacing,
        const float3* __restrict__ bvh_aabb_mins,
        const float3* __restrict__ bvh_aabb_maxs,
        const int2* __restrict__ bvh_children,
        const int* __restrict__ object_ids,
        const float3* __restrict__ vertices,
        const int3* __restrict__ triangles,
        int num_objects,
        int* __restrict__ changed_flag)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int ci = boundary_coords[block_id * 3 + 0];
        int cj = boundary_coords[block_id * 3 + 1];
        int ck = boundary_coords[block_id * 3 + 2];
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        if (my_fine_mask[local_idx] != -2) return;

        float3 pA = make_float3(
            grid_min.x + (ci * BX + fi) * fine_spacing.x,
            grid_min.y + (cj * BY + fj) * fine_spacing.y,
            grid_min.z + (ck * BZ + fk) * fine_spacing.z
        );

        bool seed = false;
        if (fi == 0 && ci > 0) {
            int n_block = boundary_lookup[(ci - 1) * (CY * CZ) + cj * CZ + ck];
            if (n_block >= 0 && fine_masks[(int64_t)n_block * (BX * BY * BZ) + (BX - 1) * (BY * BZ) + fj * BZ + fk] == 2) {
                float3 pB = make_float3(grid_min.x + ((ci - 1) * BX + (BX - 1)) * fine_spacing.x, pA.y, pA.z);
                if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
            }
        }
        if (!seed && fi == BX - 1 && ci < CX - 1) {
            int n_block = boundary_lookup[(ci + 1) * (CY * CZ) + cj * CZ + ck];
            if (n_block >= 0 && fine_masks[(int64_t)n_block * (BX * BY * BZ) + 0 * (BY * BZ) + fj * BZ + fk] == 2) {
                float3 pB = make_float3(grid_min.x + ((ci + 1) * BX + 0) * fine_spacing.x, pA.y, pA.z);
                if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
            }
        }
        if (!seed && fj == 0 && cj > 0) {
            int n_block = boundary_lookup[ci * (CY * CZ) + (cj - 1) * CZ + ck];
            if (n_block >= 0 && fine_masks[(int64_t)n_block * (BX * BY * BZ) + fi * (BY * BZ) + (BY - 1) * BZ + fk] == 2) {
                float3 pB = make_float3(pA.x, grid_min.y + ((cj - 1) * BY + (BY - 1)) * fine_spacing.y, pA.z);
                if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
            }
        }
        if (!seed && fj == BY - 1 && cj < CY - 1) {
            int n_block = boundary_lookup[ci * (CY * CZ) + (cj + 1) * CZ + ck];
            if (n_block >= 0 && fine_masks[(int64_t)n_block * (BX * BY * BZ) + fi * (BY * BZ) + 0 * BZ + fk] == 2) {
                float3 pB = make_float3(pA.x, grid_min.y + ((cj + 1) * BY + 0) * fine_spacing.y, pA.z);
                if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
            }
        }
        if (!seed && fk == 0 && ck > 0) {
            int n_block = boundary_lookup[ci * (CY * CZ) + cj * CZ + (ck - 1)];
            if (n_block >= 0 && fine_masks[(int64_t)n_block * (BX * BY * BZ) + fi * (BY * BZ) + fj * BZ + (BZ - 1)] == 2) {
                float3 pB = make_float3(pA.x, pA.y, grid_min.z + ((ck - 1) * BZ + (BZ - 1)) * fine_spacing.z);
                if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
            }
        }
        if (!seed && fk == BZ - 1 && ck < CZ - 1) {
            int n_block = boundary_lookup[ci * (CY * CZ) + cj * CZ + (ck + 1)];
            if (n_block >= 0 && fine_masks[(int64_t)n_block * (BX * BY * BZ) + fi * (BY * BZ) + fj * BZ + 0] == 2) {
                float3 pB = make_float3(pA.x, pA.y, grid_min.z + ((ck + 1) * BZ + 0) * fine_spacing.z);
                if (!test_segment_intersect_bvh_dilated(pA, pB, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects)) seed = true;
            }
        }

        if (seed) {
            my_fine_mask[local_idx] = 2;
            atomicExch(changed_flag, 1);
        }
    }

    // -------------------------------------------------------------
    // Stage 3: Establish Definitive Interior (-1) on Dry Unvisited
    // -------------------------------------------------------------

    __global__ void mark_dry_coarse_interior_dilated_kernel(
        int8_t* __restrict__ coarse_mask,
        int64_t total_coarse)
    {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_coarse && coarse_mask[idx] == -2) {
            coarse_mask[idx] = -1; // Interior
        }
    }

    __global__ void mark_dry_fine_interior_dilated_kernel(
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int BX, int BY, int BZ)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;
        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);

        if (my_fine_mask[local_idx] == -2) {
            my_fine_mask[local_idx] = -1; // Definitive Interior
        }
    }

    // -------------------------------------------------------------
    // Stage 4: Consensus Relabeling on Undetermined Band (0)
    // -------------------------------------------------------------

    __global__ void consensus_relabel_kernel(
        const int* __restrict__ boundary_coords,
        const int8_t* __restrict__ coarse_mask,
        const int32_t* __restrict__ boundary_lookup,
        const int8_t* __restrict__ fine_masks_in,
        int8_t* __restrict__ fine_masks_out,
        int num_boundary_blocks,
        int CX, int CY, int CZ,
        int BX, int BY, int BZ,
        int connectivity,
        int* __restrict__ changed_flag)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int ci = boundary_coords[block_id * 3 + 0];
        int cj = boundary_coords[block_id * 3 + 1];
        int ck = boundary_coords[block_id * 3 + 2];
        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;

        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int64_t offset = (int64_t)block_id * (BX * BY * BZ) + local_idx;

        int8_t cur_val = fine_masks_in[offset];
        if (cur_val != 0) {
            fine_masks_out[offset] = cur_val;
            return;
        }

        float vote_sum = 0.0f;
        int num_votes = 0;

        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                for (int dk = -1; dk <= 1; ++dk) {
                    if (di == 0 && dj == 0 && dk == 0) continue;

                    int manh_dist = abs(di) + abs(dj) + abs(dk);
                    if (connectivity == 6 && manh_dist > 1) continue;
                    if (connectivity == 18 && manh_dist > 2) continue;

                    float weight = 1.0f / sqrtf((float)(di * di + dj * dj + dk * dk));

                    int g_fi = fi + di;
                    int g_fj = fj + dj;
                    int g_fk = fk + dk;

                    int n_ci = ci;
                    int n_cj = cj;
                    int n_ck = ck;

                    if (g_fi < 0) { n_ci--; g_fi += BX; }
                    else if (g_fi >= BX) { n_ci++; g_fi -= BX; }

                    if (g_fj < 0) { n_cj--; g_fj += BY; }
                    else if (g_fj >= BY) { n_cj++; g_fj -= BY; }

                    if (g_fk < 0) { n_ck--; g_fk += BZ; }
                    else if (g_fk >= BZ) { n_ck++; g_fk -= BZ; }

                    int8_t n_val = 2; // Outside bounding box is Exterior Water
                    if (n_ci >= 0 && n_ci < CX && n_cj >= 0 && n_cj < CY && n_ck >= 0 && n_ck < CZ) {
                        int n_block = boundary_lookup[n_ci * (CY * CZ) + n_cj * CZ + n_ck];
                        if (n_block >= 0) {
                            n_val = fine_masks_in[(int64_t)n_block * (BX * BY * BZ) + g_fi * (BY * BZ) + g_fj * BZ + g_fk];
                        } else {
                            n_val = coarse_mask[n_ci * (CY * CZ) + n_cj * CZ + n_ck];
                        }
                    }

                    if (n_val == 2) {
                        vote_sum += weight * 1.0f;  // Exterior vote
                        num_votes++;
                    } else if (n_val == -1) {
                        vote_sum += weight * -1.0f; // Interior vote
                        num_votes++;
                    }
                }
            }
        }

        if (num_votes > 0) {
            int8_t resolved_label = (vote_sum > 0.0f) ? (int8_t)2 : (int8_t)-1;
            fine_masks_out[offset] = resolved_label;
            atomicExch(changed_flag, 1);
        } else {
            fine_masks_out[offset] = 0;
        }
    }

    __global__ void finalize_interior_dilated_kernel(
        int8_t* __restrict__ fine_masks,
        int num_boundary_blocks,
        int BX, int BY, int BZ)
    {
        int block_id = blockIdx.x;
        if (block_id >= num_boundary_blocks) return;

        int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;
        int local_idx = fi * (BY * BZ) + fj * BZ + fk;
        int8_t* my_fine_mask = fine_masks + (int64_t)block_id * (BX * BY * BZ);
        if (my_fine_mask[local_idx] == 0) {
            my_fine_mask[local_idx] = -1; // Interior
        }
    }

    // -------------------------------------------------------------
    // Host Pipeline Execution Function
    // -------------------------------------------------------------

    static int choose_best_divisor(int res, int max_b = 8) {
        int best = 1;
        for (int b = max_b; b >= 1; --b) {
            if (res % b == 0) {
                return b;
            }
        }
        return best;
    }

    CFFloodFillResult compute_flood_fill_dilated_cf(
        const torch::Tensor& vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& aabb_mins,
        const torch::Tensor& aabb_maxs,
        const torch::Tensor& bvh_children,
        const torch::Tensor& object_ids,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> grid_res,
        int dilation_radius,
        int min_cavity_size,
        std::vector<int64_t> block_size,
        int connectivity)
    {
        TORCH_CHECK(vertices.is_cuda(), "vertices must be on CUDA");
        TORCH_CHECK(triangles.is_cuda(), "triangles must be on CUDA");

        int64_t rx = grid_res[0], ry = grid_res[1], rz = grid_res[2];
        
        int64_t BX = (block_size.size() >= 3 && block_size[0] > 0) ? block_size[0] : choose_best_divisor(static_cast<int>(rx), 8);
        int64_t BY = (block_size.size() >= 3 && block_size[1] > 0) ? block_size[1] : choose_best_divisor(static_cast<int>(ry), 8);
        int64_t BZ = (block_size.size() >= 3 && block_size[2] > 0) ? block_size[2] : choose_best_divisor(static_cast<int>(rz), 8);

        int64_t CX = rx / BX;
        int64_t CY = ry / BY;
        int64_t CZ = rz / BZ;
        std::vector<int64_t> coarse_res = {CX, CY, CZ};
        block_size = {BX, BY, BZ};

        float3 g_min = make_float3(grid_min[0], grid_min[1], grid_min[2]);
        float3 g_max = make_float3(grid_max[0], grid_max[1], grid_max[2]);
        float3 fine_spacing = make_float3(
            (rx > 1) ? (g_max.x - g_min.x) / (float)(rx - 1) : 1.0f,
            (ry > 1) ? (g_max.y - g_min.y) / (float)(ry - 1) : 1.0f,
            (rz > 1) ? (g_max.z - g_min.z) / (float)(rz - 1) : 1.0f
        );

        auto options_i8 = torch::TensorOptions().device(vertices.device()).dtype(torch::kInt8);
        auto options_i32 = torch::TensorOptions().device(vertices.device()).dtype(torch::kInt32);

        int64_t total_coarse = (int64_t)CX * CY * CZ;
        torch::Tensor coarse_mask = torch::full({CX, CY, CZ}, -2, options_i8);

        const float3* d_aabb_mins = (const float3*)aabb_mins.data_ptr<float>();
        const float3* d_aabb_maxs = (const float3*)aabb_maxs.data_ptr<float>();
        const int2* d_bvh_children = (const int2*)bvh_children.data_ptr<int>();
        const int* d_object_ids = object_ids.data_ptr<int>();
        const float3* d_vertices = (const float3*)vertices.data_ptr<float>();
        const int3* d_triangles = (const int3*)triangles.data_ptr<int>();
        int num_objects = triangles.size(0);

        int threads = 256;
        int blocks = (total_coarse + threads - 1) / threads;

        // 1. Identify coarse boundary blocks with dilation padding
        init_coarse_grid_dilated_kernel<<<blocks, threads>>>(
            coarse_mask.data_ptr<int8_t>(),
            CX, CY, CZ, BX, BY, BZ,
            g_min, fine_spacing, dilation_radius,
            d_aabb_mins, d_aabb_maxs, d_bvh_children, d_object_ids,
            d_vertices, d_triangles, num_objects
        );

        init_perimeter_seeds_dilated_kernel<<<blocks, threads>>>(
            coarse_mask.data_ptr<int8_t>(),
            CX, CY, CZ
        );

        // 2. Compact boundary block coordinates and create lookup
        torch::Tensor boundary_indices = (coarse_mask == 1).nonzero();
        int num_boundary_blocks = boundary_indices.size(0);

        torch::Tensor boundary_block_coords = torch::empty({num_boundary_blocks, 3}, options_i32);
        torch::Tensor boundary_block_lookup = torch::full({CX, CY, CZ}, -1, options_i32);

        if (num_boundary_blocks > 0) {
            boundary_block_coords = boundary_indices.to(torch::kInt32).contiguous();
            torch::Tensor block_ids = torch::arange(num_boundary_blocks, options_i32);
            boundary_block_lookup.index_put_({boundary_indices.select(1, 0), boundary_indices.select(1, 1), boundary_indices.select(1, 2)}, block_ids);
        }

        torch::Tensor fine_boundary_masks = torch::full({num_boundary_blocks, BX, BY, BZ}, -2, options_i8);

        dim3 block_dim_3d(BX, BY, BZ);

        if (num_boundary_blocks > 0) {
            init_fine_boundary_blocks_dilated_kernel<<<num_boundary_blocks, block_dim_3d>>>(
                boundary_block_coords.data_ptr<int>(),
                fine_boundary_masks.data_ptr<int8_t>(),
                num_boundary_blocks,
                BX, BY, BZ,
                g_min, fine_spacing, dilation_radius,
                d_aabb_mins, d_aabb_maxs, d_bvh_children, d_object_ids,
                d_vertices, d_triangles, num_objects
            );

            init_boundary_perimeter_faces_dilated_kernel<<<num_boundary_blocks, block_dim_3d>>>(
                boundary_block_coords.data_ptr<int>(),
                fine_boundary_masks.data_ptr<int8_t>(),
                num_boundary_blocks,
                CX, CY, CZ, BX, BY, BZ
            );
        }

        // 3. Hierarchical Wavefront Propagation
        int* d_changed_flag;
        cudaMalloc(&d_changed_flag, sizeof(int));
        int h_changed = 1;
        int max_iters = (CX + CY + CZ) * 2;
        int iter = 0;

        while (h_changed && iter < max_iters) {
            h_changed = 0;
            cudaMemset(d_changed_flag, 0, sizeof(int));

            coarse_to_coarse_dilated_kernel<<<blocks, threads>>>(
                coarse_mask.data_ptr<int8_t>(),
                CX, CY, CZ, BX, BY, BZ,
                g_min, fine_spacing,
                d_aabb_mins, d_aabb_maxs, d_bvh_children, d_object_ids,
                d_vertices, d_triangles, num_objects,
                d_changed_flag
            );

            if (num_boundary_blocks > 0) {
                coarse_to_fine_dilated_kernel<<<num_boundary_blocks, block_dim_3d>>>(
                    boundary_block_coords.data_ptr<int>(),
                    coarse_mask.data_ptr<int8_t>(),
                    fine_boundary_masks.data_ptr<int8_t>(),
                    num_boundary_blocks,
                    CX, CY, CZ, BX, BY, BZ,
                    g_min, fine_spacing,
                    d_aabb_mins, d_aabb_maxs, d_bvh_children, d_object_ids,
                    d_vertices, d_triangles, num_objects,
                    d_changed_flag
                );

                fine_intra_block_bfs_dilated_kernel<<<num_boundary_blocks, block_dim_3d>>>(
                    boundary_block_coords.data_ptr<int>(),
                    fine_boundary_masks.data_ptr<int8_t>(),
                    num_boundary_blocks,
                    BX, BY, BZ,
                    g_min, fine_spacing,
                    d_aabb_mins, d_aabb_maxs, d_bvh_children, d_object_ids,
                    d_vertices, d_triangles, num_objects,
                    d_changed_flag
                );

                fine_to_fine_halo_dilated_kernel<<<num_boundary_blocks, block_dim_3d>>>(
                    boundary_block_coords.data_ptr<int>(),
                    boundary_block_lookup.data_ptr<int32_t>(),
                    fine_boundary_masks.data_ptr<int8_t>(),
                    num_boundary_blocks,
                    CX, CY, CZ, BX, BY, BZ,
                    g_min, fine_spacing,
                    d_aabb_mins, d_aabb_maxs, d_bvh_children, d_object_ids,
                    d_vertices, d_triangles, num_objects,
                    d_changed_flag
                );
            }

            cudaMemcpy(&h_changed, d_changed_flag, sizeof(int), cudaMemcpyDeviceToHost);
            iter++;
        }

        // 4. Mark all dry/unvisited regions as Definitive Interior (-1)
        mark_dry_coarse_interior_dilated_kernel<<<blocks, threads>>>(
            coarse_mask.data_ptr<int8_t>(),
            total_coarse
        );

        if (num_boundary_blocks > 0) {
            mark_dry_fine_interior_dilated_kernel<<<num_boundary_blocks, block_dim_3d>>>(
                fine_boundary_masks.data_ptr<int8_t>(),
                num_boundary_blocks,
                BX, BY, BZ
            );
        }

        // 5. Consensus Relabeling on Undetermined Band (0) via Iterative Diffusion
        if (num_boundary_blocks > 0) {
            torch::Tensor fine_boundary_masks_buf = fine_boundary_masks.clone();
            int h_changed_vote = 1;
            int max_vote_iters = dilation_radius + 4;
            int vote_iter = 0;

            while (h_changed_vote && vote_iter < max_vote_iters) {
                h_changed_vote = 0;
                cudaMemset(d_changed_flag, 0, sizeof(int));

                consensus_relabel_kernel<<<num_boundary_blocks, block_dim_3d>>>(
                    boundary_block_coords.data_ptr<int>(),
                    coarse_mask.data_ptr<int8_t>(),
                    boundary_block_lookup.data_ptr<int32_t>(),
                    fine_boundary_masks.data_ptr<int8_t>(),
                    fine_boundary_masks_buf.data_ptr<int8_t>(),
                    num_boundary_blocks,
                    CX, CY, CZ, BX, BY, BZ,
                    connectivity,
                    d_changed_flag
                );

                fine_boundary_masks.copy_(fine_boundary_masks_buf);
                cudaMemcpy(&h_changed_vote, d_changed_flag, sizeof(int), cudaMemcpyDeviceToHost);
                vote_iter++;
            }

            init_boundary_perimeter_faces_dilated_kernel<<<num_boundary_blocks, block_dim_3d>>>(
                boundary_block_coords.data_ptr<int>(),
                fine_boundary_masks.data_ptr<int8_t>(),
                num_boundary_blocks,
                CX, CY, CZ, BX, BY, BZ
            );
        }

        cudaFree(d_changed_flag);

        CFFloodFillResult result;
        result.coarse_mask = coarse_mask;
        result.boundary_block_coords = boundary_block_coords;
        result.boundary_block_lookup = boundary_block_lookup;
        result.fine_boundary_masks = fine_boundary_masks;
        result.block_size = block_size;
        result.coarse_res = coarse_res;
        result.grid_min = grid_min;
        result.grid_max = grid_max;
        result.grid_res = grid_res;
        return result;
    }

} // namespace ops
