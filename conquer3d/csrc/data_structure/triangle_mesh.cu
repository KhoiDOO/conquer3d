/**
 * @file triangle_mesh.cu
 * @brief CUDA kernel implementations for Discrete Differential Geometry (DDG) operators and topological mesh analysis.
 */

#include "triangle_mesh.h"
#include "../primitive/triangle.h"
#include "../primitive/edge.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace triangle_mesh
{
    __global__ void compute_triangle_normals_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ triangle_normals)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            float3 v0 = vertices[tri.x];
            float3 v1 = vertices[tri.y];
            float3 v2 = vertices[tri.z];
            
            triangle_normals[idx] = triangle::compute_normal(v0, v1, v2);
        }
    }

    __global__ void compute_triangle_areas_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ triangle_areas)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            float3 v0 = vertices[tri.x];
            float3 v1 = vertices[tri.y];
            float3 v2 = vertices[tri.z];
            
            triangle_areas[idx] = triangle::compute_area(v0, v1, v2);
        }
    }

    __global__ void compute_quality_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ qualities)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            qualities[idx] = T.compute_quality();
        }
    }

    __global__ void compute_aspect_ratio_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        int mode,
        float *__restrict__ aspect_ratios)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            aspect_ratios[idx] = T.compute_ar(mode);
        }
    }

    __global__ void compute_radii_ratio_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ ratios)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            ratios[idx] = T.compute_radii_ratio();
        }
    }

    __global__ void compute_triangle_regularity_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ regularities)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            regularities[idx] = T.compute_triangle_regularity();
        }
    }

    __global__ void compute_radius_edge_ratio_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ ratios)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            ratios[idx] = T.compute_radius_edge_ratio();
        }
    }

    __global__ void compute_angle_deviation_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ deviations)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            deviations[idx] = T.compute_angle_deviation();
        }
    }

    __global__ void compute_triangle_aabbs_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ aabb_mins,
        float3 *__restrict__ aabb_maxs)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            float3 v0 = vertices[tri.x];
            float3 v1 = vertices[tri.y];
            float3 v2 = vertices[tri.z];
            
            triangle::compute_aabb(v0, v1, v2, aabb_mins[idx], aabb_maxs[idx]);
        }
    }

    __host__ void compute_triangle_normals(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ triangle_normals)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_triangle_normals_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, triangle_normals);
    }

    __host__ void compute_triangle_areas(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ triangle_areas)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_triangle_areas_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, triangle_areas);
    }

    __host__ void compute_quality(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ qualities)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_quality_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, qualities);
    }

    __host__ void compute_aspect_ratio(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        int mode,
        float *__restrict__ aspect_ratios)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_aspect_ratio_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, mode, aspect_ratios);
    }

    __host__ void compute_radii_ratio(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ ratios)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_radii_ratio_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, ratios);
    }

    __host__ void compute_triangle_regularity(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ regularities)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_triangle_regularity_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, regularities);
    }

    __host__ void compute_radius_edge_ratio(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ ratios)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_radius_edge_ratio_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, ratios);
    }

    __host__ void compute_angle_deviation(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ deviations)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_angle_deviation_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, deviations);
    }

    __host__ void compute_triangle_aabbs(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ aabb_mins,
        float3 *__restrict__ aabb_maxs)
    {
        if (num_triangles == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_triangle_aabbs_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, aabb_mins, aabb_maxs);
    }

    __global__ void compute_vertex_normals_kernel(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ vertex_normals,
        int mode)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            float3 n = triangle_normals[idx];
            
            if (mode == 0) {
                atomicAdd(&vertex_normals[tri.x].x, n.x);
                atomicAdd(&vertex_normals[tri.x].y, n.y);
                atomicAdd(&vertex_normals[tri.x].z, n.z);
                
                atomicAdd(&vertex_normals[tri.y].x, n.x);
                atomicAdd(&vertex_normals[tri.y].y, n.y);
                atomicAdd(&vertex_normals[tri.y].z, n.z);
                
                atomicAdd(&vertex_normals[tri.z].x, n.x);
                atomicAdd(&vertex_normals[tri.z].y, n.y);
                atomicAdd(&vertex_normals[tri.z].z, n.z);
            } else if (mode == 1) {
                float3 v0 = vertices[tri.x];
                float3 v1 = vertices[tri.y];
                float3 v2 = vertices[tri.z];
                float3 e0 = maths::normalize(v1 - v0);
                float3 e1 = maths::normalize(v2 - v1);
                float3 e2 = maths::normalize(v0 - v2);

                float a0 = acosf(fminf(fmaxf(-maths::dot(e0, e2), -1.0f), 1.0f));
                float a1 = acosf(fminf(fmaxf(-maths::dot(e1, e0), -1.0f), 1.0f));
                float a2 = acosf(fminf(fmaxf(-maths::dot(e2, e1), -1.0f), 1.0f));

                if (isfinite(a0)) {
                    atomicAdd(&vertex_normals[tri.x].x, a0 * n.x);
                    atomicAdd(&vertex_normals[tri.x].y, a0 * n.y);
                    atomicAdd(&vertex_normals[tri.x].z, a0 * n.z);
                }
                if (isfinite(a1)) {
                    atomicAdd(&vertex_normals[tri.y].x, a1 * n.x);
                    atomicAdd(&vertex_normals[tri.y].y, a1 * n.y);
                    atomicAdd(&vertex_normals[tri.y].z, a1 * n.z);
                }
                if (isfinite(a2)) {
                    atomicAdd(&vertex_normals[tri.z].x, a2 * n.x);
                    atomicAdd(&vertex_normals[tri.z].y, a2 * n.y);
                    atomicAdd(&vertex_normals[tri.z].z, a2 * n.z);
                }
            }
        }
    }

    __global__ void normalize_vertex_normals_kernel(
        const uint32_t num_vertices,
        float3 *__restrict__ vertex_normals)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_vertices)
        {
            float3 n = vertex_normals[idx];
            float length = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
            if (length > 1e-8f) {
                vertex_normals[idx] = make_float3(n.x / length, n.y / length, n.z / length);
            }
        }
    }

    __host__ void compute_vertex_normals(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ vertex_normals,
        int mode)
    {
        if (num_triangles == 0 || num_vertices == 0) return;
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        
        compute_vertex_normals_kernel<<<blocks, threads>>>(
            num_triangles, vertices, triangles, triangle_normals, vertex_normals, mode);
            
        int blocks_vert = (num_vertices + threads - 1) / threads;
        normalize_vertex_normals_kernel<<<blocks_vert, threads>>>(
            num_vertices, vertex_normals);
    }

    __global__ void extract_edge_slots_kernel(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        Edge *__restrict__ edge_keys,
        int *__restrict__ edge_indices)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles)
        {
            int3 tri = triangles[idx];
            edge_keys[3 * idx + 0] = Edge(tri.x, tri.y);
            edge_indices[3 * idx + 0] = 3 * idx + 0;

            edge_keys[3 * idx + 1] = Edge(tri.y, tri.z);
            edge_indices[3 * idx + 1] = 3 * idx + 1;

            edge_keys[3 * idx + 2] = Edge(tri.z, tri.x);
            edge_indices[3 * idx + 2] = 3 * idx + 2;
        }
    }

    __global__ void compute_edge_normals_kernel(
        const uint32_t num_edges,
        const Edge *__restrict__ sorted_edge_keys,
        const int *__restrict__ sorted_edge_indices,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ edge_normals)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_edges)
        {
            Edge my_key = sorted_edge_keys[idx];
            int my_orig_idx = sorted_edge_indices[idx];

            float3 N = triangle_normals[my_orig_idx / 3];

            int j = idx - 1;
            while (j >= 0 && sorted_edge_keys[j] == my_key)
            {
                N = N + triangle_normals[sorted_edge_indices[j] / 3];
                j--;
            }

            j = idx + 1;
            while (j < num_edges && sorted_edge_keys[j] == my_key)
            {
                N = N + triangle_normals[sorted_edge_indices[j] / 3];
                j++;
            }

            edge_normals[my_orig_idx] = maths::normalize(N);
        }
    }

    __host__ void compute_edge_normals(
        const uint32_t num_triangles,
        const torch::Tensor &triangles,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ edge_normals)
    {
        if (num_triangles == 0) return;

        at::cuda::CUDAGuard device_guard(triangles.device());
        auto allocator = at::cuda::ThrustAllocator();
        auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());

        int threads = NTHREADS;
        int blocks_tri = (num_triangles + threads - 1) / threads;

        uint32_t num_edges = num_triangles * 3;
        int blocks_edge = (num_edges + threads - 1) / threads;

        auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(triangles.device());
        auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(triangles.device());

        torch::Tensor edge_keys = torch::empty({static_cast<int64_t>(num_edges)}, options_i64);
        torch::Tensor edge_indices = torch::empty({static_cast<int64_t>(num_edges)}, options_i32);

        extract_edge_slots_kernel<<<blocks_tri, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_triangles, (const int3*)triangles.data_ptr<int>(),
            reinterpret_cast<Edge *>(edge_keys.data_ptr<int64_t>()),
            edge_indices.data_ptr<int>());

        thrust::sort_by_key(
            policy,
            reinterpret_cast<Edge *>(edge_keys.data_ptr<int64_t>()),
            reinterpret_cast<Edge *>(edge_keys.data_ptr<int64_t>()) + num_edges,
            edge_indices.data_ptr<int>());

        compute_edge_normals_kernel<<<blocks_edge, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_edges,
            reinterpret_cast<const Edge *>(edge_keys.data_ptr<int64_t>()),
            edge_indices.data_ptr<int>(),
            triangle_normals,
            edge_normals);
    }

    __global__ void extract_edges_kernel(
        const uint32_t num_triangles,
        const int3* triangles,
        Edge* edge_keys,
        int* triangle_indices)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            
            edge_keys[3*idx + 0] = Edge(tri.x, tri.y);
            triangle_indices[3*idx + 0] = idx;
            
            edge_keys[3*idx + 1] = Edge(tri.y, tri.z);
            triangle_indices[3*idx + 1] = idx;
            
            edge_keys[3*idx + 2] = Edge(tri.z, tri.x);
            triangle_indices[3*idx + 2] = idx;
        }
    }
    
    __global__ void unpack_edges_kernel(
        const uint32_t num_unique_edges,
        const Edge* unique_edge_keys,
        int* unique_edges_out)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_unique_edges) {
            Edge key = unique_edge_keys[idx];
            unique_edges_out[2*idx + 0] = key.v0;
            unique_edges_out[2*idx + 1] = key.v1;
        }
    }

    __host__ void compute_edges_to_triangle_map(
        const uint32_t num_triangles,
        const torch::Tensor &triangles,
        torch::Tensor &out_unique_edges,
        torch::Tensor &out_offsets,
        torch::Tensor &out_counts,
        torch::Tensor &out_sorted_triangle_indices)
    {
        if (num_triangles == 0) return;

        at::cuda::CUDAGuard device_guard(triangles.device());
        auto allocator = at::cuda::ThrustAllocator();
        auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());

        auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(triangles.device());
        auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(triangles.device());

        uint32_t num_edges = num_triangles * 3;
        
        torch::Tensor edge_keys = torch::empty({num_edges}, options_i64);
        out_sorted_triangle_indices = torch::empty({num_edges}, options_i32);

        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        extract_edges_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_triangles, (const int3*)triangles.data_ptr<int>(), 
            (Edge*)edge_keys.data_ptr<int64_t>(), 
            out_sorted_triangle_indices.data_ptr<int>());

        thrust::sort_by_key(
            policy,
            (Edge*)edge_keys.data_ptr<int64_t>(),
            (Edge*)edge_keys.data_ptr<int64_t>() + num_edges,
            out_sorted_triangle_indices.data_ptr<int>()
        );

        torch::Tensor unique_keys = torch::empty({num_edges}, options_i64);
        out_counts = torch::empty({num_edges}, options_i32);
        torch::Tensor ones = torch::ones({num_edges}, options_i32);

        auto new_end = thrust::reduce_by_key(
            policy,
            (Edge*)edge_keys.data_ptr<int64_t>(),
            (Edge*)edge_keys.data_ptr<int64_t>() + num_edges,
            ones.data_ptr<int>(),
            (Edge*)unique_keys.data_ptr<int64_t>(),
            out_counts.data_ptr<int>()
        );

        int num_unique_edges = new_end.first - (Edge*)unique_keys.data_ptr<int64_t>();

        unique_keys = unique_keys.slice(0, 0, num_unique_edges);
        out_counts = out_counts.slice(0, 0, num_unique_edges);

        out_offsets = torch::empty({num_unique_edges}, options_i32);
        thrust::exclusive_scan(
            policy,
            out_counts.data_ptr<int>(),
            out_counts.data_ptr<int>() + num_unique_edges,
            out_offsets.data_ptr<int>()
        );

        out_unique_edges = torch::empty({num_unique_edges, 2}, options_i32);
        int blocks2 = (num_unique_edges + threads - 1) / threads;
        if (blocks2 > 0) {
            unpack_edges_kernel<<<blocks2, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_unique_edges,
                (Edge*)unique_keys.data_ptr<int64_t>(),
                out_unique_edges.data_ptr<int>()
            );
        }
    }

    __global__ void compute_vertex_triangle_counts_kernel(
        const uint32_t num_triangles,
        const int3* triangles,
        int* counts)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            atomicAdd(&counts[tri.x], 1);
            atomicAdd(&counts[tri.y], 1);
            atomicAdd(&counts[tri.z], 1);
        }
    }

    __global__ void compute_vertex_triangle_indices_kernel(
        const uint32_t num_triangles,
        const int3* triangles,
        int* current_offsets,
        int* indices)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            int pos_x = atomicAdd(&current_offsets[tri.x], 1);
            indices[pos_x] = idx;
            int pos_y = atomicAdd(&current_offsets[tri.y], 1);
            indices[pos_y] = idx;
            int pos_z = atomicAdd(&current_offsets[tri.z], 1);
            indices[pos_z] = idx;
        }
    }

    void build_vertices_to_triangle_map(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const torch::Tensor& triangles,
        torch::Tensor& out_counts,
        torch::Tensor& out_offsets,
        torch::Tensor& out_indices)
    {
        if (num_triangles == 0 || num_vertices == 0) return;

        at::cuda::CUDAGuard device_guard(triangles.device());
        auto allocator = at::cuda::ThrustAllocator();
        auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());

        auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(triangles.device());
        out_counts = torch::zeros({num_vertices}, options_i32);
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;

        compute_vertex_triangle_counts_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_triangles,
            (const int3*)triangles.data_ptr<int>(),
            out_counts.data_ptr<int>()
        );

        out_offsets = torch::empty({num_vertices}, options_i32);
        thrust::exclusive_scan(
            policy,
            out_counts.data_ptr<int>(),
            out_counts.data_ptr<int>() + num_vertices,
            out_offsets.data_ptr<int>()
        );

        // We need a temporary copy of offsets to use as sliding pointers
        torch::Tensor current_offsets = out_offsets.clone();
        out_indices = torch::empty({num_triangles * 3}, options_i32);

        compute_vertex_triangle_indices_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_triangles,
            (const int3*)triangles.data_ptr<int>(),
            current_offsets.data_ptr<int>(),
            out_indices.data_ptr<int>()
        );
    }

    __global__ void get_non_manifold_vertices_kernel(
        const uint32_t num_vertices,
        const int3* triangles,
        const int* v2t_offsets,
        const int* v2t_counts,
        const int* v2t_indices,
        bool* out_is_non_manifold)
    {
        uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
        if (v >= num_vertices) return;
        
        int count = v2t_counts[v];
        if (count == 0) {
            out_is_non_manifold[v] = false;
            return;
        }
        
        if (count > 64) {
            out_is_non_manifold[v] = true; // Fallback for safely avoiding overflow
            return;
        }
        
        int offset = v2t_offsets[v];
        int neighbors[128]; // max 64 triangles * 2
        
        for (int i = 0; i < count; ++i) {
            int3 t = triangles[v2t_indices[offset + i]];
            int n1 = -1, n2 = -1;
            if (t.x != v) { n1 = t.x; }
            if (t.y != v) { if (n1 == -1) n1 = t.y; else n2 = t.y; }
            if (t.z != v) { n2 = t.z; }
            neighbors[2*i + 0] = n1;
            neighbors[2*i + 1] = n2;
        }
        
        // 1. Check for bad edges (spoke shared by >2 triangles)
        for (int i = 0; i < count * 2; ++i) {
            int target = neighbors[i];
            int occurrences = 0;
            for (int j = 0; j < count * 2; ++j) {
                if (neighbors[j] == target) occurrences++;
            }
            if (occurrences > 2) {
                out_is_non_manifold[v] = true;
                return;
            }
        }
        
        // 2. Check for bowtie (disconnected components) via bitmask BFS
        unsigned long long visited = 1ULL;
        unsigned long long frontier = 1ULL;
        
        while (frontier != 0) {
            int current_idx = __ffsll(frontier) - 1;
            frontier &= ~(1ULL << current_idx);
            
            int n1_current = neighbors[2*current_idx + 0];
            int n2_current = neighbors[2*current_idx + 1];
            
            for (int i = 0; i < count; ++i) {
                if ((visited & (1ULL << i)) == 0) {
                    int n1_other = neighbors[2*i + 0];
                    int n2_other = neighbors[2*i + 1];
                    if (n1_current == n1_other || n1_current == n2_other ||
                        n2_current == n1_other || n2_current == n2_other) {
                        visited |= (1ULL << i);
                        frontier |= (1ULL << i);
                    }
                }
            }
        }
        
        unsigned long long expected_visited = (count == 64) ? ~0ULL : ((1ULL << count) - 1);
        if (visited != expected_visited) {
            out_is_non_manifold[v] = true;
        } else {
            out_is_non_manifold[v] = false;
        }
    }

    torch::Tensor get_non_manifold_vertices(
        const uint32_t num_vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& v2t_offsets,
        const torch::Tensor& v2t_counts,
        const torch::Tensor& v2t_indices)
    {
        auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(triangles.device());
        torch::Tensor out_is_non_manifold = torch::empty({num_vertices}, options_bool);
        
        if (num_vertices == 0) return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(triangles.device()));
        
        int threads = NTHREADS;
        int blocks = (num_vertices + threads - 1) / threads;
        
        get_non_manifold_vertices_kernel<<<blocks, threads>>>(
            num_vertices,
            (const int3*)triangles.data_ptr<int>(),
            v2t_offsets.data_ptr<int>(),
            v2t_counts.data_ptr<int>(),
            v2t_indices.data_ptr<int>(),
            out_is_non_manifold.data_ptr<bool>()
        );
        
        return torch::nonzero(out_is_non_manifold).squeeze(1);
    }
    __global__ void sample_points_triangle_mesh_kernel(
        const int num_points,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const int64_t *__restrict__ tri_indices,
        const float2 *__restrict__ r1_r2,
        const float3 *__restrict__ vertex_normals,
        const float3 *__restrict__ triangle_normals,
        const float3 *__restrict__ vertex_colors,
        float3 *__restrict__ out_points,
        float3 *__restrict__ out_normals,
        float3 *__restrict__ out_colors)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points)
            return;

        int tri_idx = tri_indices[idx];
        int3 tri = triangles[tri_idx];
        float2 r = r1_r2[idx];

        Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
        out_points[idx] = T.sample_point(r.x, r.y);

        float sqrt_r1 = sqrtf(r.x);
        float u = 1.0f - sqrt_r1;
        float v = r.y * sqrt_r1;
        float w = 1.0f - u - v;

        if (out_normals) {
            if (triangle_normals) {
                out_normals[idx] = triangle_normals[tri_idx];
            } else if (vertex_normals) {
                float3 n0 = vertex_normals[tri.x];
                float3 n1 = vertex_normals[tri.y];
                float3 n2 = vertex_normals[tri.z];
                float3 n = make_float3(
                    n0.x * u + n1.x * v + n2.x * w,
                    n0.y * u + n1.y * v + n2.y * w,
                    n0.z * u + n1.z * v + n2.z * w);
                
                float length = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
                if (length > 1e-8f) {
                    out_normals[idx] = make_float3(n.x / length, n.y / length, n.z / length);
                } else {
                    out_normals[idx] = make_float3(0.0f, 0.0f, 0.0f);
                }
            }
        }

        if (out_colors && vertex_colors) {
            float3 c0 = vertex_colors[tri.x];
            float3 c1 = vertex_colors[tri.y];
            float3 c2 = vertex_colors[tri.z];
            out_colors[idx] = make_float3(
                c0.x * u + c1.x * v + c2.x * w,
                c0.y * u + c1.y * v + c2.y * w,
                c0.z * u + c1.z * v + c2.z * w);
        }
    }

    __host__ void sample_points_triangle_mesh(
        const int num_points,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const int64_t *__restrict__ tri_indices,
        const float2 *__restrict__ r1_r2,
        const float3 *__restrict__ vertex_normals,
        const float3 *__restrict__ triangle_normals,
        const float3 *__restrict__ vertex_colors,
        float3 *__restrict__ out_points,
        float3 *__restrict__ out_normals,
        float3 *__restrict__ out_colors)
    {
        int threads = NTHREADS;
        int blocks = (num_points + threads - 1) / threads;

        sample_points_triangle_mesh_kernel<<<blocks, threads>>>(
            num_points,
            vertices,
            triangles,
            tri_indices,
            r1_r2,
            vertex_normals,
            triangle_normals,
            vertex_colors,
            out_points,
            out_normals,
            out_colors);
    }

    __global__ void compute_vertex_degree_kernel(
        const uint32_t num_unique_edges,
        const int *__restrict__ unique_edges,
        int *__restrict__ vertex_degrees)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_unique_edges) {
            int v0 = unique_edges[2 * idx];
            int v1 = unique_edges[2 * idx + 1];
            atomicAdd(&vertex_degrees[v0], 1);
            atomicAdd(&vertex_degrees[v1], 1);
        }
    }

    __host__ void compute_vertex_degree(
        const uint32_t num_unique_edges,
        const int *__restrict__ unique_edges,
        int *__restrict__ vertex_degrees)
    {
        if (num_unique_edges == 0) return;

        int threads = NTHREADS;
        int blocks = (num_unique_edges + threads - 1) / threads;

        compute_vertex_degree_kernel<<<blocks, threads>>>(
            num_unique_edges,
            unique_edges,
            vertex_degrees);
    }

    __global__ void compute_uniform_laplacian_kernel(
        const uint32_t num_unique_edges,
        const int *__restrict__ unique_edges,
        const float3 *__restrict__ vertices,
        float3 *__restrict__ vertex_lb_uniform)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_unique_edges) {
            int u = unique_edges[2 * idx];
            int v = unique_edges[2 * idx + 1];
            
            float3 pos_u = vertices[u];
            float3 pos_v = vertices[v];
            
            atomicAdd(&vertex_lb_uniform[u], pos_v - pos_u);
            atomicAdd(&vertex_lb_uniform[v], pos_u - pos_v);
        }
    }

    __global__ void normalize_uniform_laplacian_kernel(
        const uint32_t num_vertices,
        const int *__restrict__ vertex_degrees,
        float3 *__restrict__ vertex_lb_uniform)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_vertices) {
            int degree = vertex_degrees[idx];
            if (degree > 0) {
                vertex_lb_uniform[idx] /= degree;
            }
        }
    }

    __host__ void compute_uniform_laplacian(
        const uint32_t num_vertices,
        const uint32_t num_unique_edges,
        const int *__restrict__ unique_edges,
        const int *__restrict__ vertex_degrees,
        const float3 *__restrict__ vertices,
        float3 *__restrict__ vertex_lb_uniform)
    {
        if (num_unique_edges == 0 || num_vertices == 0) return;

        int threads = NTHREADS;
        int blocks = (num_unique_edges + threads - 1) / threads;

        compute_uniform_laplacian_kernel<<<blocks, threads>>>(
            num_unique_edges,
            unique_edges,
            vertices,
            vertex_lb_uniform);

        int blocks_vert = (num_vertices + threads - 1) / threads;
        normalize_uniform_laplacian_kernel<<<blocks_vert, threads>>>(
            num_vertices,
            vertex_degrees,
            vertex_lb_uniform);
    }

    __global__ void compute_voronoi_areas_kernel(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float *__restrict__ voronoi_areas)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            int v0 = tri.x;
            int v1 = tri.y;
            int v2 = tri.z;
            
            float3 p0 = vertices[v0];
            float3 p1 = vertices[v1];
            float3 p2 = vertices[v2];
            
            float area = triangle::compute_area(p0, p1, p2);
            float area0, area1, area2;
            
            if (triangle::is_obtuse(p0, p1, p2)) {
                float3 e01 = p1 - p0;
                float3 e12 = p2 - p1;
                float3 e20 = p0 - p2;
                
                float d0 = maths::dot(e01, -e20);
                float d1 = maths::dot(e12, -e01);
                
                if (d0 < 0.0f) {
                    area0 = area * 0.5f;
                    area1 = area * 0.25f;
                    area2 = area * 0.25f;
                } else if (d1 < 0.0f) {
                    area0 = area * 0.25f;
                    area1 = area * 0.5f;
                    area2 = area * 0.25f;
                } else {
                    area0 = area * 0.25f;
                    area1 = area * 0.25f;
                    area2 = area * 0.5f;
                }
            } else {
                float cot0, cot1, cot2;
                Triangle(p0, p1, p2).compute_cotangents(cot0, cot1, cot2);
                
                float3 e01 = p1 - p0;
                float3 e12 = p2 - p1;
                float3 e20 = p0 - p2;
                
                float l01 = maths::dot(e01, e01);
                float l12 = maths::dot(e12, e12);
                float l20 = maths::dot(e20, e20);
                
                area0 = 0.125f * (l01 * cot2 + l20 * cot1);
                area1 = 0.125f * (l01 * cot2 + l12 * cot0);
                area2 = 0.125f * (l20 * cot1 + l12 * cot0);
            }
            
            atomicAdd(&voronoi_areas[v0], area0);
            atomicAdd(&voronoi_areas[v1], area1);
            atomicAdd(&voronoi_areas[v2], area2);
        }
    }

    __host__ void compute_voronoi_areas(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float *__restrict__ voronoi_areas)
    {
        if (num_triangles == 0) return;
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        compute_voronoi_areas_kernel<<<blocks, threads>>>(
            num_triangles, triangles, vertices, voronoi_areas);
    }

    __global__ void compute_cotangent_laplacian_kernel(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float3 *__restrict__ vertex_lb_cot)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            int v0 = tri.x;
            int v1 = tri.y;
            int v2 = tri.z;
            
            float3 p0 = vertices[v0];
            float3 p1 = vertices[v1];
            float3 p2 = vertices[v2];
            
            float cot0, cot1, cot2;
            Triangle(p0, p1, p2).compute_cotangents(cot0, cot1, cot2);
            
            // Contribution to edge (v1, v2) from v0
            float3 w0 = cot0 * (p2 - p1);
            atomicAdd(&vertex_lb_cot[v1], w0);
            atomicAdd(&vertex_lb_cot[v2], -w0);
            
            // Contribution to edge (v2, v0) from v1
            float3 w1 = cot1 * (p0 - p2);
            atomicAdd(&vertex_lb_cot[v2], w1);
            atomicAdd(&vertex_lb_cot[v0], -w1);
            
            // Contribution to edge (v0, v1) from v2
            float3 w2 = cot2 * (p1 - p0);
            atomicAdd(&vertex_lb_cot[v0], w2);
            atomicAdd(&vertex_lb_cot[v1], -w2);
        }
    }

    __global__ void normalize_cotangent_laplacian_kernel(
        const uint32_t num_vertices,
        const float *__restrict__ voronoi_areas,
        float3 *__restrict__ vertex_lb_cot)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_vertices) {
            float area = voronoi_areas[idx];
            if (area > 1e-8f) {
                float inv_area = 1.0f / (2.0f * area);
                vertex_lb_cot[idx] *= inv_area;
            }
        }
    }

    __host__ void compute_cotangent_laplacian(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float *__restrict__ voronoi_areas,
        float3 *__restrict__ vertex_lb_cot)
    {
        if (num_triangles == 0 || num_vertices == 0) return;

        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;

        compute_cotangent_laplacian_kernel<<<blocks, threads>>>(
            num_triangles,
            triangles,
            vertices,
            vertex_lb_cot);

        int blocks_vert = (num_vertices + threads - 1) / threads;
        normalize_cotangent_laplacian_kernel<<<blocks_vert, threads>>>(
            num_vertices,
            voronoi_areas,
            vertex_lb_cot);
    }

    __global__ void compute_incident_angles_kernel(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        float *__restrict__ vertex_angle_sum)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_triangles) {
            int3 tri = triangles[idx];
            float a0, a1, a2;
            Triangle(vertices[tri.x], vertices[tri.y], vertices[tri.z]).compute_angles(a0, a1, a2);
            
            atomicAdd(&vertex_angle_sum[tri.x], a0);
            atomicAdd(&vertex_angle_sum[tri.y], a1);
            atomicAdd(&vertex_angle_sum[tri.z], a2);
        }
    }

    __global__ void finalize_gaussian_curvature_kernel(
        const uint32_t num_vertices,
        const float *__restrict__ voronoi_areas,
        const float *__restrict__ vertex_angle_sum,
        float *__restrict__ gaussian_curvature)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_vertices) {
            float area = voronoi_areas[idx];
            float angle_sum = vertex_angle_sum[idx];
            // 2.0f * M_PI = 6.28318530718f
            gaussian_curvature[idx] = (6.28318530718f - angle_sum) / area;
        }
    }

    __host__ void compute_gaussian_curvature(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ vertices,
        const float *__restrict__ voronoi_areas,
        float *__restrict__ vertex_angle_sum,
        float *__restrict__ gaussian_curvature)
    {
        if (num_triangles == 0 || num_vertices == 0) return;
        
        int threads = NTHREADS;
        int blocks_tri = (num_triangles + threads - 1) / threads;
        compute_incident_angles_kernel<<<blocks_tri, threads>>>(
            num_triangles, triangles, vertices, vertex_angle_sum);
            
        int blocks_vert = (num_vertices + threads - 1) / threads;
        finalize_gaussian_curvature_kernel<<<blocks_vert, threads>>>(
            num_vertices, voronoi_areas, vertex_angle_sum, gaussian_curvature);
    }
    __global__ void find_unvisited_kernel(
        const int num_triangles,
        const int *__restrict__ visited,
        int *__restrict__ seed,
        int *__restrict__ found)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_triangles) return;
        
        if (visited[idx] == 0) {
            if (atomicCAS(found, 0, 1) == 0) {
                *seed = idx;
            }
        }
    }

    __global__ void fix_winding_bfs_kernel(
        const int num_triangles,
        int3 *__restrict__ triangles,
        const int *__restrict__ v2t_offsets,
        const int *__restrict__ v2t_counts,
        const int *__restrict__ v2t_indices,
        int *__restrict__ visited,
        const int *__restrict__ frontier,
        const int frontier_size,
        int *__restrict__ next_frontier,
        int *__restrict__ next_frontier_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= frontier_size) return;

        int tri_id = frontier[idx];
        int3 tri = triangles[tri_id];

        int edge_v[3][2] = {{tri.x, tri.y}, {tri.y, tri.z}, {tri.z, tri.x}};
        
        for (int e = 0; e < 3; ++e) {
            int v0 = edge_v[e][0];
            int v1 = edge_v[e][1];
            
            int shared_faces[10];
            int num_shared = 0;
            
            int start0 = v2t_offsets[v0];
            int end0 = start0 + v2t_counts[v0];
            int start1 = v2t_offsets[v1];
            int end1 = start1 + v2t_counts[v1];
            for (int i = start0; i < end0; ++i) {
                int f0 = v2t_indices[i];
                for (int j = start1; j < end1; ++j) {
                    if (f0 == v2t_indices[j]) {
                        if (num_shared < 10) {
                            shared_faces[num_shared++] = f0;
                        }
                    }
                }
            }
            
            for (int k = 0; k < num_shared; ++k) {
                int neighbor_id = shared_faces[k];
                if (neighbor_id != tri_id) {
                    if (atomicCAS(&visited[neighbor_id], 0, 1) == 0) {
                        int3 n_tri = triangles[neighbor_id];
                        bool n_has_v0_v1 = (n_tri.x == v0 && n_tri.y == v1) || 
                                           (n_tri.y == v0 && n_tri.z == v1) || 
                                           (n_tri.z == v0 && n_tri.x == v1);
                        if (n_has_v0_v1) {
                            triangles[neighbor_id] = make_int3(n_tri.x, n_tri.z, n_tri.y);
                        }
                        int push_idx = atomicAdd(next_frontier_size, 1);
                        next_frontier[push_idx] = neighbor_id;
                    }
                }
            }
        }
    }

    __global__ void component_signed_volume_kernel(
        const int num_component_faces,
        const int *__restrict__ component_faces,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ volumes)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_component_faces) return;
        
        int tri_id = component_faces[idx];
        int3 tri = triangles[tri_id];
        float3 a = vertices[tri.x];
        float3 b = vertices[tri.y];
        float3 c = vertices[tri.z];
        
        float3 cross_bc = maths::cross(b, c);
        float vol = maths::dot(a, cross_bc) / 6.0f;
        volumes[idx] = vol;
    }

    __global__ void invert_component_kernel(
        const int num_component_faces,
        const int *__restrict__ component_faces,
        int3 *__restrict__ triangles)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_component_faces) return;
        
        int tri_id = component_faces[idx];
        int3 tri = triangles[tri_id];
        triangles[tri_id] = make_int3(tri.x, tri.z, tri.y);
    }

    __host__ void fix_normals(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const torch::Tensor &v2t_offsets,
        const torch::Tensor &v2t_counts,
        const torch::Tensor &v2t_indices,
        int3 *__restrict__ triangles)
    {
        auto options = torch::TensorOptions().device(v2t_offsets.device()).dtype(torch::kInt32);
        torch::Tensor visited = torch::zeros({num_triangles}, options);
        torch::Tensor frontier = torch::empty({num_triangles}, options);
        torch::Tensor next_frontier = torch::empty({num_triangles}, options);
        torch::Tensor component_faces = torch::empty({num_triangles}, options);
        
        int *d_visited = visited.data_ptr<int>();
        int *d_frontier = frontier.data_ptr<int>();
        int *d_next_frontier = next_frontier.data_ptr<int>();
        int *d_component_faces = component_faces.data_ptr<int>();
        
        int *d_seed; cudaMalloc(&d_seed, sizeof(int));
        int *d_found; cudaMalloc(&d_found, sizeof(int));
        int *d_next_frontier_size; cudaMalloc(&d_next_frontier_size, sizeof(int));
        
        int h_found = 0;
        int h_seed = 0;
        
        while (true) {
            cudaMemset(d_found, 0, sizeof(int));
            int blocks = (num_triangles + NTHREADS - 1) / NTHREADS;
            find_unvisited_kernel<<<blocks, NTHREADS>>>(num_triangles, d_visited, d_seed, d_found);
            cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
            
            if (h_found == 0) break;
            
            cudaMemcpy(&h_seed, d_seed, sizeof(int), cudaMemcpyDeviceToHost);
            
            int h_one = 1;
            cudaMemcpy(&d_visited[h_seed], &h_one, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(&d_frontier[0], &h_seed, sizeof(int), cudaMemcpyHostToDevice);
            
            int frontier_size = 1;
            int component_size = 0;
            
            while (frontier_size > 0) {
                cudaMemcpy(&d_component_faces[component_size], d_frontier, frontier_size * sizeof(int), cudaMemcpyDeviceToDevice);
                component_size += frontier_size;
                
                cudaMemset(d_next_frontier_size, 0, sizeof(int));
                
                int bfs_blocks = (frontier_size + NTHREADS - 1) / NTHREADS;
                fix_winding_bfs_kernel<<<bfs_blocks, NTHREADS>>>(
                    num_triangles, triangles, 
                    v2t_offsets.data_ptr<int>(), v2t_counts.data_ptr<int>(), v2t_indices.data_ptr<int>(),
                    d_visited, d_frontier, frontier_size, d_next_frontier, d_next_frontier_size);
                    
                cudaMemcpy(&frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
                
                int *tmp = d_frontier;
                d_frontier = d_next_frontier;
                d_next_frontier = tmp;
            }
            
            auto vol_options = torch::TensorOptions().device(torch::kCUDA, ::c10::cuda::current_device()).dtype(torch::kFloat32);
            torch::Tensor volumes = torch::empty({component_size}, vol_options);
            
            int vol_blocks = (component_size + NTHREADS - 1) / NTHREADS;
            component_signed_volume_kernel<<<vol_blocks, NTHREADS>>>(
                component_size, d_component_faces, vertices, triangles, volumes.data_ptr<float>());
                
            float comp_volume = volumes.sum().item<float>();
            
            if (comp_volume < 0.0f) {
                invert_component_kernel<<<vol_blocks, NTHREADS>>>(
                    component_size, d_component_faces, triangles);
            }
        }
        
        cudaFree(d_seed);
        cudaFree(d_found);
        cudaFree(d_next_frontier_size);
    }
}
