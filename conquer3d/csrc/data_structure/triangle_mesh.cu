#include "triangle_mesh.h"
#include "../primitive/triangles.h"
#include "../primitive/edge.h"
#include <cuda_runtime.h>
#include <cuda_runtime.h>
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
            
            triangle_normals[idx] = compute_normal(v0, v1, v2);
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
            
            triangle_areas[idx] = compute_area(v0, v1, v2);
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
        const int3 *__restrict__ triangles,
        torch::Tensor &out_unique_edges,
        torch::Tensor &out_offsets,
        torch::Tensor &out_counts,
        torch::Tensor &out_sorted_triangle_indices)
    {
        if (num_triangles == 0) return;

        auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

        uint32_t num_edges = num_triangles * 3;
        
        torch::Tensor edge_keys = torch::empty({num_edges}, options_i64);
        out_sorted_triangle_indices = torch::empty({num_edges}, options_i32);

        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;
        extract_edges_kernel<<<blocks, threads>>>(
            num_triangles, triangles, 
            (Edge*)edge_keys.data_ptr<int64_t>(), 
            out_sorted_triangle_indices.data_ptr<int>());

        thrust::sort_by_key(
            thrust::device,
            (Edge*)edge_keys.data_ptr<int64_t>(),
            (Edge*)edge_keys.data_ptr<int64_t>() + num_edges,
            out_sorted_triangle_indices.data_ptr<int>()
        );

        torch::Tensor unique_keys = torch::empty({num_edges}, options_i64);
        out_counts = torch::empty({num_edges}, options_i32);
        torch::Tensor ones = torch::ones({num_edges}, options_i32);

        auto new_end = thrust::reduce_by_key(
            thrust::device,
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
            thrust::device,
            out_counts.data_ptr<int>(),
            out_counts.data_ptr<int>() + num_unique_edges,
            out_offsets.data_ptr<int>()
        );

        out_unique_edges = torch::empty({num_unique_edges, 2}, options_i32);
        int blocks2 = (num_unique_edges + threads - 1) / threads;
        if (blocks2 > 0) {
            unpack_edges_kernel<<<blocks2, threads>>>(
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

        auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(triangles.device());
        out_counts = torch::zeros({num_vertices}, options_i32);
        
        int threads = NTHREADS;
        int blocks = (num_triangles + threads - 1) / threads;

        compute_vertex_triangle_counts_kernel<<<blocks, threads>>>(
            num_triangles,
            (const int3*)triangles.data_ptr<int>(),
            out_counts.data_ptr<int>()
        );

        out_offsets = torch::empty({num_vertices}, options_i32);
        thrust::exclusive_scan(
            thrust::device,
            out_counts.data_ptr<int>(),
            out_counts.data_ptr<int>() + num_vertices,
            out_offsets.data_ptr<int>()
        );

        // We need a temporary copy of offsets to use as sliding pointers
        torch::Tensor current_offsets = out_offsets.clone();
        out_indices = torch::empty({num_triangles * 3}, options_i32);

        compute_vertex_triangle_indices_kernel<<<blocks, threads>>>(
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

    torch::Tensor get_non_manifold_vertices_cuda(
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
}
