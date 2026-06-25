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
}
