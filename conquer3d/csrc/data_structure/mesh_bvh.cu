#include "mesh_bvh.h"
#include "../primitive/triangles.h"
#include <thrust/device_vector.h>


__global__ void filter_self_intersections_kernel(
    const int num_pairs,
    const int64_t* query_ids,
    const int64_t* object_ids,
    const float3* vertices,
    const int3* triangles,
    int64_t* out_query_ids,
    int64_t* out_object_ids,
    int64_t* valid_counter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int64_t idA = query_ids[idx];
    int64_t idB = object_ids[idx];

    int3 triA = triangles[idA];
    int3 triB = triangles[idB];

    // Filter shared vertices
    if (triA.x == triB.x || triA.x == triB.y || triA.x == triB.z ||
        triA.y == triB.x || triA.y == triB.y || triA.y == triB.z ||
        triA.z == triB.x || triA.z == triB.y || triA.z == triB.z) {
        return;
    }

    float3 v0 = vertices[triA.x];
    float3 v1 = vertices[triA.y];
    float3 v2 = vertices[triA.z];

    float3 u0 = vertices[triB.x];
    float3 u1 = vertices[triB.y];
    float3 u2 = vertices[triB.z];

    if (triangle::test_intersection(v0, v1, v2, u0, u1, u2)) {
        uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int*)valid_counter, 1ULL);
        out_query_ids[write_idx] = idA;
        out_object_ids[write_idx] = idB;
    }
}

torch::Tensor MeshBVH::get_self_intersection(
    const torch::Tensor &vertices,
    const torch::Tensor &triangles)
{
    // 1. Broad-phase AABB overlap query
    auto [query_ids, object_ids] = this->query_self();
    
    int num_pairs = query_ids.size(0);
    if (num_pairs == 0) {
        return torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kInt64).device(query_ids.device()));
    }

    // 2. Narrow-phase intersection
    auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(query_ids.device());
    torch::Tensor out_query_ids = torch::empty({num_pairs}, options_i64);
    torch::Tensor out_object_ids = torch::empty({num_pairs}, options_i64);
    torch::Tensor valid_counter = torch::zeros({1}, options_i64);

    int threads = NTHREADS;
    int blocks = (num_pairs + threads - 1) / threads;

    filter_self_intersections_kernel<<<blocks, threads>>>(
        num_pairs,
        query_ids.data_ptr<int64_t>(),
        object_ids.data_ptr<int64_t>(),
        (const float3*)vertices.data_ptr<float>(),
        (const int3*)triangles.data_ptr<int>(),
        out_query_ids.data_ptr<int64_t>(),
        out_object_ids.data_ptr<int64_t>(),
        valid_counter.data_ptr<int64_t>()
    );

    int64_t h_valid_counter = 0;
    cudaMemcpy(&h_valid_counter, valid_counter.data_ptr<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (h_valid_counter == 0) {
        return torch::empty({0, 2}, options_i64);
    }

    out_query_ids = out_query_ids.slice(0, 0, h_valid_counter).unsqueeze(1);
    out_object_ids = out_object_ids.slice(0, 0, h_valid_counter).unsqueeze(1);

    return torch::cat({out_query_ids, out_object_ids}, 1);
}

bool MeshBVH::is_self_intersection(
    const torch::Tensor &vertices,
    const torch::Tensor &triangles)
{
    torch::Tensor pairs = this->get_self_intersection(vertices, triangles);
    return pairs.size(0) > 0;
}
