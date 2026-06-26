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

    Triangle T1(vertices[triA.x], vertices[triA.y], vertices[triA.z]);
    Triangle T2(vertices[triB.x], vertices[triB.y], vertices[triB.z]);

    if (triangle::test_intersection(T1, T2)) {
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

    int64_t h_valid_counter = valid_counter.item<int64_t>();

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

__global__ void filter_ray_triangle_intersections_kernel(
    const int num_pairs,
    const int64_t* query_ids,
    const int64_t* object_ids,
    const float3* ray_origins,
    const float3* ray_dirs,
    const float3* vertices,
    const int3* triangles,
    int64_t* out_query_ids,
    int64_t* out_object_ids,
    float3* out_intersect_pts,
    float* out_distances,
    bool return_distance,
    int64_t* valid_counter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int64_t ray_id = query_ids[idx];
    int64_t tri_id = object_ids[idx];

    Ray ray(ray_origins[ray_id], ray_dirs[ray_id]);
    
    int3 tri = triangles[tri_id];
    Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);

    float t_hit, u, v;
    if (T.is_intersect_ray(ray, t_hit, u, v)) {
        uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int*)valid_counter, 1ULL);
        out_query_ids[write_idx] = ray_id;
        out_object_ids[write_idx] = tri_id;
        out_intersect_pts[write_idx] = ray.at(t_hit);
        if (return_distance) {
            out_distances[write_idx] = t_hit;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MeshBVH::get_ray_intersection(
    const torch::Tensor &ray_origins,
    const torch::Tensor &ray_dirs,
    const torch::Tensor &vertices,
    const torch::Tensor &triangles,
    bool return_distance)
{
    // 1. Broad-phase BVH query
    auto [query_ids, object_ids] = this->query_ray(ray_origins, ray_dirs);
    
    int num_pairs = query_ids.size(0);
    auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(query_ids.device());
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(query_ids.device());

    if (num_pairs == 0) {
        return std::make_tuple(
            torch::empty({0}, options_i64),
            torch::empty({0}, options_i64),
            torch::empty({0, 3}, options_f32),
            torch::empty({0}, options_f32)
        );
    }

    // 2. Narrow-phase intersection
    torch::Tensor out_query_ids = torch::empty({num_pairs}, options_i64);
    torch::Tensor out_object_ids = torch::empty({num_pairs}, options_i64);
    torch::Tensor out_intersect_pts = torch::empty({num_pairs, 3}, options_f32);
    
    torch::Tensor out_distances;
    if (return_distance) {
        out_distances = torch::empty({num_pairs}, options_f32);
    } else {
        out_distances = torch::empty({0}, options_f32);
    }
    
    torch::Tensor valid_counter = torch::zeros({1}, options_i64);

    int threads = NTHREADS;
    int blocks = (num_pairs + threads - 1) / threads;

    filter_ray_triangle_intersections_kernel<<<blocks, threads>>>(
        num_pairs,
        query_ids.data_ptr<int64_t>(),
        object_ids.data_ptr<int64_t>(),
        (const float3*)ray_origins.data_ptr<float>(),
        (const float3*)ray_dirs.data_ptr<float>(),
        (const float3*)vertices.data_ptr<float>(),
        (const int3*)triangles.data_ptr<int>(),
        out_query_ids.data_ptr<int64_t>(),
        out_object_ids.data_ptr<int64_t>(),
        (float3*)out_intersect_pts.data_ptr<float>(),
        return_distance ? out_distances.data_ptr<float>() : nullptr,
        return_distance,
        valid_counter.data_ptr<int64_t>()
    );

    int64_t h_valid_counter = valid_counter.item<int64_t>();

    if (h_valid_counter == 0) {
        return std::make_tuple(
            torch::empty({0}, options_i64),
            torch::empty({0}, options_i64),
            torch::empty({0, 3}, options_f32),
            torch::empty({0}, options_f32)
        );
    }

    return std::make_tuple(
        out_query_ids.slice(0, 0, h_valid_counter),
        out_object_ids.slice(0, 0, h_valid_counter),
        out_intersect_pts.slice(0, 0, h_valid_counter),
        return_distance ? out_distances.slice(0, 0, h_valid_counter) : out_distances
    );
}
