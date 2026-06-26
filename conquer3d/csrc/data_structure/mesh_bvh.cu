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

__device__ __forceinline__ float compute_sign_ray_parity(
    const float3& p,
    const int best_tri_id,
    const float3* bvh_aabb_mins,
    const float3* bvh_aabb_maxs,
    const int2* bvh_children,
    const int* object_ids,
    const float3* vertices,
    const int3* triangles,
    const uint32_t num_objects)
{
    int3 tri = triangles[best_tri_id];
    float3 v0 = vertices[tri.x];
    float3 v1 = vertices[tri.y];
    float3 v2 = vertices[tri.z];
    float3 centroid = (v0 + v1 + v2) / 3.0f;

    float3 ray_dir = centroid - p;
    float dir_len = maths::norm(ray_dir);
    if (dir_len > 1e-6f) {
        ray_dir = ray_dir / dir_len;
    } else {
        ray_dir = make_float3(1.0f, 0.0f, 0.0f);
    }

    Ray ray2(p, ray_dir, 0.0f);
    
    int hit_count = 0;
    int stack[BVH_STACK_SIZE];
    int stack_ptr = 0;
    stack[0] = 0;
    
    while (stack_ptr >= 0) 
    {
        int node_idx = stack[stack_ptr--];
        
        float t_hit_aabb;
        if (!ray2.is_intersect_aabb(bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx], t_hit_aabb)) {
            continue;
        }

        if (node_idx >= num_objects - 1) 
        {
            int tri_id = object_ids[node_idx - (num_objects - 1)];
            int3 tri = triangles[tri_id];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            
            float t_hit, u, v;
            if (T.is_intersect_ray(ray2, t_hit, u, v)) {
                hit_count++;
            }
        } 
        else 
        {
            if (stack_ptr + 2 < BVH_STACK_SIZE)
            {
                int2 children = bvh_children[node_idx];
                stack[++stack_ptr] = children.x;
                stack[++stack_ptr] = children.y;
            }
        }
    }
    
    return (hit_count % 2 == 1) ? -1.0f : 1.0f;
}

__global__ void query_point_mesh_bvh_kernel(
    const uint32_t num_queries,
    const uint32_t num_objects,
    const float3* __restrict__ query_points,
    const float3* __restrict__ vertices,
    const int3* __restrict__ triangles,
    const float3* __restrict__ bvh_aabb_mins,
    const float3* __restrict__ bvh_aabb_maxs,
    const int2* __restrict__ bvh_children,
    const int* __restrict__ object_ids,
    int64_t* __restrict__ out_query_ids,
    int64_t* __restrict__ out_object_ids,
    float3* __restrict__ out_projected_pts,
    float* __restrict__ out_distances,
    bool return_sdf,
    bool return_prj_pts,
    int sign_mode)
{
    uint32_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= num_queries) return;

    float3 p = query_points[q_idx];

    int stack[BVH_STACK_SIZE];
    int stack_ptr = 0;
    stack[0] = 0; 
    
    float best_dist_sq = FLT_MAX;
    int best_tri_id = -1;
    float3 best_pt = make_float3(0, 0, 0);

    while (stack_ptr >= 0) 
    {
        int node_idx = stack[stack_ptr--]; 

        float dist_sq = aabb::compute_squared_distance(p, bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx]);
        
        if (dist_sq > best_dist_sq) continue; // Prune branch

        if (node_idx >= num_objects - 1) 
        {
            int tri_id = object_ids[node_idx - (num_objects - 1)];
            int3 tri = triangles[tri_id];
            Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);
            
            float3 closest_pt = T.compute_closest_point(p);
            float3 diff = p - closest_pt;
            float pt_dist_sq = maths::dot(diff, diff);

            if (pt_dist_sq < best_dist_sq) {
                best_dist_sq = pt_dist_sq;
                best_tri_id = tri_id;
                best_pt = closest_pt;
            }
        } 
        else 
        {
            if (stack_ptr + 2 < BVH_STACK_SIZE)
            {
                int2 children = bvh_children[node_idx];
                
                float dist_l = aabb::compute_squared_distance(p, bvh_aabb_mins[children.x], bvh_aabb_maxs[children.x]);
                float dist_r = aabb::compute_squared_distance(p, bvh_aabb_mins[children.y], bvh_aabb_maxs[children.y]);
                
                if (dist_l > dist_r) {
                    stack[++stack_ptr] = children.x; 
                    stack[++stack_ptr] = children.y;
                } else {
                    stack[++stack_ptr] = children.y;
                    stack[++stack_ptr] = children.x;
                }
            }
        }
    }
    
    float dist = sqrtf(best_dist_sq);

    if (return_sdf && sign_mode == 0) {
        dist *= compute_sign_ray_parity(p, best_tri_id, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects);
    }

    out_query_ids[q_idx] = q_idx;
    out_object_ids[q_idx] = best_tri_id;
    if (return_prj_pts && out_projected_pts != nullptr) {
        out_projected_pts[q_idx] = best_pt;
    }
    out_distances[q_idx] = dist;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MeshBVH::query_point(
    const torch::Tensor &query_points,
    const torch::Tensor &vertices,
    const torch::Tensor &triangles,
    bool return_sdf,
    bool return_prj_pts,
    int sign_mode)
{
    if (sign_mode != 0) {
        throw std::runtime_error("sign_mode != 0 not implemented yet");
    }

    int num_queries = query_points.size(0);
    int num_objects = this->object_ids.size(0);

    auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(query_points.device());
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(query_points.device());

    torch::Tensor out_query_ids = torch::empty({num_queries}, options_i64);
    torch::Tensor out_object_ids = torch::empty({num_queries}, options_i64);
    torch::Tensor out_projected_pts;
    if (return_prj_pts) {
        out_projected_pts = torch::empty({num_queries, 3}, options_f32);
    } else {
        out_projected_pts = torch::empty({0, 3}, options_f32);
    }
    torch::Tensor out_distances = torch::empty({num_queries}, options_f32);

    int threads = NTHREADS;
    int blocks = (num_queries + threads - 1) / threads;

    query_point_mesh_bvh_kernel<<<blocks, threads>>>(
        num_queries,
        num_objects,
        (const float3*)query_points.data_ptr<float>(),
        (const float3*)vertices.data_ptr<float>(),
        (const int3*)triangles.data_ptr<int>(),
        (const float3*)this->aabb_mins.data_ptr<float>(),
        (const float3*)this->aabb_maxs.data_ptr<float>(),
        (const int2*)this->bvh_children.data_ptr<int>(),
        this->object_ids.data_ptr<int>(),
        out_query_ids.data_ptr<int64_t>(),
        out_object_ids.data_ptr<int64_t>(),
        return_prj_pts ? (float3*)out_projected_pts.data_ptr<float>() : nullptr,
        out_distances.data_ptr<float>(),
        return_sdf,
        return_prj_pts,
        sign_mode
    );

    return std::make_tuple(out_query_ids, out_object_ids, out_projected_pts, out_distances);
}
