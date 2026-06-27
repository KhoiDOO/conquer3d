#include "mesh_bvh.h"
#include "../primitive/triangles.h"
#include <thrust/device_vector.h>

namespace mesh_bvh
{
    __global__ void filter_self_intersections_kernel(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        int64_t *valid_counter)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_pairs)
            return;

        int64_t idA = query_ids[idx];
        int64_t idB = object_ids[idx];

        int3 triA = triangles[idA];
        int3 triB = triangles[idB];

        // Filter shared vertices
        if (triA.x == triB.x || triA.x == triB.y || triA.x == triB.z ||
            triA.y == triB.x || triA.y == triB.y || triA.y == triB.z ||
            triA.z == triB.x || triA.z == triB.y || triA.z == triB.z)
        {
            return;
        }

        Triangle T1(vertices[triA.x], vertices[triA.y], vertices[triA.z]);
        Triangle T2(vertices[triB.x], vertices[triB.y], vertices[triB.z]);

        if (triangle::test_intersection(T1, T2))
        {
            uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int *)valid_counter, 1ULL);
            out_query_ids[write_idx] = idA;
            out_object_ids[write_idx] = idB;
        }
    }

    void filter_self_intersections(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        int64_t *valid_counter)
    {
        int threads = NTHREADS;
        int blocks = (num_pairs + threads - 1) / threads;

        filter_self_intersections_kernel<<<blocks, threads>>>(
            num_pairs,
            query_ids,
            object_ids,
            vertices,
            triangles,
            out_query_ids,
            out_object_ids,
            valid_counter);
    }

    __global__ void filter_ray_triangle_intersections_kernel(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *ray_origins,
        const float3 *ray_dirs,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        float3 *out_intersect_pts,
        float *out_distances,
        bool return_distance,
        int64_t *valid_counter)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_pairs)
            return;

        int64_t ray_id = query_ids[idx];
        int64_t tri_id = object_ids[idx];

        Ray ray(ray_origins[ray_id], ray_dirs[ray_id]);

        int3 tri = triangles[tri_id];
        Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);

        float t_hit, u, v;
        if (T.is_intersect_ray(ray, t_hit, u, v))
        {
            uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int *)valid_counter, 1ULL);
            out_query_ids[write_idx] = ray_id;
            out_object_ids[write_idx] = tri_id;
            out_intersect_pts[write_idx] = ray.at(t_hit);
            if (return_distance)
            {
                out_distances[write_idx] = t_hit;
            }
        }
    }

    void filter_ray_triangle_intersections(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *ray_origins,
        const float3 *ray_dirs,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        float3 *out_intersect_pts,
        float *out_distances,
        bool return_distance,
        int64_t *valid_counter)
    {
        int threads = NTHREADS;
        int blocks = (num_pairs + threads - 1) / threads;

        filter_ray_triangle_intersections_kernel<<<blocks, threads>>>(
            num_pairs,
            query_ids,
            object_ids,
            ray_origins,
            ray_dirs,
            vertices,
            triangles,
            out_query_ids,
            out_object_ids,
            out_intersect_pts,
            return_distance ? out_distances : nullptr,
            return_distance,
            valid_counter);
    }

    __device__ __forceinline__ float compute_sign_ray_parity(
        const float3 &p,
        const int best_tri_id,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        const uint32_t num_objects)
    {
        int3 tri = triangles[best_tri_id];
        float3 v0 = vertices[tri.x];
        float3 v1 = vertices[tri.y];
        float3 v2 = vertices[tri.z];
        float3 centroid = (v0 + v1 + v2) / 3.0f;

        float3 ray_dir = centroid - p;
        float dir_len = maths::norm(ray_dir);
        if (dir_len > 1e-6f)
        {
            ray_dir = ray_dir / dir_len;
        }
        else
        {
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
            if (!ray2.is_intersect_aabb(bvh_aabb_mins[node_idx], bvh_aabb_maxs[node_idx], t_hit_aabb))
            {
                continue;
            }

            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);

                float t_hit, u, v;
                if (T.is_intersect_ray(ray2, t_hit, u, v))
                {
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

    __device__ __forceinline__ float compute_solid_angle_tri(float3 p, float3 a, float3 b, float3 c) {
        float3 a_v = a - p;
        float3 b_v = b - p;
        float3 c_v = c - p;
        float a_len = sqrtf(maths::dot2(a_v));
        float b_len = sqrtf(maths::dot2(b_v));
        float c_len = sqrtf(maths::dot2(c_v));
        
        float num = maths::dot(a_v, maths::cross(b_v, c_v));
        float den = a_len * b_len * c_len + maths::dot(a_v, b_v) * c_len + maths::dot(b_v, c_v) * a_len + maths::dot(c_v, a_v) * b_len;
        
        return 2.0f * atan2f(num, den);
    }

    __device__ __forceinline__ float compute_fast_winding_number(
        const float3 &p,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        const WindingData *winding_data,
        const uint32_t num_objects,
        const float accuracy_scale)
    {
        float total_omega = 0.0f;
        float accuracy_scale2 = accuracy_scale * accuracy_scale;

        int stack[BVH_STACK_SIZE];
        int stack_ptr = 0;
        stack[0] = 0;

        while (stack_ptr >= 0)
        {
            int node_idx = stack[stack_ptr--];

            WindingData data = winding_data[node_idx];
            float3 q = p - data.average_p;
            float qlength2 = maths::dot2(q);

            if (qlength2 > accuracy_scale2 * data.max_p_dist2)
            {
                if (qlength2 > 1e-6f) {
                    float qlength = sqrtf(qlength2);
                    float qlength3 = qlength2 * qlength;
                    float Omega_approx = -maths::dot(q, data.n) / qlength3;
                    total_omega += Omega_approx;
                }
            }
            else
            {
                if (node_idx >= num_objects - 1)
                {
                    int tri_id = object_ids[node_idx - (num_objects - 1)];
                    int3 tri = triangles[tri_id];
                    float3 v0 = vertices[tri.x];
                    float3 v1 = vertices[tri.y];
                    float3 v2 = vertices[tri.z];
                    
                    total_omega += compute_solid_angle_tri(p, v0, v1, v2);
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
        }

        return total_omega * 0.07957747154f; // 1.0f / (4.0f * PI)
    }

    __global__ void query_point_mesh_bvh_kernel(
        const int num_queries,
        const int num_objects,
        const float3 *__restrict__ query_points,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const WindingData *__restrict__ winding_data,
        int64_t *__restrict__ out_query_ids,
        int64_t *__restrict__ out_object_ids,
        float3 *__restrict__ out_projected_pts,
        float *__restrict__ out_distances,
        bool return_sdf,
        bool return_prj_pts,
        int sign_mode)
    {
        int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (q_idx >= num_queries)
            return;

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

            if (dist_sq > best_dist_sq)
                continue; // Prune branch

            if (node_idx >= num_objects - 1)
            {
                int tri_id = object_ids[node_idx - (num_objects - 1)];
                int3 tri = triangles[tri_id];
                Triangle T(vertices[tri.x], vertices[tri.y], vertices[tri.z]);

                float3 closest_pt = T.compute_closest_point(p);
                float3 diff = p - closest_pt;
                float pt_dist_sq = maths::dot(diff, diff);

                if (pt_dist_sq < best_dist_sq)
                {
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

                    if (dist_l > dist_r)
                    {
                        stack[++stack_ptr] = children.x;
                        stack[++stack_ptr] = children.y;
                    }
                    else
                    {
                        stack[++stack_ptr] = children.y;
                        stack[++stack_ptr] = children.x;
                    }
                }
            }
        }

        float dist = sqrtf(best_dist_sq);

        if (return_sdf)
        {
            if (sign_mode == 0) {
                dist *= compute_sign_ray_parity(p, best_tri_id, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, num_objects);
            } else if (sign_mode == 1) {
                float wn = compute_fast_winding_number(p, bvh_aabb_mins, bvh_aabb_maxs, bvh_children, object_ids, vertices, triangles, winding_data, num_objects, 2.0f);
                dist *= (wn >= 0.5f) ? -1.0f : 1.0f;
            }
        }

        out_query_ids[q_idx] = q_idx;
        out_object_ids[q_idx] = best_tri_id;
        if (return_prj_pts && out_projected_pts != nullptr)
        {
            out_projected_pts[q_idx] = best_pt;
        }
        out_distances[q_idx] = dist;
    }

    void query_point_mesh_bvh(const int num_queries,
        const int num_objects,
        const float3 *__restrict__ query_points,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const WindingData *__restrict__ winding_data,
        int64_t *__restrict__ out_query_ids,
        int64_t *__restrict__ out_object_ids,
        float3 *__restrict__ out_projected_pts,
        float *__restrict__ out_distances,
        bool return_sdf,
        bool return_prj_pts,
        int sign_mode)
    {
        int threads = NTHREADS;
        int blocks = (num_queries + threads - 1) / threads;

        query_point_mesh_bvh_kernel<<<blocks, threads>>>(
            num_queries,
            num_objects,
            query_points,
            vertices,
            triangles,
            bvh_aabb_mins,
            bvh_aabb_maxs,
            bvh_children,
            object_ids,
            winding_data,
            out_query_ids,
            out_object_ids,
            out_projected_pts,
            out_distances,
            return_sdf,
            return_prj_pts,
            sign_mode);
    }

    __global__ void bottom_up_winding_data_kernel(
        const int num_objects,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const int *__restrict__ bvh_parents,
        const int2 *__restrict__ bvh_children,
        WindingData *__restrict__ winding_data,
        int *__restrict__ atomic_flags)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_objects)
            return;

        int leaf_idx = num_objects - 1 + idx;
        int original_obj_id = object_ids[idx];

        int3 tri = triangles[original_obj_id];
        float3 a = vertices[tri.x];
        float3 b = vertices[tri.y];
        float3 c = vertices[tri.z];

        float3 ab = b - a;
        float3 ac = c - a;

        float3 N = 0.5f * maths::cross(ab, ac);
        float area = maths::norm(N);
        float3 P = (a + b + c) / 3.0f;

        winding_data[leaf_idx].n = N;
        winding_data[leaf_idx].area = area;
        winding_data[leaf_idx].area_p = P * area;
        winding_data[leaf_idx].average_p = P;

        float d1 = maths::dot2(a - P);
        float d2 = maths::dot2(b - P);
        float d3 = maths::dot2(c - P);
        winding_data[leaf_idx].max_p_dist2 = fmaxf(d1, fmaxf(d2, d3));

        int current_node = bvh_parents[leaf_idx];

        while (current_node != -1)
        {
            __threadfence();

            int old = atomicAdd(&atomic_flags[current_node], 1);

            if (old == 0)
            {
                return;
            }

            int left_child = bvh_children[current_node].x;
            int right_child = bvh_children[current_node].y;

            WindingData left = winding_data[left_child];
            WindingData right = winding_data[right_child];

            float3 parent_N = left.n + right.n;
            float3 parent_area_p = left.area_p + right.area_p;
            float parent_area = left.area + right.area;

            float3 parent_average_p = make_float3(0.0f, 0.0f, 0.0f);
            if (parent_area > 0.0f)
            {
                parent_average_p = parent_area_p / parent_area;
            }

            winding_data[current_node].n = parent_N;
            winding_data[current_node].area_p = parent_area_p;
            winding_data[current_node].area = parent_area;
            winding_data[current_node].average_p = parent_average_p;

            float dist_to_left = maths::norm(parent_average_p - left.average_p);
            float dist_to_right = maths::norm(parent_average_p - right.average_p);

            float max_dist_left = dist_to_left + sqrtf(left.max_p_dist2);
            float max_dist_right = dist_to_right + sqrtf(right.max_p_dist2);

            float max_dist = fmaxf(max_dist_left, max_dist_right);
            winding_data[current_node].max_p_dist2 = max_dist * max_dist;

            current_node = bvh_parents[current_node];
        }
    }

    void bottom_up_winding_data(
        const int num_objects,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        const int *__restrict__ bvh_parents,
        const int2 *__restrict__ bvh_children,
        WindingData *__restrict__ winding_data)
    {
        int threads = NTHREADS;
        int blocks = (num_objects + threads - 1) / threads;

        thrust::device_vector<int> atomic_flags(num_objects - 1, 0);

        bottom_up_winding_data_kernel<<<blocks, threads>>>(
            num_objects,
            object_ids,
            vertices,
            triangles,
            bvh_parents,
            bvh_children,
            winding_data,
            thrust::raw_pointer_cast(atomic_flags.data()));
    }
}
