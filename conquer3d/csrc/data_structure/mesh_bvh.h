#ifndef MESH_BVH_H
#define MESH_BVH_H

#include "bvh.h"
#include <torch/extension.h>

struct WindingData {
    float3 n;          // Area-weighted normal (unnormalized)
    float3 area_p;     // Area-weighted centroid
    float area;        // Total area
    float3 average_p;  // Final centroid (area_p / area)
    float max_p_dist2; // Reserved for query thresholding
};

class MeshBVH : public BVH
{
public:
    torch::Tensor winding_data; // Size: [2N - 1] * sizeof(WindingData)
    bool has_winding_data = false;

    using BVH::BVH;
    using BVH::query;
    using BVH::query_self;

    using BVH::query_ray;

    void build_winding_data(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    // Returns a [N, 2] tensor of intersecting triangle index pairs
    torch::Tensor get_self_intersection(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    bool is_self_intersection(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_ray_intersection(
        const torch::Tensor &ray_origins,
        const torch::Tensor &ray_dirs,
        const torch::Tensor &vertices,
        const torch::Tensor &triangles,
        bool return_distance = false);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> query_point(
        const torch::Tensor &query_points,
        const torch::Tensor &vertices,
        const torch::Tensor &triangles,
        bool return_sdf = false,
        bool return_prj_pts = true,
        int sign_mode = 0,
        std::optional<torch::Tensor> triangle_normals = std::nullopt,
        std::optional<torch::Tensor> vertex_normals = std::nullopt,
        std::optional<torch::Tensor> edge_normals = std::nullopt,
        std::optional<torch::Tensor> flood_fill_mask = std::nullopt,
        std::optional<std::vector<float>> flood_grid_min = std::nullopt,
        std::optional<std::vector<float>> flood_grid_max = std::nullopt,
        std::optional<std::vector<int64_t>> flood_grid_res = std::nullopt);

    torch::Tensor query_voxel(
        const torch::Tensor &query_mins,
        const torch::Tensor &query_maxs,
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    torch::Tensor get_active_voxel_ids_from_grid(
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> res,
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);
};

namespace mesh_bvh
{
    __host__ void filter_self_intersections(
        const int num_pairs,
        const int64_t *query_ids,
        const int64_t *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        int64_t *valid_counter);

    __host__ void filter_ray_triangle_intersections(
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
        int64_t *valid_counter);
    
    __host__ void query_point_mesh_bvh(
        const int num_queries,
        const int num_objects,
        const float3 *query_points,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        const WindingData *winding_data,
        const float3 *pseudonormal_vertices,
        const float3 *pseudonormal_edges,
        const float3 *pseudonormal_faces,
        int64_t *out_query_ids,
        int64_t *out_object_ids,
        float3 *out_projected_pts,
        float *out_distances,
        bool return_sdf,
        bool return_prj_pts,
        int sign_mode,
        const int8_t *flood_mask = nullptr,
        float3 flood_min = make_float3(0.0f, 0.0f, 0.0f),
        float3 flood_spacing = make_float3(1.0f, 1.0f, 1.0f),
        int3 flood_dims = make_int3(0, 0, 0));
    
    __host__ void bottom_up_winding_data(
        const int num_objects,
        const int *object_ids,
        const float3 *vertices,
        const int3 *triangles,
        const int *bvh_parents,
        const int2 *bvh_children,
        WindingData *winding_data);
    __host__ void query_voxel_mesh_bvh(
        const int num_queries,
        const int num_objects,
        const float3 *query_mins,
        const float3 *query_maxs,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        bool *out_intersect);

    __host__ void count_active_voxels_mesh_bvh(
        const int3 res,
        const float3 grid_min,
        const float3 voxel_size,
        const int num_objects,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        int64_t *active_counter);

    __host__ void collect_active_voxels_mesh_bvh(
        const int3 res,
        const float3 grid_min,
        const float3 voxel_size,
        const int num_objects,
        const float3 *vertices,
        const int3 *triangles,
        const float3 *bvh_aabb_mins,
        const float3 *bvh_aabb_maxs,
        const int2 *bvh_children,
        const int *object_ids,
        int64_t *active_counter,
        int64_t *out_active_ids);
}

#endif // MESH_BVH_H
