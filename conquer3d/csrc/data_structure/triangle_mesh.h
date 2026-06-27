#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "../maths/maths.h"
#include "../constants.h"
#include "../primitive/aabb.h"
#include "mesh_bvh.h"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <optional>
#include <tuple>

class TriangleMesh
{
protected:
    uint32_t num_triangles;

    torch::Tensor vertices;       // Size: [N, 3]
    torch::Tensor vertex_normals; // Size: [N, 3]
    torch::Tensor vertex_colors;  // Size: [N, 3]

    torch::Tensor triangles;        // Size: [M, 3] -> Each row contains indices into the vertices tensor
    torch::Tensor triangle_normals; // Size: [M, 3] -> Each row contains the normal of the corresponding triangle
    torch::Tensor triangle_areas;   // Size: [M] -> Each row contains the area of the corresponding triangle
    torch::Tensor surface_area;     // Size: [] -> Total surface area

    std::optional<MeshBVH> bvh;

    torch::Tensor edges;
    torch::Tensor edge_to_triangle_offsets;
    torch::Tensor edge_to_triangle_counts;
    torch::Tensor edge_to_triangle_indices;

    torch::Tensor vertex_to_triangle_offsets;
    torch::Tensor vertex_to_triangle_counts;
    torch::Tensor vertex_to_triangle_indices;

public:
    TriangleMesh(
        const torch::Tensor &in_vertices,
        const torch::Tensor &in_triangles,
        std::optional<torch::Tensor> in_vertex_normals = std::nullopt,
        std::optional<torch::Tensor> in_vertex_colors = std::nullopt);

    uint32_t get_num_triangles() const { return num_triangles; }
    torch::Tensor get_vertices() const { return vertices; }
    torch::Tensor get_vertex_normals();
    torch::Tensor get_vertex_colors() const { return vertex_colors; }
    torch::Tensor get_triangles() const { return triangles; }

    void compute_triangle_normals();
    void compute_vertex_normals();
    void compute_triangle_areas();

    torch::Tensor get_triangle_areas();
    torch::Tensor get_triangle_normals();
    torch::Tensor get_surface_area();
    
    MeshBVH build_bvh();
    torch::Tensor get_self_intersection();
    bool is_self_intersection();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> query_points(
        const torch::Tensor &query_pts,
        bool return_sdf = false,
        bool return_prj_pts = true,
        int sign_mode = 0,
        int distance_mode = 0);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_ray_intersection(
        const torch::Tensor &ray_origins,
        const torch::Tensor &ray_dirs,
        bool return_distance = false);

    std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> sample_points(
        int num_points,
        bool uniform = false,
        bool return_normals = false,
        bool return_colors = false,
        bool use_triangle_normal = true);

    void compute_edges_to_triangle_map();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_edges_to_triangle_map();
    
    torch::Tensor get_edges();
    torch::Tensor get_edge_to_triangle_offsets();
    torch::Tensor get_edge_to_triangle_counts();
    torch::Tensor get_edge_to_triangle_indices();

    void compute_vertices_to_triangle_map();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_vertices_to_triangle_map();
    torch::Tensor get_vertex_to_triangle_offsets();
    torch::Tensor get_vertex_to_triangle_counts();
    torch::Tensor get_vertex_to_triangle_indices();

    bool is_edge_manifold(bool allow_boundary_edge = true);
    bool is_vertex_manifold();
    bool is_manifold(bool allow_boundary_edge = true);
    torch::Tensor get_non_manifold_vertices();

    void remove_triangles_by_mask(const torch::Tensor &keep_mask);

    int32_t get_euler_characteristic();
    int32_t get_genus();
};

namespace triangle_mesh
{
    __host__ void compute_triangle_normals(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ triangle_normals);

    __host__ void compute_vertex_normals(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        const float3 *__restrict__ triangle_normals,
        float3 *__restrict__ vertex_normals);

    __host__ void compute_triangle_areas(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float *__restrict__ triangle_areas);

    __host__ void compute_triangle_aabbs(
        const uint32_t num_triangles,
        const float3 *__restrict__ vertices,
        const int3 *__restrict__ triangles,
        float3 *__restrict__ aabb_mins,
        float3 *__restrict__ aabb_maxs);

    __host__ void compute_edges_to_triangle_map(
        const uint32_t num_triangles,
        const int3 *__restrict__ triangles,
        torch::Tensor &out_unique_edges,
        torch::Tensor &out_offsets,
        torch::Tensor &out_counts,
        torch::Tensor &out_sorted_triangle_indices);

    __host__ void build_vertices_to_triangle_map(
        const uint32_t num_vertices,
        const uint32_t num_triangles,
        const torch::Tensor& triangles,
        torch::Tensor& out_counts,
        torch::Tensor& out_offsets,
        torch::Tensor& out_indices);

    torch::Tensor get_non_manifold_vertices(
        const uint32_t num_vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& v2t_offsets,
        const torch::Tensor& v2t_counts,
        const torch::Tensor& v2t_indices);

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
        float3 *__restrict__ out_colors);
}

#endif // TRIANGLE_MESH_H