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
    torch::Tensor get_vertex_normals() const { return vertex_normals; }
    torch::Tensor get_vertex_colors() const { return vertex_colors; }
    torch::Tensor get_triangles() const { return triangles; }

    void compute_triangle_normals();
    void compute_triangle_areas();

    torch::Tensor get_triangle_areas();
    torch::Tensor get_triangle_normals();
    torch::Tensor get_surface_area();
    
    MeshBVH build_bvh();
    torch::Tensor get_self_intersection();
    bool is_self_intersection();

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
}

#endif // TRIANGLE_MESH_H