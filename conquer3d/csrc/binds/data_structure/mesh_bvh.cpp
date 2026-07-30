#include <torch/extension.h>
#include "../../data_structure/mesh_bvh.h"
#include "../../data_structure/triangle_mesh.h"
#include <pybind11/pybind11.h>
#include "../../check.h"

namespace py = pybind11;

torch::Tensor MeshBVH::get_self_intersection(
    const torch::Tensor &vertices,
    const torch::Tensor &triangles)
{
    // 1. Broad-phase AABB overlap query
    auto [query_ids, object_ids] = this->query_self();

    int num_pairs = query_ids.size(0);
    if (num_pairs == 0)
    {
        return torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kInt64).device(query_ids.device()));
    }

    // 2. Narrow-phase intersection
    auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(query_ids.device());
    torch::Tensor out_query_ids = torch::empty({num_pairs}, options_i64);
    torch::Tensor out_object_ids = torch::empty({num_pairs}, options_i64);
    torch::Tensor valid_counter = torch::zeros({1}, options_i64);

    mesh_bvh::filter_self_intersections(
        num_pairs,
        query_ids.data_ptr<int64_t>(),
        object_ids.data_ptr<int64_t>(),
        (const float3 *)vertices.data_ptr<float>(),
        (const int3 *)triangles.data_ptr<int>(),
        out_query_ids.data_ptr<int64_t>(),
        out_object_ids.data_ptr<int64_t>(),
        valid_counter.data_ptr<int64_t>());

    int64_t h_valid_counter = valid_counter.item<int64_t>();

    if (h_valid_counter == 0)
    {
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

    if (num_pairs == 0)
    {
        return std::make_tuple(
            torch::empty({0}, options_i64),
            torch::empty({0}, options_i64),
            torch::empty({0, 3}, options_f32),
            torch::empty({0}, options_f32));
    }

    // 2. Narrow-phase intersection
    torch::Tensor out_query_ids = torch::empty({num_pairs}, options_i64);
    torch::Tensor out_object_ids = torch::empty({num_pairs}, options_i64);
    torch::Tensor out_intersect_pts = torch::empty({num_pairs, 3}, options_f32);

    torch::Tensor out_distances;
    if (return_distance)
    {
        out_distances = torch::empty({num_pairs}, options_f32);
    }
    else
    {
        out_distances = torch::empty({0}, options_f32);
    }

    torch::Tensor valid_counter = torch::zeros({1}, options_i64);

    mesh_bvh::filter_ray_triangle_intersections(
        num_pairs,
        query_ids.data_ptr<int64_t>(),
        object_ids.data_ptr<int64_t>(),
        (const float3 *)ray_origins.data_ptr<float>(),
        (const float3 *)ray_dirs.data_ptr<float>(),
        (const float3 *)vertices.data_ptr<float>(),
        (const int3 *)triangles.data_ptr<int>(),
        out_query_ids.data_ptr<int64_t>(),
        out_object_ids.data_ptr<int64_t>(),
        (float3 *)out_intersect_pts.data_ptr<float>(),
        return_distance ? out_distances.data_ptr<float>() : nullptr,
        return_distance,
        valid_counter.data_ptr<int64_t>());

    int64_t h_valid_counter = valid_counter.item<int64_t>();

    if (h_valid_counter == 0)
    {
        return std::make_tuple(
            torch::empty({0}, options_i64),
            torch::empty({0}, options_i64),
            torch::empty({0, 3}, options_f32),
            torch::empty({0}, options_f32));
    }

    return std::make_tuple(
        out_query_ids.slice(0, 0, h_valid_counter),
        out_object_ids.slice(0, 0, h_valid_counter),
        out_intersect_pts.slice(0, 0, h_valid_counter),
        return_distance ? out_distances.slice(0, 0, h_valid_counter) : out_distances);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MeshBVH::query_point(
    const torch::Tensor &query_points,
    const torch::Tensor &vertices,
    const torch::Tensor &triangles,
    bool return_sdf,
    bool return_prj_pts,
    int sign_mode,
    std::optional<torch::Tensor> triangle_normals,
    std::optional<torch::Tensor> vertex_normals,
    std::optional<torch::Tensor> edge_normals,
    std::optional<torch::Tensor> flood_fill_mask,
    std::optional<std::vector<float>> flood_grid_min,
    std::optional<std::vector<float>> flood_grid_max,
    std::optional<std::vector<int64_t>> flood_grid_res)
{
    if (sign_mode != 0 && sign_mode != 1 && sign_mode != 2 && sign_mode != 3 && sign_mode != 4)
    {
        throw std::runtime_error("sign_mode must be 0 (ray casting), 1 (fast winding number), 2 (pseudonormals), 3 (flood fill), or 4 (hybrid WN + pseudonormal)");
    }

    if ((sign_mode == 1 || sign_mode == 4) && !this->has_winding_data)
    {
        this->build_winding_data(vertices, triangles);
    }

    torch::Tensor use_tri_normals, use_vert_normals, use_edge_normals;
    if (sign_mode == 2 || sign_mode == 4)
    {
        if (triangle_normals.has_value() && triangle_normals->defined()) {
            use_tri_normals = *triangle_normals;
        } else {
            int num_tri = triangles.size(0);
            auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device());
            use_tri_normals = torch::empty({num_tri, 3}, options_f32);
            triangle_mesh::compute_triangle_normals(
                num_tri,
                (const float3 *)vertices.data_ptr<float>(),
                (const int3 *)triangles.data_ptr<int>(),
                (float3 *)use_tri_normals.data_ptr<float>());
        }

        if (vertex_normals.has_value() && vertex_normals->defined()) {
            use_vert_normals = *vertex_normals;
        } else {
            int num_vert = vertices.size(0);
            int num_tri = triangles.size(0);
            auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device());
            use_vert_normals = torch::zeros({num_vert, 3}, options_f32);
            triangle_mesh::compute_vertex_normals(
                num_vert,
                num_tri,
                (const float3 *)vertices.data_ptr<float>(),
                (const int3 *)triangles.data_ptr<int>(),
                (const float3 *)use_tri_normals.data_ptr<float>(),
                (float3 *)use_vert_normals.data_ptr<float>(),
                1);
        }

        if (edge_normals.has_value() && edge_normals->defined()) {
            use_edge_normals = *edge_normals;
        } else {
            int num_tri = triangles.size(0);
            auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device());
            use_edge_normals = torch::empty({num_tri * 3, 3}, options_f32);
            triangle_mesh::compute_edge_normals(
                num_tri,
                (const int3 *)triangles.data_ptr<int>(),
                (const float3 *)use_tri_normals.data_ptr<float>(),
                (float3 *)use_edge_normals.data_ptr<float>());
        }
    }

    int num_queries = query_points.size(0);
    int num_objects = this->object_ids.size(0);

    auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(query_points.device());
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(query_points.device());

    torch::Tensor out_query_ids = torch::empty({num_queries}, options_i64);
    torch::Tensor out_object_ids = torch::empty({num_queries}, options_i64);
    torch::Tensor out_projected_pts;
    if (return_prj_pts)
    {
        out_projected_pts = torch::empty({num_queries, 3}, options_f32);
    }
    else
    {
        out_projected_pts = torch::empty({0, 3}, options_f32);
    }
    torch::Tensor out_distances = torch::empty({num_queries}, options_f32);

    const int8_t *p_flood_mask = nullptr;
    float3 f_min = make_float3(0.0f, 0.0f, 0.0f);
    float3 f_spacing = make_float3(1.0f, 1.0f, 1.0f);
    int3 f_dims = make_int3(0, 0, 0);

    if (sign_mode == 3)
    {
        if (!flood_fill_mask.has_value() || !flood_fill_mask->defined() || !flood_grid_min.has_value() || !flood_grid_max.has_value() || !flood_grid_res.has_value())
        {
            throw std::runtime_error("For sign_mode == 3 (flood fill), flood fill data must be provided or built beforehand.");
        }
        p_flood_mask = flood_fill_mask->data_ptr<int8_t>();
        f_min = make_float3(flood_grid_min->at(0), flood_grid_min->at(1), flood_grid_min->at(2));
        int64_t rx = flood_grid_res->at(0);
        int64_t ry = flood_grid_res->at(1);
        int64_t rz = flood_grid_res->at(2);
        f_dims = make_int3(static_cast<int>(rx), static_cast<int>(ry), static_cast<int>(rz));
        f_spacing = make_float3(
            (flood_grid_max->at(0) - flood_grid_min->at(0)) / (rx - 1),
            (flood_grid_max->at(1) - flood_grid_min->at(1)) / (ry - 1),
            (flood_grid_max->at(2) - flood_grid_min->at(2)) / (rz - 1)
        );
    }

    mesh_bvh::query_point_mesh_bvh(
        num_queries,
        num_objects,
        (const float3 *)query_points.data_ptr<float>(),
        (const float3 *)vertices.data_ptr<float>(),
        (const int3 *)triangles.data_ptr<int>(),
        (const float3 *)this->aabb_mins.data_ptr<float>(),
        (const float3 *)this->aabb_maxs.data_ptr<float>(),
        (const int2 *)this->bvh_children.data_ptr<int>(),
        this->object_ids.data_ptr<int>(),
        this->has_winding_data ? (const WindingData *)this->winding_data.data_ptr() : nullptr,
        (sign_mode == 2 || sign_mode == 4) ? (const float3 *)use_vert_normals.data_ptr<float>() : nullptr,
        (sign_mode == 2 || sign_mode == 4) ? (const float3 *)use_edge_normals.data_ptr<float>() : nullptr,
        (sign_mode == 2 || sign_mode == 4) ? (const float3 *)use_tri_normals.data_ptr<float>() : nullptr,
        out_query_ids.data_ptr<int64_t>(),
        out_object_ids.data_ptr<int64_t>(),
        return_prj_pts ? (float3 *)out_projected_pts.data_ptr<float>() : nullptr,
        out_distances.data_ptr<float>(),
        return_sdf,
        return_prj_pts,
        sign_mode,
        p_flood_mask,
        f_min,
        f_spacing,
        f_dims);

    return std::make_tuple(out_query_ids, out_object_ids, out_projected_pts, out_distances);
}

torch::Tensor MeshBVH::query_voxel(
    const torch::Tensor &query_mins,
    const torch::Tensor &query_maxs,
    const torch::Tensor &vertices,
    const torch::Tensor &triangles)
{
    int num_queries = query_mins.size(0);
    int num_objects = this->object_ids.size(0);

    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(query_mins.device());
    torch::Tensor out_intersect = torch::empty({num_queries}, options_bool);

    if (num_queries == 0 || num_objects == 0)
    {
        out_intersect.fill_(false);
        return out_intersect;
    }

    mesh_bvh::query_voxel_mesh_bvh(
        num_queries,
        num_objects,
        (const float3 *)query_mins.data_ptr<float>(),
        (const float3 *)query_maxs.data_ptr<float>(),
        (const float3 *)vertices.data_ptr<float>(),
        (const int3 *)triangles.data_ptr<int>(),
        (const float3 *)this->aabb_mins.data_ptr<float>(),
        (const float3 *)this->aabb_maxs.data_ptr<float>(),
        (const int2 *)this->bvh_children.data_ptr<int>(),
        this->object_ids.data_ptr<int>(),
        out_intersect.data_ptr<bool>());

    return out_intersect;
}

torch::Tensor MeshBVH::get_active_voxel_ids_from_grid(
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> res,
    const torch::Tensor &vertices,
    const torch::Tensor &triangles)
{
    int64_t rx = res[0] - 1;
    int64_t ry = res[1] - 1;
    int64_t rz = res[2] - 1;
    int64_t num_queries = rx * ry * rz;
    int num_objects = this->object_ids.size(0);

    auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(vertices.device());

    if (num_queries <= 0 || num_objects == 0)
    {
        return torch::empty({0}, options_i64);
    }

    float spacing_x = (grid_max[0] - grid_min[0]) / (res[0] - 1);
    float spacing_y = (grid_max[1] - grid_min[1]) / (res[1] - 1);
    float spacing_z = (grid_max[2] - grid_min[2]) / (res[2] - 1);

    int3 res_int = make_int3(rx, ry, rz);
    float3 g_min = make_float3(grid_min[0], grid_min[1], grid_min[2]);
    float3 v_size = make_float3(spacing_x, spacing_y, spacing_z);

    // Pass 1: Count active voxels
    torch::Tensor active_counter = torch::zeros({1}, options_i64);

    mesh_bvh::count_active_voxels_mesh_bvh(
        res_int,
        g_min,
        v_size,
        num_objects,
        (const float3 *)vertices.data_ptr<float>(),
        (const int3 *)triangles.data_ptr<int>(),
        (const float3 *)this->aabb_mins.data_ptr<float>(),
        (const float3 *)this->aabb_maxs.data_ptr<float>(),
        (const int2 *)this->bvh_children.data_ptr<int>(),
        this->object_ids.data_ptr<int>(),
        active_counter.data_ptr<int64_t>());

    int64_t num_active = active_counter.item<int64_t>();

    if (num_active == 0)
    {
        return torch::empty({0}, options_i64);
    }

    // Pass 2: Collect active voxels
    torch::Tensor out_active_ids = torch::empty({num_active}, options_i64);
    active_counter.zero_();

    mesh_bvh::collect_active_voxels_mesh_bvh(
        res_int,
        g_min,
        v_size,
        num_objects,
        (const float3 *)vertices.data_ptr<float>(),
        (const int3 *)triangles.data_ptr<int>(),
        (const float3 *)this->aabb_mins.data_ptr<float>(),
        (const float3 *)this->aabb_maxs.data_ptr<float>(),
        (const int2 *)this->bvh_children.data_ptr<int>(),
        this->object_ids.data_ptr<int>(),
        active_counter.data_ptr<int64_t>(),
        out_active_ids.data_ptr<int64_t>());

    return out_active_ids;
}

void MeshBVH::build_winding_data(
    const torch::Tensor &vertices,
    const torch::Tensor &triangles)
{
    int num_objects = this->object_ids.size(0);
    int num_nodes = num_objects * 2 - 1;

    auto options = torch::TensorOptions().dtype(torch::kByte).device(vertices.device());
    this->winding_data = torch::empty({num_nodes * (int64_t)sizeof(WindingData)}, options);

    mesh_bvh::bottom_up_winding_data(
        num_objects,
        this->object_ids.data_ptr<int>(),
        (const float3 *)vertices.data_ptr<float>(),
        (const int3 *)triangles.data_ptr<int>(),
        this->bvh_parents.data_ptr<int>(),
        (const int2 *)this->bvh_children.data_ptr<int>(),
        (WindingData *)this->winding_data.data_ptr());

    cudaDeviceSynchronize();
    this->has_winding_data = true;
}

void bind_ds_mesh_bvh(py::module_ &m)
{
    py::class_<MeshBVH, BVH>(m, "MeshBVH", R"doc(
    A specialized Bounding Volume Hierarchy for Triangle Meshes.
    Inherits from BVH.
    )doc")
        .def(py::init<const torch::Tensor &, const torch::Tensor &>(),
             py::arg("in_aabb_mins"),
             py::arg("in_aabb_maxs"),
             R"doc(
        Construct MeshBVH from Triangle AABBs.

        Args:
            in_aabb_mins (torch.Tensor): Shape (M, 3) float32 tensor of triangle AABB minimums.
            in_aabb_maxs (torch.Tensor): Shape (M, 3) float32 tensor of triangle AABB maximums.
        )doc")
        .def("get_self_intersection", &MeshBVH::get_self_intersection,
             py::arg("vertices"),
             py::arg("triangles"),
             R"doc(
        Find all self-intersecting triangle pairs.

        Args:
            vertices (torch.Tensor): Shape (N, 3) float32 tensor of vertices.
            triangles (torch.Tensor): Shape (M, 3) int32 tensor of triangles.

        Returns:
            torch.Tensor: Shape (K, 2) int64 tensor of intersecting triangle index pairs.
        )doc")
        .def("is_self_intersection", &MeshBVH::is_self_intersection,
             py::arg("vertices"),
             py::arg("triangles"),
             R"doc(
        Check if there are any self-intersecting triangle pairs.

        Args:
            vertices (torch.Tensor): Shape (N, 3) float32 tensor of vertices.
            triangles (torch.Tensor): Shape (M, 3) int32 tensor of triangles.

        Returns:
            bool: True if there is at least one self-intersection.
        )doc")
        .def("get_ray_intersection", [](MeshBVH &self, const torch::Tensor &ray_origins, const torch::Tensor &ray_dirs, const torch::Tensor &vertices, const torch::Tensor &triangles, bool return_distance) -> py::object
             {
                 auto result = self.get_ray_intersection(ray_origins, ray_dirs, vertices, triangles, return_distance);
                 if (return_distance) {
                     return py::cast(result);
                 } else {
                     return py::cast(std::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result)));
                 } }, py::arg("ray_origins"), py::arg("ray_dirs"), py::arg("vertices"), py::arg("triangles"), py::arg("return_distance") = false,
             R"doc(
        Find all ray-triangle intersections.

        Args:
            ray_origins (torch.Tensor): Shape (R, 3) float32 tensor of ray origins.
            ray_dirs (torch.Tensor): Shape (R, 3) float32 tensor of ray directions.
            vertices (torch.Tensor): Shape (N, 3) float32 tensor of vertices.
            triangles (torch.Tensor): Shape (M, 3) int32 tensor of triangles.
            return_distance (bool): Whether to compute and return ray hit distances. Defaults to false.

        Returns:
            tuple: (ray_ids, triangle_ids, intersect_points, [distances])
        )doc")
        .def("query_point", &MeshBVH::query_point, py::arg("query_points"), py::arg("vertices"), py::arg("triangles"), py::arg("return_sdf") = false, py::arg("return_prj_pts") = true, py::arg("sign_mode") = 0, py::arg("triangle_normals") = py::none(), py::arg("vertex_normals") = py::none(), py::arg("edge_normals") = py::none(), py::arg("flood_fill_mask") = py::none(), py::arg("flood_grid_min") = py::none(), py::arg("flood_grid_max") = py::none(), py::arg("flood_grid_res") = py::none(),
             R"doc(
        Find the closest triangle to each query point.

        Args:
            query_points (torch.Tensor): Shape (Q, 3) float32 tensor of query points.
            vertices (torch.Tensor): Shape (N, 3) float32 tensor of vertices.
            triangles (torch.Tensor): Shape (M, 3) int32 tensor of triangles.
            return_sdf (bool): Whether to compute signed distances instead of unsigned. Defaults to false.
            return_prj_pts (bool): Whether to compute and return projected points. Defaults to true.
            sign_mode (int): Winding number method (0 for ray casting, 1 for fast winding number, 2 for pseudonormals, 3 for flood fill, 4 for hybrid WN + pseudonormal). Defaults to 0.
            triangle_normals (torch.Tensor, optional): Pre-computed face normals from TriangleMesh.
            vertex_normals (torch.Tensor, optional): Pre-computed vertex normals from TriangleMesh.
            edge_normals (torch.Tensor, optional): Pre-computed edge normals from TriangleMesh.
            flood_fill_mask (torch.Tensor, optional): Pre-computed flood fill volume mask.
            flood_grid_min (list, optional): Flood grid bounding box min.
            flood_grid_max (list, optional): Flood grid bounding box max.
            flood_grid_res (list, optional): Flood grid vertex resolution.

        Returns:
            tuple: (query_ids, triangle_ids, projected_points, distances)
        )doc")
        .def("build_winding_data", &MeshBVH::build_winding_data, py::arg("vertices"), py::arg("triangles"),
             R"doc(
        Computes the hierarchical area-weighted constants for Fast Winding Number queries.

        Args:
            vertices (torch.Tensor): Shape (N, 3) float32 tensor of vertices.
            triangles (torch.Tensor): Shape (M, 3) int32 tensor of triangles.
        )doc")
        .def("query_voxel", &MeshBVH::query_voxel, py::arg("query_mins"), py::arg("query_maxs"), py::arg("vertices"), py::arg("triangles"),
             R"doc(
        Queries whether each voxel intersects the mesh.
        
        Args:
            query_mins (torch.Tensor): Shape (Q, 3) float32 tensor of voxel minimums.
            query_maxs (torch.Tensor): Shape (Q, 3) float32 tensor of voxel maximums.
            vertices (torch.Tensor): Shape (N, 3) float32 tensor of vertices.
            triangles (torch.Tensor): Shape (M, 3) int32 tensor of triangles.
            
        Returns:
            torch.Tensor: Boolean tensor of shape (Q,) indicating intersection.
        )doc");
}
