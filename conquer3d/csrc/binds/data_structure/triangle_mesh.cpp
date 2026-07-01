#include <torch/extension.h>
#include "../../data_structure/triangle_mesh.h"
#include "../../check.h"

#include <pybind11/pybind11.h>
#include <optional>
#include <vector>

namespace py = pybind11;

#include <optional>

TriangleMesh::TriangleMesh(
    const torch::Tensor &in_vertices,
    const torch::Tensor &in_triangles,
    std::optional<torch::Tensor> in_vertex_normals,
    std::optional<torch::Tensor> in_vertex_colors)
{
    CHECK_INPUT(in_vertices);
    CHECK_INPUT(in_triangles);
    TORCH_CHECK(in_vertices.scalar_type() == torch::kFloat32, "vertices must be float32");
    TORCH_CHECK(in_triangles.scalar_type() == torch::kInt32, "triangles must be int32");
    TORCH_CHECK(in_vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(in_triangles.size(1) == 3, "triangles must have shape (M, 3)");

    this->num_triangles = static_cast<uint32_t>(in_triangles.size(0));

    this->vertices = in_vertices;
    this->triangles = in_triangles;

    if (in_vertex_normals.has_value() && in_vertex_normals->defined())
    {
        CHECK_INPUT(*in_vertex_normals);
        TORCH_CHECK(in_vertex_normals->scalar_type() == torch::kFloat32, "vertex_normals must be float32");
        TORCH_CHECK(in_vertex_normals->size(1) == 3, "vertex_normals must have shape (N, 3)");
        this->vertex_normals = in_vertex_normals->clone();
    }

    if (in_vertex_colors.has_value() && in_vertex_colors->defined())
    {
        CHECK_INPUT(*in_vertex_colors);
        TORCH_CHECK(in_vertex_colors->scalar_type() == torch::kFloat32, "vertex_colors must be float32");
        TORCH_CHECK(in_vertex_colors->size(1) == 3, "vertex_colors must have shape (N, 3)");
        this->vertex_colors = in_vertex_colors->clone();
    }
}

void TriangleMesh::compute_triangle_normals()
{
    this->triangle_normals = torch::empty({static_cast<int64_t>(this->num_triangles), 3}, torch::dtype(torch::kFloat32).device(this->vertices.device()));
    triangle_mesh::compute_triangle_normals(
        this->num_triangles,
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float3 *>(this->triangle_normals.data_ptr<float>()));
}

void TriangleMesh::compute_vertex_normals()
{
    uint32_t num_vertices = this->vertices.size(0);
    this->vertex_normals = torch::zeros({static_cast<int64_t>(num_vertices), 3}, torch::dtype(torch::kFloat32).device(this->vertices.device()));

    triangle_mesh::compute_vertex_normals(
        num_vertices,
        this->num_triangles,
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<const float3 *>(this->get_triangle_normals().data_ptr<float>()),
        reinterpret_cast<float3 *>(this->vertex_normals.data_ptr<float>()));
}

torch::Tensor TriangleMesh::get_vertex_normals()
{
    if (!this->vertex_normals.defined())
    {
        this->compute_vertex_normals();
    }
    return this->vertex_normals;
}

void TriangleMesh::compute_triangle_areas()
{
    this->triangle_areas = torch::empty({static_cast<int64_t>(this->num_triangles)}, torch::dtype(torch::kFloat32).device(this->vertices.device()));
    triangle_mesh::compute_triangle_areas(
        this->num_triangles,
        reinterpret_cast<float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float *>(this->triangle_areas.data_ptr<float>()));
}

MeshBVH TriangleMesh::build_bvh()
{
    if (!this->bvh.has_value())
    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(this->vertices.device());
        torch::Tensor aabb_mins = torch::empty({static_cast<int64_t>(this->num_triangles), 3}, options);
        torch::Tensor aabb_maxs = torch::empty({static_cast<int64_t>(this->num_triangles), 3}, options);

        triangle_mesh::compute_triangle_aabbs(
            this->num_triangles,
            reinterpret_cast<float3 *>(this->vertices.data_ptr<float>()),
            reinterpret_cast<int3 *>(this->triangles.data_ptr<int>()),
            reinterpret_cast<float3 *>(aabb_mins.data_ptr<float>()),
            reinterpret_cast<float3 *>(aabb_maxs.data_ptr<float>()));

        this->bvh = MeshBVH(aabb_mins, aabb_maxs);
    }
    return this->bvh.value();
}

torch::Tensor TriangleMesh::get_self_intersection()
{
    this->build_bvh();
    return this->bvh.value().get_self_intersection(this->vertices, this->triangles);
}

bool TriangleMesh::is_self_intersection()
{
    this->build_bvh();
    return this->bvh.value().is_self_intersection(this->vertices, this->triangles);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TriangleMesh::query_points(
    const torch::Tensor &query_pts,
    bool return_sdf,
    bool return_prj_pts,
    int sign_mode,
    int distance_mode)
{
    if (distance_mode == 0)
    {
        this->build_bvh();
        return this->bvh.value().query_point(query_pts, this->vertices, this->triangles, return_sdf, return_prj_pts, sign_mode);
    }
    else
    {
        throw std::runtime_error("distance_mode != 0 not implemented yet");
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TriangleMesh::get_ray_intersection(
    const torch::Tensor &ray_origins,
    const torch::Tensor &ray_dirs,
    bool return_distance)
{
    this->build_bvh();
    return this->bvh.value().get_ray_intersection(ray_origins, ray_dirs, this->vertices, this->triangles, return_distance);
}

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> TriangleMesh::sample_points(
    int num_points, bool uniform, bool return_normals, bool return_colors, bool use_triangle_normal)
{
    if (this->num_triangles == 0)
    {
        throw std::runtime_error("Cannot sample points from an empty mesh.");
    }

    torch::Tensor tri_indices;
    if (!uniform)
    {
        tri_indices = torch::randint(0, this->num_triangles, {num_points}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kInt64));
    }
    else
    {
        torch::Tensor areas = this->get_triangle_areas();
        tri_indices = torch::multinomial(areas, num_points, true);
    }

    torch::Tensor r1_r2 = torch::rand({num_points, 2}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));
    torch::Tensor out_points = torch::empty({num_points, 3}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));

    torch::Tensor out_normals, out_colors;
    const float3 *d_vertex_normals = nullptr;
    const float3 *d_triangle_normals = nullptr;
    const float3 *d_vertex_colors = nullptr;
    float3 *d_out_normals = nullptr;
    float3 *d_out_colors = nullptr;

    if (return_normals)
    {
        out_normals = torch::empty({num_points, 3}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));
        d_out_normals = reinterpret_cast<float3 *>(out_normals.data_ptr<float>());
        if (use_triangle_normal)
        {
            d_triangle_normals = reinterpret_cast<const float3 *>(this->get_triangle_normals().data_ptr<float>());
        }
        else
        {
            d_vertex_normals = reinterpret_cast<const float3 *>(this->get_vertex_normals().data_ptr<float>());
        }
    }

    if (return_colors)
    {
        if (!this->vertex_colors.defined())
        {
            throw std::runtime_error("Cannot sample colors because vertex_colors is not defined.");
        }
        out_colors = torch::empty({num_points, 3}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));
        d_out_colors = reinterpret_cast<float3 *>(out_colors.data_ptr<float>());
        d_vertex_colors = reinterpret_cast<const float3 *>(this->vertex_colors.data_ptr<float>());
    }

    triangle_mesh::sample_points_triangle_mesh(
        num_points,
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<const int64_t *>(tri_indices.data_ptr<int64_t>()),
        reinterpret_cast<const float2 *>(r1_r2.data_ptr<float>()),
        d_vertex_normals,
        d_triangle_normals,
        d_vertex_colors,
        reinterpret_cast<float3 *>(out_points.data_ptr<float>()),
        d_out_normals,
        d_out_colors);

    std::optional<torch::Tensor> opt_normals = return_normals ? std::make_optional(out_normals) : std::nullopt;
    std::optional<torch::Tensor> opt_colors = return_colors ? std::make_optional(out_colors) : std::nullopt;

    return std::make_tuple(out_points, tri_indices, opt_normals, opt_colors);
}

torch::Tensor TriangleMesh::get_triangle_areas()
{
    if (!this->triangle_areas.defined())
    {
        this->compute_triangle_areas();
    }
    return this->triangle_areas;
}

torch::Tensor TriangleMesh::get_triangle_normals()
{
    if (!this->triangle_normals.defined())
    {
        this->compute_triangle_normals();
    }
    return this->triangle_normals;
}

torch::Tensor TriangleMesh::get_surface_area()
{
    if (!this->surface_area.defined())
    {
        this->surface_area = this->get_triangle_areas().sum();
    }
    return this->surface_area;
}

void TriangleMesh::compute_edges_to_triangle_map()
{
    if (this->num_triangles == 0)
    {
        auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(this->triangles.device());
        this->edges = torch::empty({0, 2}, options_i32);
        this->edge_to_triangle_offsets = torch::empty({0}, options_i32);
        this->edge_to_triangle_counts = torch::empty({0}, options_i32);
        this->edge_to_triangle_indices = torch::empty({0}, options_i32);
        return;
    }

    triangle_mesh::compute_edges_to_triangle_map(
        this->num_triangles,
        reinterpret_cast<int3 *>(this->triangles.data_ptr<int>()),
        this->edges,
        this->edge_to_triangle_offsets,
        this->edge_to_triangle_counts,
        this->edge_to_triangle_indices);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TriangleMesh::get_edges_to_triangle_map()
{
    if (!this->edges.defined())
    {
        this->compute_edges_to_triangle_map();
    }
    return std::make_tuple(
        this->edges,
        this->edge_to_triangle_offsets,
        this->edge_to_triangle_counts,
        this->edge_to_triangle_indices);
}

torch::Tensor TriangleMesh::get_edges()
{
    if (!this->edges.defined())
        this->compute_edges_to_triangle_map();
    return this->edges;
}
torch::Tensor TriangleMesh::get_edge_to_triangle_offsets()
{
    if (!this->edge_to_triangle_offsets.defined())
        this->compute_edges_to_triangle_map();
    return this->edge_to_triangle_offsets;
}
torch::Tensor TriangleMesh::get_edge_to_triangle_counts()
{
    if (!this->edge_to_triangle_counts.defined())
        this->compute_edges_to_triangle_map();
    return this->edge_to_triangle_counts;
}
torch::Tensor TriangleMesh::get_edge_to_triangle_indices()
{
    if (!this->edge_to_triangle_indices.defined())
        this->compute_edges_to_triangle_map();
    return this->edge_to_triangle_indices;
}

bool TriangleMesh::is_edge_manifold(bool allow_boundary_edge)
{
    if (this->num_triangles == 0)
        return true;
    torch::Tensor counts = this->get_edge_to_triangle_counts();
    if (allow_boundary_edge)
    {
        return (counts <= 2).all().item<bool>();
    }
    else
    {
        return (counts == 2).all().item<bool>();
    }
}

void TriangleMesh::remove_triangles_by_mask(const torch::Tensor &keep_mask)
{
    CHECK_INPUT(keep_mask);
    TORCH_CHECK(keep_mask.scalar_type() == torch::kBool, "keep_mask must be a boolean tensor");
    TORCH_CHECK(keep_mask.dim() == 1 && keep_mask.size(0) == this->num_triangles, "keep_mask must have shape (num_triangles,)");

    this->triangles = this->triangles.index({keep_mask});
    this->num_triangles = this->triangles.size(0);

    // Invalidate all caches
    this->triangle_areas = torch::Tensor();
    this->triangle_normals = torch::Tensor();
    this->surface_area = torch::Tensor();
    this->bvh.reset();
    this->edges = torch::Tensor();
    this->edge_to_triangle_offsets = torch::Tensor();
    this->edge_to_triangle_counts = torch::Tensor();
    this->edge_to_triangle_indices = torch::Tensor();

    this->vertex_to_triangle_offsets = torch::Tensor();
    this->vertex_to_triangle_counts = torch::Tensor();
    this->vertex_to_triangle_indices = torch::Tensor();
}

int32_t TriangleMesh::get_euler_characteristic()
{
    int32_t V = this->vertices.size(0);
    int32_t E = this->get_edges().size(0);
    int32_t F = this->num_triangles;
    return V - E + F;
}

int32_t TriangleMesh::get_genus()
{
    // For a single closed connected component, chi = 2 - 2g => g = (2 - chi) / 2
    int32_t chi = this->get_euler_characteristic();
    return (2 - chi) / 2;
}

void TriangleMesh::compute_vertices_to_triangle_map()
{
    if (this->vertex_to_triangle_offsets.defined())
        return;

    uint32_t num_vertices = this->vertices.size(0);
    triangle_mesh::build_vertices_to_triangle_map(
        num_vertices,
        this->num_triangles,
        this->triangles,
        this->vertex_to_triangle_counts,
        this->vertex_to_triangle_offsets,
        this->vertex_to_triangle_indices);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> TriangleMesh::get_vertices_to_triangle_map()
{
    this->compute_vertices_to_triangle_map();
    return {this->vertex_to_triangle_offsets, this->vertex_to_triangle_counts, this->vertex_to_triangle_indices};
}

torch::Tensor TriangleMesh::get_vertex_to_triangle_offsets()
{
    this->compute_vertices_to_triangle_map();
    return this->vertex_to_triangle_offsets;
}

torch::Tensor TriangleMesh::get_vertex_to_triangle_counts()
{
    this->compute_vertices_to_triangle_map();
    return this->vertex_to_triangle_counts;
}

torch::Tensor TriangleMesh::get_vertex_to_triangle_indices()
{
    this->compute_vertices_to_triangle_map();
    return this->vertex_to_triangle_indices;
}

torch::Tensor TriangleMesh::get_non_manifold_vertices()
{
    this->compute_vertices_to_triangle_map();
    return triangle_mesh::get_non_manifold_vertices(
        this->vertices.size(0),
        this->triangles,
        this->vertex_to_triangle_offsets,
        this->vertex_to_triangle_counts,
        this->vertex_to_triangle_indices);
}

bool TriangleMesh::is_vertex_manifold()
{
    torch::Tensor nm_vertices = this->get_non_manifold_vertices();
    return nm_vertices.size(0) == 0;
}

bool TriangleMesh::is_manifold(bool allow_boundary_edge)
{
    return this->is_edge_manifold(allow_boundary_edge) && this->is_vertex_manifold() && !this->is_self_intersection();
}

void bind_ds_triangle_mesh(py::module_ &m)
{
    py::class_<TriangleMesh>(m, "TriangleMesh", R"doc(
        A highly efficient Triangle Mesh data structure natively backed by CUDA.

        Args:
            in_vertices (torch.Tensor): A tensor of shape (N, 3) containing float32 vertex coordinates.
            in_triangles (torch.Tensor): A tensor of shape (M, 3) containing int32 triangle indices.
            in_vertex_normals (torch.Tensor, optional): A tensor of shape (N, 3) containing float32 vertex normals. Defaults to None.
            in_vertex_colors (torch.Tensor, optional): A tensor of shape (N, 3) containing float32 vertex colors. Defaults to None.
        )doc")
        .def(py::init<const torch::Tensor &, const torch::Tensor &, std::optional<torch::Tensor>, std::optional<torch::Tensor>>(),
             py::arg("in_vertices"),
             py::arg("in_triangles"),
             py::arg("in_vertex_normals") = std::nullopt,
             py::arg("in_vertex_colors") = std::nullopt)
        .def_property_readonly("num_triangles", &TriangleMesh::get_num_triangles, R"doc(
        Total number of triangles.

        Returns:
            int - Total number of triangles.
        )doc")
        .def_property_readonly("vertices", &TriangleMesh::get_vertices, R"doc(
        Mesh vertices coordinates.

        Returns:
            torch.Tensor - Shape (N, 3) float32 tensor of vertices.
        )doc")
        .def_property_readonly("vertex_normals", &TriangleMesh::get_vertex_normals, R"doc(
        Area-weighted vertex normals.

        Returns:
            torch.Tensor - Shape (N, 3) float32 tensor of vertex normals.
        )doc")
        .def_property_readonly("vertex_colors", &TriangleMesh::get_vertex_colors, R"doc(
        Vertex colors.

        Returns:
            torch.Tensor - Shape (N, 3) float32 tensor of vertex colors.
        )doc")
        .def_property_readonly("triangles", &TriangleMesh::get_triangles, R"doc(
        Mesh triangles.

        Returns:
            torch.Tensor - Shape (M, 3) int32 tensor of triangles.
        )doc")
        .def_property_readonly("triangle_areas", &TriangleMesh::get_triangle_areas, R"doc(
        Areas of each triangle.

        Returns:
            torch.Tensor - Shape (M,) float32 tensor of triangle areas.
        )doc")
        .def_property_readonly("triangle_normals", &TriangleMesh::get_triangle_normals, R"doc(
        Normals of each triangle.

        Returns:
            torch.Tensor - Shape (M, 3) float32 tensor of triangle normals.
        )doc")
        .def_property_readonly("surface_area", &TriangleMesh::get_surface_area, R"doc(
        Total surface area of the mesh.

        Returns:
            torch.Tensor - Total surface area of the mesh.
        )doc")
        .def_property_readonly("bvh", &TriangleMesh::build_bvh, R"doc(
        The Bounding Volume Hierarchy built for this mesh.

        Returns:
            MeshBVH - The Bounding Volume Hierarchy built for this mesh.
        )doc")
        .def("build_bvh", &TriangleMesh::build_bvh, R"doc(
        Builds and returns the Bounding Volume Hierarchy for the mesh.

        Returns:
            MeshBVH: The constructed BVH object.
        )doc")
        .def("get_self_intersection", &TriangleMesh::get_self_intersection, R"doc(
        Finds all self-intersecting triangle pairs in the mesh.

        Returns:
            torch.Tensor: A tensor of shape (K, 2) containing pairs of intersecting triangle indices.
        )doc")
        .def("is_self_intersection", &TriangleMesh::is_self_intersection, R"doc(
        Checks whether the mesh contains any self-intersecting triangles.

        Returns:
            bool: True if there is at least one self-intersection, False otherwise.
        )doc")
        .def("get_ray_intersection", [](TriangleMesh &self, const torch::Tensor &ray_origins, const torch::Tensor &ray_dirs, bool return_distance) -> py::object
             {
                 auto result = self.get_ray_intersection(ray_origins, ray_dirs, return_distance);
                 if (return_distance) {
                     return py::cast(result);
                 } else {
                     return py::cast(std::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result)));
                 } }, py::arg("ray_origins"), py::arg("ray_dirs"), py::arg("return_distance") = false,
             R"doc(
        Computes intersections between a batch of rays and the mesh triangles.

        Args:
            ray_origins (torch.Tensor): Shape (R, 3) float32 tensor of ray origins.
            ray_dirs (torch.Tensor): Shape (R, 3) float32 tensor of ray directions.
            return_distance (bool, optional): If True, returns intersection distances. Defaults to False.

        Returns:
            tuple: (ray_ids, triangle_ids, intersect_points, [distances])
        )doc")
        .def("query_points", &TriangleMesh::query_points, py::arg("query_pts"), py::arg("return_sdf") = false, py::arg("return_prj_pts") = true, py::arg("sign_mode") = 0, py::arg("distance_mode") = 0,
             R"doc(
        Finds the closest triangles and computes distances/SDFs for query points.

        Args:
            query_pts (torch.Tensor): Shape (Q, 3) float32 tensor of query points.
            return_sdf (bool, optional): Whether to return Signed Distance Field values. Defaults to False.
            return_prj_pts (bool, optional): Whether to return projected points on the mesh. Defaults to True.
            sign_mode (int, optional): The method for computing signs. Defaults to 0.
            distance_mode (int, optional): Distance computation mode. Defaults to 0.

        Returns:
            tuple: (query_ids, triangle_ids, projected_points, distances)
        )doc")
        .def_property_readonly("edges", &TriangleMesh::get_edges, R"doc(
        Unique edges of the mesh.

        Returns:
            torch.Tensor - Shape (E, 2) int32 tensor of unique edges.
        )doc")
        .def_property_readonly("edge_to_triangle_offsets", &TriangleMesh::get_edge_to_triangle_offsets, R"doc(
        Edge to triangle offsets.

        Returns:
            torch.Tensor - Edge to triangle offsets.
        )doc")
        .def_property_readonly("edge_to_triangle_counts", &TriangleMesh::get_edge_to_triangle_counts, R"doc(
        Edge to triangle counts.

        Returns:
            torch.Tensor - Edge to triangle counts.
        )doc")
        .def_property_readonly("edge_to_triangle_indices", &TriangleMesh::get_edge_to_triangle_indices, R"doc(
        Edge to triangle indices.

        Returns:
            torch.Tensor - Edge to triangle indices.
        )doc")
        .def("compute_triangle_areas", &TriangleMesh::compute_triangle_areas, "Computes the areas of all triangles.")
        .def("compute_triangle_normals", &TriangleMesh::compute_triangle_normals, "Computes the normals for all triangles.")
        .def("compute_edges_to_triangle_map", &TriangleMesh::compute_edges_to_triangle_map, "Computes the edge-to-triangle connectivity map.")
        .def("get_edges_to_triangle_map", &TriangleMesh::get_edges_to_triangle_map, R"doc(
        Gets the edge to triangle connectivity mapping.

        Returns:
            tuple: (edges, offsets, counts, indices)
        )doc")
        .def("compute_vertices_to_triangle_map", &TriangleMesh::compute_vertices_to_triangle_map, "Computes the vertex-to-triangle connectivity map.")
        .def("get_vertices_to_triangle_map", &TriangleMesh::get_vertices_to_triangle_map, R"doc(
        Gets the vertex to triangle connectivity mapping.

        Returns:
            tuple: (offsets, counts, indices)
        )doc")
        .def_property_readonly("vertex_to_triangle_offsets", &TriangleMesh::get_vertex_to_triangle_offsets, R"doc(
        Vertex to triangle offsets.

        Returns:
            torch.Tensor - Vertex to triangle offsets.
        )doc")
        .def_property_readonly("vertex_to_triangle_counts", &TriangleMesh::get_vertex_to_triangle_counts, R"doc(
        Vertex to triangle counts.

        Returns:
            torch.Tensor - Vertex to triangle counts.
        )doc")
        .def_property_readonly("vertex_to_triangle_indices", &TriangleMesh::get_vertex_to_triangle_indices, R"doc(
        Vertex to triangle indices.

        Returns:
            torch.Tensor - Vertex to triangle indices.
        )doc")
        .def("is_edge_manifold", &TriangleMesh::is_edge_manifold, py::arg("allow_boundary_edge") = true, R"doc(
        Checks if the mesh is edge manifold.

        Args:
            allow_boundary_edge (bool, optional): Whether to permit boundary edges (count <= 2). Defaults to True.

        Returns:
            bool: True if the mesh is edge manifold.
        )doc")
        .def("is_vertex_manifold", &TriangleMesh::is_vertex_manifold, R"doc(
        Checks if the mesh is vertex manifold.

        Returns:
            bool: True if all vertices are manifold.
        )doc")
        .def("is_manifold", &TriangleMesh::is_manifold, py::arg("allow_boundary_edge") = true, R"doc(
        Checks if the mesh is fully manifold (edge, vertex, and no self-intersections).

        Args:
            allow_boundary_edge (bool, optional): Whether to permit boundary edges. Defaults to True.

        Returns:
            bool: True if fully manifold.
        )doc")
        .def("get_non_manifold_vertices", &TriangleMesh::get_non_manifold_vertices, R"doc(
        Gets all non-manifold vertices.

        Returns:
            torch.Tensor: Tensor of vertex indices that are non-manifold.
        )doc")
        .def("remove_triangles_by_mask", &TriangleMesh::remove_triangles_by_mask, py::arg("keep_mask"), R"doc(
        Removes triangles from the mesh based on a boolean mask.

        Args:
            keep_mask (torch.Tensor): Shape (M,) boolean tensor indicating which triangles to keep.
        )doc")
        .def("sample_points", &TriangleMesh::sample_points, py::arg("num_points"), py::arg("uniform") = false, py::arg("return_normals") = false, py::arg("return_colors") = false, py::arg("use_triangle_normal") = true, R"doc(
        Samples random points on the surface of the mesh.

        Args:
            num_points (int): The number of points to sample.
            uniform (bool, optional): If True, samples uniformly by area. Defaults to False.
            return_normals (bool, optional): If True, returns normals at sampled points. Defaults to False.
            return_colors (bool, optional): If True, returns colors at sampled points. Defaults to False.
            use_triangle_normal (bool, optional): If True, uses flat triangle normals instead of interpolated vertex normals. Defaults to True.

        Returns:
            tuple: (points, triangle_indices, [normals], [colors])
        )doc")
        .def("compute_vertex_normals", &TriangleMesh::compute_vertex_normals, "Computes area-weighted vertex normals.")
        .def_property_readonly("euler_characteristic", &TriangleMesh::get_euler_characteristic, R"doc(
        The Euler characteristic (V - E + F) of the mesh.

        Returns:
            int - The Euler characteristic (V - E + F) of the mesh.
        )doc")
        .def_property_readonly("genus", &TriangleMesh::get_genus, R"doc(
        The topological genus of the mesh.

        Returns:
            int - The topological genus of the mesh.
        )doc");
}