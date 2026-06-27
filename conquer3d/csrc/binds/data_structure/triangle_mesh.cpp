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
    if (distance_mode == 0) {
        this->build_bvh();
        return this->bvh.value().query_point(query_pts, this->vertices, this->triangles, return_sdf, return_prj_pts, sign_mode);
    } else {
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
    if (this->num_triangles == 0) {
        throw std::runtime_error("Cannot sample points from an empty mesh.");
    }

    torch::Tensor tri_indices;
    if (!uniform) {
        tri_indices = torch::randint(0, this->num_triangles, {num_points}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kInt64));
    } else {
        torch::Tensor areas = this->get_triangle_areas();
        tri_indices = torch::multinomial(areas, num_points, true);
    }

    torch::Tensor r1_r2 = torch::rand({num_points, 2}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));
    torch::Tensor out_points = torch::empty({num_points, 3}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));

    torch::Tensor out_normals, out_colors;
    const float3* d_vertex_normals = nullptr;
    const float3* d_triangle_normals = nullptr;
    const float3* d_vertex_colors = nullptr;
    float3* d_out_normals = nullptr;
    float3* d_out_colors = nullptr;

    if (return_normals) {
        out_normals = torch::empty({num_points, 3}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));
        d_out_normals = reinterpret_cast<float3 *>(out_normals.data_ptr<float>());
        if (use_triangle_normal) {
            d_triangle_normals = reinterpret_cast<const float3 *>(this->get_triangle_normals().data_ptr<float>());
        } else {
            d_vertex_normals = reinterpret_cast<const float3 *>(this->get_vertex_normals().data_ptr<float>());
        }
    }

    if (return_colors) {
        if (!this->vertex_colors.defined()) {
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

torch::Tensor TriangleMesh::get_vertex_to_triangle_indices() {
    this->compute_vertices_to_triangle_map();
    return this->vertex_to_triangle_indices;
}

torch::Tensor TriangleMesh::get_non_manifold_vertices() {
    this->compute_vertices_to_triangle_map();
    return triangle_mesh::get_non_manifold_vertices(
        this->vertices.size(0),
        this->triangles,
        this->vertex_to_triangle_offsets,
        this->vertex_to_triangle_counts,
        this->vertex_to_triangle_indices
    );
}

bool TriangleMesh::is_vertex_manifold() {
    torch::Tensor nm_vertices = this->get_non_manifold_vertices();
    return nm_vertices.size(0) == 0;
}

bool TriangleMesh::is_manifold(bool allow_boundary_edge) {
    return this->is_edge_manifold(allow_boundary_edge) && this->is_vertex_manifold() && !this->is_self_intersection();
}

void bind_ds_triangle_mesh(py::module_ &m)
{
    py::class_<TriangleMesh>(m, "TriangleMesh")
        .def(py::init<const torch::Tensor &, const torch::Tensor &, std::optional<torch::Tensor>, std::optional<torch::Tensor>>(),
             py::arg("in_vertices"),
             py::arg("in_triangles"),
             py::arg("in_vertex_normals") = std::nullopt,
             py::arg("in_vertex_colors") = std::nullopt)
        .def_property_readonly("num_triangles", &TriangleMesh::get_num_triangles)
        .def_property_readonly("vertices", &TriangleMesh::get_vertices)
        .def_property_readonly("vertex_normals", &TriangleMesh::get_vertex_normals)
        .def_property_readonly("vertex_colors", &TriangleMesh::get_vertex_colors)
        .def_property_readonly("triangles", &TriangleMesh::get_triangles)
        .def_property_readonly("triangle_areas", &TriangleMesh::get_triangle_areas)
        .def_property_readonly("triangle_normals", &TriangleMesh::get_triangle_normals)
        .def_property_readonly("surface_area", &TriangleMesh::get_surface_area)
        .def_property_readonly("bvh", &TriangleMesh::build_bvh)
        .def("build_bvh", &TriangleMesh::build_bvh)
        .def("get_self_intersection", &TriangleMesh::get_self_intersection,
             "Find all self-intersecting triangle pairs")
        .def("is_self_intersection", &TriangleMesh::is_self_intersection,
             "Check if there are any self-intersecting triangle pairs")
        .def("get_ray_intersection", [](TriangleMesh& self, const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, bool return_distance) -> py::object {
                 auto result = self.get_ray_intersection(ray_origins, ray_dirs, return_distance);
                 if (return_distance) {
                     return py::cast(result);
                 } else {
                     return py::cast(std::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result)));
                 }
             },
             py::arg("ray_origins"),
             py::arg("ray_dirs"),
             py::arg("return_distance") = false,
             "Find all ray-triangle intersections. Returns (ray_ids, triangle_ids, intersect_points, [distances])")
        .def("query_points", &TriangleMesh::query_points,
             py::arg("query_pts"),
             py::arg("return_sdf") = false,
             py::arg("return_prj_pts") = true,
             py::arg("sign_mode") = 0,
             py::arg("distance_mode") = 0,
             "Query points against the mesh. Returns (query_ids, triangle_ids, projected_points, distances)")
        .def_property_readonly("edges", &TriangleMesh::get_edges)
        .def_property_readonly("edge_to_triangle_offsets", &TriangleMesh::get_edge_to_triangle_offsets)
        .def_property_readonly("edge_to_triangle_counts", &TriangleMesh::get_edge_to_triangle_counts)
        .def_property_readonly("edge_to_triangle_indices", &TriangleMesh::get_edge_to_triangle_indices)
        .def("compute_triangle_areas", &TriangleMesh::compute_triangle_areas)
        .def("compute_triangle_normals", &TriangleMesh::compute_triangle_normals)
        .def("compute_edges_to_triangle_map", &TriangleMesh::compute_edges_to_triangle_map)
        .def("get_edges_to_triangle_map", &TriangleMesh::get_edges_to_triangle_map)
        .def("compute_vertices_to_triangle_map", &TriangleMesh::compute_vertices_to_triangle_map)
        .def("get_vertices_to_triangle_map", &TriangleMesh::get_vertices_to_triangle_map)
        .def_property_readonly("vertex_to_triangle_offsets", &TriangleMesh::get_vertex_to_triangle_offsets)
        .def_property_readonly("vertex_to_triangle_counts", &TriangleMesh::get_vertex_to_triangle_counts)
        .def_property_readonly("vertex_to_triangle_indices", &TriangleMesh::get_vertex_to_triangle_indices)
        .def("is_edge_manifold", &TriangleMesh::is_edge_manifold, py::arg("allow_boundary_edge") = true)
        .def("is_vertex_manifold", &TriangleMesh::is_vertex_manifold)
        .def("is_manifold", &TriangleMesh::is_manifold, py::arg("allow_boundary_edge") = true)
        .def("get_non_manifold_vertices", &TriangleMesh::get_non_manifold_vertices)
        .def("remove_triangles_by_mask", &TriangleMesh::remove_triangles_by_mask, py::arg("keep_mask"))
        .def("sample_points", &TriangleMesh::sample_points, py::arg("num_points"), py::arg("uniform") = false, py::arg("return_normals") = false, py::arg("return_colors") = false, py::arg("use_triangle_normal") = true, "Sample points on the mesh")
        .def("compute_vertex_normals", &TriangleMesh::compute_vertex_normals)
        .def_property_readonly("euler_characteristic", &TriangleMesh::get_euler_characteristic)
        .def_property_readonly("genus", &TriangleMesh::get_genus);
}