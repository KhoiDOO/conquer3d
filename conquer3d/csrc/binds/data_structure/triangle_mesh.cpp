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
        reinterpret_cast<float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float3 *>(this->triangle_normals.data_ptr<float>()));
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
    return triangle_mesh::get_non_manifold_vertices_cuda(
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
        .def("get_non_manifold_vertices", &TriangleMesh::get_non_manifold_vertices)
        .def("remove_triangles_by_mask", &TriangleMesh::remove_triangles_by_mask, py::arg("keep_mask"))
        .def_property_readonly("euler_characteristic", &TriangleMesh::get_euler_characteristic)
        .def_property_readonly("genus", &TriangleMesh::get_genus);
}