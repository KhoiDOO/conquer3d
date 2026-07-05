#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/mc.h"
#include "../../check.h"

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>> marching_cubes_wrapper(
    torch::Tensor grid_vertices,
    torch::Tensor voxels,
    torch::Tensor voxel_values,
    std::optional<torch::Tensor> grid_normals,
    std::optional<torch::Tensor> grid_colors,
    float iso,
    bool return_unique_edges
) {
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(voxels);
    CHECK_INPUT(voxel_values);

    uint32_t num_voxels = voxels.size(0);

    const float3* __restrict__ p_grid_vertices = (float3*)grid_vertices.data_ptr<float>();
    const uint32_t* __restrict__ p_voxels = (uint32_t*)voxels.data_ptr<int32_t>();
    const float* __restrict__ p_voxel_values = voxel_values.data_ptr<float>();

    const float3* __restrict__ p_grid_normals = nullptr;
    if (grid_normals.has_value()) {
        CHECK_INPUT(grid_normals.value());
        p_grid_normals = (float3*)grid_normals.value().data_ptr<float>();
    }

    const float3* __restrict__ p_grid_colors = nullptr;
    if (grid_colors.has_value()) {
        CHECK_INPUT(grid_colors.value());
        p_grid_colors = (float3*)grid_colors.value().data_ptr<float>();
    }

    return mc::marching_cubes(
        num_voxels,
        p_grid_vertices,
        p_voxels,
        p_voxel_values,
        p_grid_normals,
        p_grid_colors,
        iso,
        grid_vertices.options(),
        voxels.options(),
        return_unique_edges
    );
}

void marching_cubes_backward_wrapper(
    torch::Tensor unique_edges,
    torch::Tensor grid_vertices,
    std::optional<torch::Tensor> grid_colors,
    torch::Tensor values,
    torch::Tensor adj_verts,
    std::optional<torch::Tensor> adj_colors,
    torch::Tensor adj_values,
    std::optional<torch::Tensor> adj_grid_colors,
    float iso
) {
    CHECK_INPUT(unique_edges);
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(values);
    CHECK_INPUT(adj_verts);
    CHECK_INPUT(adj_values);

    uint32_t n_verts = unique_edges.size(0);

    const Edge* p_unique_edges = (const Edge*)unique_edges.data_ptr<int32_t>();
    const float3* p_grid_vertices = (const float3*)grid_vertices.data_ptr<float>();
    const float* p_values = values.data_ptr<float>();
    const float3* p_adj_verts = (const float3*)adj_verts.data_ptr<float>();
    float* p_adj_values = adj_values.data_ptr<float>();

    const float3* p_grid_colors = nullptr;
    if (grid_colors.has_value()) {
        CHECK_INPUT(grid_colors.value());
        p_grid_colors = (const float3*)grid_colors.value().data_ptr<float>();
    }

    const float3* p_adj_colors = nullptr;
    if (adj_colors.has_value()) {
        CHECK_INPUT(adj_colors.value());
        p_adj_colors = (const float3*)adj_colors.value().data_ptr<float>();
    }

    float3* p_adj_grid_colors = nullptr;
    if (adj_grid_colors.has_value()) {
        CHECK_INPUT(adj_grid_colors.value());
        p_adj_grid_colors = (float3*)adj_grid_colors.value().data_ptr<float>();
    }

    mc::backward(
        n_verts,
        p_unique_edges,
        p_grid_vertices,
        p_grid_colors,
        p_values,
        p_adj_verts,
        p_adj_colors,
        p_adj_values,
        p_adj_grid_colors,
        iso
    );
}

void bind_ops_mc(py::module_& m) {
    m.def("marching_cubes", &marching_cubes_wrapper, "Marching Cubes",
          py::arg("grid_vertices"), py::arg("voxels"), py::arg("voxel_values"),
          py::arg("grid_normals") = std::nullopt, py::arg("grid_colors") = std::nullopt, py::arg("iso") = 0.0f, py::arg("return_unique_edges") = false);

    m.def("marching_cubes_backward", &marching_cubes_backward_wrapper, "Marching Cubes Backward",
          py::arg("unique_edges"), py::arg("grid_vertices"), py::arg("grid_colors"), py::arg("values"),
          py::arg("adj_verts"), py::arg("adj_colors"), py::arg("adj_values"), py::arg("adj_grid_colors"), py::arg("iso"));
}
