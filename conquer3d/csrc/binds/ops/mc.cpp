#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/mc.h"
#include "../../check.h"

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> marching_cubes_wrapper(
    torch::Tensor grid_vertices,
    torch::Tensor voxels,
    torch::Tensor voxel_values,
    std::optional<torch::Tensor> grid_normals,
    float iso
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

    return mc::marching_cubes(
        num_voxels,
        p_grid_vertices,
        p_voxels,
        p_voxel_values,
        p_grid_normals,
        iso,
        grid_vertices.options(),
        voxels.options()
    );
}

void bind_ops_mc(py::module_& m) {
    m.def("marching_cubes", &marching_cubes_wrapper, "Marching Cubes",
          py::arg("grid_vertices"), py::arg("voxels"), py::arg("voxel_values"),
          py::arg("grid_normals") = std::nullopt, py::arg("iso") = 0.0f);
}
