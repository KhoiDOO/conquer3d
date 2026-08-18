#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/mca.h"
#include "../../check.h"

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> marching_cubes_asymptotic_wrapper(
    torch::Tensor grid_vertices,
    torch::Tensor voxels,
    torch::Tensor sdf,
    std::optional<torch::Tensor> colors,
    float iso
) {
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(voxels);
    CHECK_INPUT(sdf);
    if (colors.has_value() && colors.value().defined()) {
        CHECK_INPUT(colors.value());
    }

    return mca::marching_cubes_asymptotic(
        grid_vertices,
        voxels,
        sdf,
        colors,
        iso
    );
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> marching_cubes_asymptotic_backward_wrapper(
    torch::Tensor grad_vertices,
    std::optional<torch::Tensor> grad_colors,
    torch::Tensor grid_vertices,
    torch::Tensor unique_edges,
    torch::Tensor sdf,
    std::optional<torch::Tensor> colors,
    float iso
) {
    CHECK_INPUT(grad_vertices);
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(unique_edges);
    CHECK_INPUT(sdf);

    if (grad_colors.has_value() && grad_colors.value().defined()) {
        CHECK_INPUT(grad_colors.value());
    }
    if (colors.has_value() && colors.value().defined()) {
        CHECK_INPUT(colors.value());
    }

    return mca::marching_cubes_asymptotic_backward(
        grad_vertices,
        grad_colors,
        grid_vertices,
        unique_edges,
        sdf,
        colors,
        iso
    );
}

void bind_ops_mca(py::module &m) {
    m.def(
        "marching_cubes_asymptotic",
        &marching_cubes_asymptotic_wrapper,
        "Extract watertight isosurface using Marching Cubes with Asymptotic Decider (CUDA)",
        py::arg("grid_vertices"),
        py::arg("voxels"),
        py::arg("sdf"),
        py::arg("colors") = py::none(),
        py::arg("iso") = 0.0f
    );
    m.def(
        "marching_cubes_asymptotic_backward",
        &marching_cubes_asymptotic_backward_wrapper,
        "Backward gradient propagation for Marching Cubes with Asymptotic Decider (CUDA)",
        py::arg("grad_vertices"),
        py::arg("grad_colors"),
        py::arg("grid_vertices"),
        py::arg("unique_edges"),
        py::arg("sdf"),
        py::arg("colors"),
        py::arg("iso") = 0.0f
    );
}
