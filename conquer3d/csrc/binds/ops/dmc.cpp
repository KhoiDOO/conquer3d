#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/dmc.h"
#include "../../check.h"

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> dual_marching_cubes_wrapper(
    torch::Tensor grid_vertices,
    torch::Tensor voxels,
    torch::Tensor sdf,
    std::optional<torch::Tensor> colors,
    float iso,
    bool quad_split,
    int project_iters
) {
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(voxels);
    CHECK_INPUT(sdf);

    if (colors.has_value() && colors.value().defined()) {
        CHECK_INPUT(colors.value());
    }

    return conquer3d::ops::dual_marching_cubes(
        grid_vertices,
        voxels,
        sdf,
        colors,
        iso,
        quad_split,
        project_iters
    );
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> dual_marching_cubes_backward_wrapper(
    torch::Tensor grad_verts,
    std::optional<torch::Tensor> grad_colors,
    torch::Tensor grid_vertices,
    torch::Tensor voxels,
    torch::Tensor sdf,
    std::optional<torch::Tensor> colors,
    float iso,
    int project_iters
) {
    CHECK_INPUT(grad_verts);
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(voxels);
    CHECK_INPUT(sdf);

    if (grad_colors.has_value() && grad_colors.value().defined()) {
        CHECK_INPUT(grad_colors.value());
    }
    if (colors.has_value() && colors.value().defined()) {
        CHECK_INPUT(colors.value());
    }

    return conquer3d::ops::dual_marching_cubes_backward(
        grad_verts,
        grad_colors,
        grid_vertices,
        voxels,
        sdf,
        colors,
        iso,
        project_iters
    );
}

void bind_ops_dmc(py::module &m) {
    m.def(
        "dual_marching_cubes",
        &dual_marching_cubes_wrapper,
        "Extract 2-manifold surface mesh using Differentiable Dual Marching Cubes (CUDA)",
        py::arg("grid_vertices"),
        py::arg("voxels"),
        py::arg("sdf"),
        py::arg("colors") = py::none(),
        py::arg("iso") = 0.0f,
        py::arg("quad_split") = true,
        py::arg("project_iters") = 5
    );
    m.def(
        "dual_marching_cubes_backward",
        &dual_marching_cubes_backward_wrapper,
        "Backward gradient propagation for Dual Marching Cubes (CUDA)",
        py::arg("grad_verts"),
        py::arg("grad_colors"),
        py::arg("grid_vertices"),
        py::arg("voxels"),
        py::arg("sdf"),
        py::arg("colors") = py::none(),
        py::arg("iso") = 0.0f,
        py::arg("project_iters") = 5
    );
}
