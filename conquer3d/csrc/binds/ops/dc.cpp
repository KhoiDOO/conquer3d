#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/dc.h"
#include "../../check.h"

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> dual_contouring_wrapper(
    torch::Tensor grid_vertices,
    torch::Tensor voxels,
    torch::Tensor sdf,
    std::optional<torch::Tensor> grid_normals,
    std::optional<torch::Tensor> colors,
    float iso,
    bool quad_split
) {
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(voxels);
    CHECK_INPUT(sdf);

    if (grid_normals.has_value() && grid_normals.value().defined()) {
        CHECK_INPUT(grid_normals.value());
    }
    if (colors.has_value() && colors.value().defined()) {
        CHECK_INPUT(colors.value());
    }

    return conquer3d::ops::dual_contouring(
        grid_vertices,
        voxels,
        sdf,
        grid_normals,
        colors,
        iso,
        quad_split
    );
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> dual_contouring_backward_wrapper(
    torch::Tensor grad_verts,
    std::optional<torch::Tensor> grad_colors,
    torch::Tensor grid_vertices,
    torch::Tensor voxels,
    torch::Tensor sdf,
    std::optional<torch::Tensor> grid_normals,
    std::optional<torch::Tensor> colors,
    float iso
) {
    CHECK_INPUT(grad_verts);
    CHECK_INPUT(grid_vertices);
    CHECK_INPUT(voxels);
    CHECK_INPUT(sdf);

    if (grad_colors.has_value() && grad_colors.value().defined()) {
        CHECK_INPUT(grad_colors.value());
    }
    if (grid_normals.has_value() && grid_normals.value().defined()) {
        CHECK_INPUT(grid_normals.value());
    }
    if (colors.has_value() && colors.value().defined()) {
        CHECK_INPUT(colors.value());
    }

    return conquer3d::ops::dual_contouring_backward(
        grad_verts,
        grad_colors,
        grid_vertices,
        voxels,
        sdf,
        grid_normals,
        colors,
        iso
    );
}

void bind_ops_dc(py::module &m) {
    m.def(
        "dual_contouring",
        &dual_contouring_wrapper,
        "Extract surface mesh using Differentiable Dual Contouring with GPU QEF (CUDA)",
        py::arg("grid_vertices"),
        py::arg("voxels"),
        py::arg("sdf"),
        py::arg("grid_normals") = py::none(),
        py::arg("colors") = py::none(),
        py::arg("iso") = 0.0f,
        py::arg("quad_split") = true
    );
    m.def(
        "dual_contouring_backward",
        &dual_contouring_backward_wrapper,
        "Backward gradient propagation for Dual Contouring (CUDA)",
        py::arg("grad_verts"),
        py::arg("grad_colors"),
        py::arg("grid_vertices"),
        py::arg("voxels"),
        py::arg("sdf"),
        py::arg("grid_normals") = py::none(),
        py::arg("colors") = py::none(),
        py::arg("iso") = 0.0f
    );
}
