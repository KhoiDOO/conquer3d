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
    m.def("dual_contouring", &dual_contouring_wrapper,
          py::arg("grid_vertices"), py::arg("voxels"), py::arg("sdf"),
          py::arg("grid_normals") = py::none(), py::arg("colors") = py::none(),
          py::arg("iso") = 0.0f, py::arg("quad_split") = true,
          R"pbdoc(
          Extracts a sharp-feature preserving surface mesh using Dual Contouring with GPU QEF solver (Ju et al. 2002).

          Args:
              grid_vertices (torch.Tensor): (N, 3) float32 corner coordinates on CUDA.
              voxels (torch.Tensor): (M, 8) int32 corner indices per voxel cell.
              sdf (torch.Tensor): (N,) float32 scalar SDF values on CUDA.
              grid_normals (torch.Tensor, optional): (N, 3) float32 explicit vertex normals for sharp CAD features. Defaults to None.
              colors (torch.Tensor, optional): (N, C) float32 vertex features/colors on CUDA. Defaults to None.
              iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.
              quad_split (bool, optional): If True, splits quads into Delaunay triangles; if False, returns quads. Defaults to True.

          Returns:
              Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                  - vertices (torch.Tensor): (V, 3) float32 extracted surface vertices.
                  - faces (torch.Tensor): (F, 3) int32 triangles if quad_split=True, or (Q, 4) quads if quad_split=False.
                  - [colors] (torch.Tensor, optional): (V, C) float32 interpolated colors.

          Example:
              >>> import torch
              >>> from conquer3d._C import dual_contouring
              >>> verts, faces, _ = dual_contouring(grid_verts, voxels, sdf, iso=0.0)
          )pbdoc");
    m.def("dual_contouring_backward", &dual_contouring_backward_wrapper,
          py::arg("grad_verts"), py::arg("grad_colors"), py::arg("grid_vertices"),
          py::arg("voxels"), py::arg("sdf"), py::arg("grid_normals") = py::none(),
          py::arg("colors") = py::none(), py::arg("iso") = 0.0f,
          R"pbdoc(
          Analytical backward gradient propagation for Dual Contouring w.r.t. SDF and colors.

          Args:
              grad_verts (torch.Tensor): Upstream gradient w.r.t. extracted vertices (V, 3).
              grad_colors (torch.Tensor, optional): Upstream gradient w.r.t. colors (V, C).
              grid_vertices (torch.Tensor): (N, 3) float32 corner coordinates on CUDA.
              voxels (torch.Tensor): (M, 8) int32 corner indices.
              sdf (torch.Tensor): (N,) float32 scalar SDF values on CUDA.
              grid_normals (torch.Tensor, optional): (N, 3) float32 normals on CUDA.
              colors (torch.Tensor, optional): (N, C) float32 vertex colors on CUDA.
              iso (float, optional): Isosurface threshold. Defaults to 0.0.

          Returns:
              Tuple[torch.Tensor, Optional[torch.Tensor]]: (grad_sdf, grad_colors)

          Example:
              >>> grad_sdf, grad_colors = dual_contouring_backward(g_verts, g_colors, grid_verts, voxels, sdf, iso=0.0)
          )pbdoc");
}
