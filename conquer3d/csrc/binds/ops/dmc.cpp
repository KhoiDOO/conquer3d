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
    m.def("dual_marching_cubes", &dual_marching_cubes_wrapper,
          py::arg("grid_vertices"), py::arg("voxels"), py::arg("sdf"),
          py::arg("colors") = py::none(), py::arg("iso") = 0.0f,
          py::arg("quad_split") = true, py::arg("project_iters") = 5,
          R"pbdoc(
          Extracts a watertight 2-manifold surface mesh using Differentiable Dual Marching Cubes (Schaefer & Warren 2004).

          Args:
              grid_vertices (torch.Tensor): (N, 3) float32 corner coordinates on CUDA.
              voxels (torch.Tensor): (M, 8) int32 corner indices per voxel cell.
              sdf (torch.Tensor): (N,) float32 scalar SDF values on CUDA.
              colors (torch.Tensor, optional): (N, C) float32 vertex features/colors on CUDA. Defaults to None.
              iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.
              quad_split (bool, optional): If True, splits into Delaunay triangles; if False, returns quads. Defaults to True.
              project_iters (int, optional): Newton-Raphson level-set projection iterations. Defaults to 5.

          Returns:
              Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                  - vertices (torch.Tensor): (V, 3) float32 extracted surface vertices.
                  - faces (torch.Tensor): (F, 3) int32 triangles if quad_split=True, or (Q, 4) quads if quad_split=False.
                  - [colors] (torch.Tensor, optional): (V, C) float32 interpolated colors.

          Example:
              >>> import torch
              >>> from conquer3d._C import dual_marching_cubes
              >>> verts, faces, _ = dual_marching_cubes(grid_verts, voxels, sdf, iso=0.0)
          )pbdoc");
    m.def("dual_marching_cubes_backward", &dual_marching_cubes_backward_wrapper,
          py::arg("grad_verts"), py::arg("grad_colors"), py::arg("grid_vertices"),
          py::arg("voxels"), py::arg("sdf"), py::arg("colors") = py::none(),
          py::arg("iso") = 0.0f, py::arg("project_iters") = 5,
          R"pbdoc(
          Analytical backward gradient propagation for Dual Marching Cubes w.r.t. SDF and colors.

          Args:
              grad_verts (torch.Tensor): Upstream gradient w.r.t. extracted vertices (V, 3).
              grad_colors (torch.Tensor, optional): Upstream gradient w.r.t. colors (V, C).
              grid_vertices (torch.Tensor): (N, 3) float32 corner coordinates on CUDA.
              voxels (torch.Tensor): (M, 8) int32 corner indices.
              sdf (torch.Tensor): (N,) float32 scalar SDF values on CUDA.
              colors (torch.Tensor, optional): (N, C) float32 vertex features/colors on CUDA.
              iso (float, optional): Isosurface threshold. Defaults to 0.0.
              project_iters (int, optional): Newton-Raphson iterations. Defaults to 5.

          Returns:
              Tuple[torch.Tensor, Optional[torch.Tensor]]: (grad_sdf, grad_colors)

          Example:
              >>> grad_sdf, grad_colors = dual_marching_cubes_backward(g_verts, g_colors, grid_verts, voxels, sdf, iso=0.0)
          )pbdoc");
}
