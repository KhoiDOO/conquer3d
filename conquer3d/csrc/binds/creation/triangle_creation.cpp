#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../creation/triangle_creation.h"

namespace py = pybind11;

void bind_creation_triangle_creation(py::module_ &m) {
    m.def("create_sphere", &triangle_creation::create_sphere,
          py::arg("sectors") = 32, py::arg("stacks") = 16, py::arg("radius") = 1.0f,
          R"pbdoc(
          Generates a parameterized UV sphere mesh on CPU.

          Args:
              sectors (int, optional): Number of longitudinal angular slices. Defaults to 32.
              stacks (int, optional): Number of latitudinal vertical stacks. Defaults to 16.
              radius (float, optional): Radius of the sphere. Defaults to 1.0.

          Returns:
              Tuple[torch.Tensor, torch.Tensor]:
                  - vertices (torch.Tensor): ((sectors-1)*stacks + 2, 3) float32 coordinates on CPU.
                  - triangles (torch.Tensor): (2*(stacks-1)*sectors, 3) int32 triangle vertex indices on CPU.

          Example:
              >>> import torch
              >>> from conquer3d._C import create_sphere
              >>> verts, tris = create_sphere(sectors=32, stacks=16, radius=1.0)
          )pbdoc");
    m.def("create_tetrahedra", &triangle_creation::create_tetrahedra,
          py::arg("radius") = 1.0f,
          R"pbdoc(
          Generates a regular 4-faced tetrahedron mesh inscribed in a sphere of given radius on CPU.

          Args:
              radius (float, optional): Circumscribed sphere radius. Defaults to 1.0.

          Returns:
              Tuple[torch.Tensor, torch.Tensor]:
                  - vertices (torch.Tensor): (4, 3) float32 vertex coordinates on CPU.
                  - triangles (torch.Tensor): (4, 3) int32 triangle vertex indices on CPU.

          Example:
              >>> import torch
              >>> from conquer3d._C import create_tetrahedra
              >>> verts, tris = create_tetrahedra(radius=1.0)
          )pbdoc");
}
