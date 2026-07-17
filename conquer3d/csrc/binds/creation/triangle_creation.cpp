#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../creation/triangle_creation.h"

namespace py = pybind11;

void bind_creation_triangle_creation(py::module_ &m)
{
    m.def("create_sphere", &triangle_creation::create_sphere,
          py::arg("sectors") = 32,
          py::arg("stacks") = 16,
          py::arg("radius") = 1.0f,
          R"doc(Creates a UV sphere returning (vertices, triangles) tensors on the CPU.

Args:
    sectors (int): The number of longitudinal sectors. Defaults to 32.
    stacks (int): The number of latitudinal stacks. Defaults to 16.
    radius (float): The radius of the sphere. Defaults to 1.0.

Returns:
    tuple: A tuple containing the vertices tensor and the triangles tensor.
)doc");

    m.def("create_tetrahedra", &triangle_creation::create_tetrahedra,
          py::arg("radius") = 1.0f,
          R"doc(Creates a regular tetrahedron returning (vertices, triangles) tensors on the CPU.

Args:
    radius (float): The circumscribed radius of the tetrahedron. Defaults to 1.0.

Returns:
    tuple: A tuple containing the vertices tensor and the triangles tensor.
)doc");
}
