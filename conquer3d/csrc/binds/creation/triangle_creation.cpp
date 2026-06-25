#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../creation/triangle_creation.h"

namespace py = pybind11;

void bind_creation_triangle_creation(py::module_& m) {
    m.def("create_sphere", &triangle_creation::create_sphere, 
          py::arg("sectors") = 32, 
          py::arg("stacks") = 16, 
          py::arg("radius") = 1.0f,
          "Creates a UV sphere returning (vertices, triangles) tensors on the CPU.");
}
