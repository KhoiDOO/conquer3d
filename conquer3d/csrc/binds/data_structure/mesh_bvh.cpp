#include <torch/extension.h>
#include "../../data_structure/mesh_bvh.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_ds_mesh_bvh(py::module_& m)
{
    py::class_<MeshBVH, BVH>(m, "MeshBVH")
        .def(py::init<const torch::Tensor &, const torch::Tensor &>(),
             py::arg("in_aabb_mins"), 
             py::arg("in_aabb_maxs"),
             "Construct MeshBVH from Triangle AABBs.")
        .def("get_self_intersection", &MeshBVH::get_self_intersection,
             py::arg("vertices"), 
             py::arg("triangles"),
             "Find all self-intersecting triangle pairs")
        .def("is_self_intersection", &MeshBVH::is_self_intersection,
             py::arg("vertices"), 
             py::arg("triangles"),
             "Check if there are any self-intersecting triangle pairs");
}
