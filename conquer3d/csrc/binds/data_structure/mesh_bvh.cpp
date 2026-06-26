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
             "Check if there are any self-intersecting triangle pairs")
        .def("get_ray_intersection", [](MeshBVH& self, const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& vertices, const torch::Tensor& triangles, bool return_distance) -> py::object {
                 auto result = self.get_ray_intersection(ray_origins, ray_dirs, vertices, triangles, return_distance);
                 if (return_distance) {
                     return py::cast(result);
                 } else {
                     return py::cast(std::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result)));
                 }
             },
             py::arg("ray_origins"),
             py::arg("ray_dirs"),
             py::arg("vertices"),
             py::arg("triangles"),
             py::arg("return_distance") = false,
             "Find all ray-triangle intersections. Returns (ray_ids, triangle_ids, intersect_points, [distances])")
        .def("query_point", &MeshBVH::query_point,
             py::arg("query_points"),
             py::arg("vertices"),
             py::arg("triangles"),
             py::arg("return_sdf") = false,
             py::arg("return_prj_pts") = true,
             py::arg("sign_mode") = 0,
             "Find the closest triangle to each query point. Returns (query_ids, triangle_ids, projected_points, distances)");
}
