#include <torch/extension.h>
#include "../../primitive/triangle.h"
#include "../../check.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

inline float3 tensor_to_float3(const torch::Tensor& t) {
    TORCH_CHECK(t.dim() == 1 && t.size(0) == 3, "Tensor must be 1D with 3 elements");
    auto t_contig = t.contiguous().cpu().to(torch::kFloat32);
    float* ptr = t_contig.data_ptr<float>();
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

inline torch::Tensor float3_to_tensor(const float3& f) {
    return torch::tensor({f.x, f.y, f.z}, torch::dtype(torch::kFloat32));
}

void bind_primitive_triangle(py::module_& m) {
    py::class_<Triangle>(m, "Triangle")
        .def(py::init([](const torch::Tensor& v0, const torch::Tensor& v1, const torch::Tensor& v2) {
            return Triangle(tensor_to_float3(v0), tensor_to_float3(v1), tensor_to_float3(v2));
        }), py::arg("v0"), py::arg("v1"), py::arg("v2"))
        .def("is_intersect_ray", [](const Triangle& self, const Ray& ray) {
            float t_hit, u, v;
            bool hit = self.is_intersect_ray(ray, t_hit, u, v);
            return py::make_tuple(hit, t_hit, u, v);
        }, py::arg("ray"))
        .def("compute_closest_point", [](const Triangle& self, const torch::Tensor& p) {
            return float3_to_tensor(self.compute_closest_point(tensor_to_float3(p)));
        }, py::arg("p"))
        .def("compute_normal", [](const Triangle& self) {
            return float3_to_tensor(self.compute_normal());
        })
        .def("compute_area", &Triangle::compute_area)
        .def("sample_point", [](const Triangle& self, float r1, float r2) {
            return float3_to_tensor(self.sample_point(r1, r2));
        }, py::arg("r1"), py::arg("r2"))
        .def("compute_aabb", [](const Triangle& self) {
            float3 aabb_min, aabb_max;
            self.compute_aabb(aabb_min, aabb_max);
            return py::make_tuple(float3_to_tensor(aabb_min), float3_to_tensor(aabb_max));
        })
        .def("test_intersection", [](const Triangle& self, const Triangle& other) {
            return self.test_intersection(other);
        }, py::arg("other"))
        .def("is_obtuse", [](const Triangle& self) {
            return self.is_obtuse();
        })
        .def("compute_centroid", [](const Triangle& self) {
            return float3_to_tensor(self.compute_centroid());
        })
        .def("compute_circumcenter", [](const Triangle& self, bool strict_inside) {
            return float3_to_tensor(self.compute_circumcenter(strict_inside));
        }, py::arg("strict_inside") = false)
        .def("test_point_on_tria_plane", [](const Triangle& self, const torch::Tensor& p, float eps) {
            return self.test_point_on_tria_plane(tensor_to_float3(p), eps);
        }, py::arg("p"), py::arg("eps") = 1e-5f)
        .def("test_point_inside_on_tria_plane", [](const Triangle& self, const torch::Tensor& p) {
            return self.test_point_inside_on_tria_plane(tensor_to_float3(p));
        }, py::arg("p"))
        .def("test_point_inside", [](const Triangle& self, const torch::Tensor& p, float eps) {
            return self.test_point_inside(tensor_to_float3(p), eps);
        }, py::arg("p"), py::arg("eps") = 1e-5f)
        .def_property_readonly("v0", [](const Triangle& self) { return float3_to_tensor(self.v0); })
        .def_property_readonly("v1", [](const Triangle& self) { return float3_to_tensor(self.v1); })
        .def_property_readonly("v2", [](const Triangle& self) { return float3_to_tensor(self.v2); });
}
