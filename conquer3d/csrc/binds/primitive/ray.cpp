#include <torch/extension.h>
#include "../../primitive/ray.h"
#include "../../check.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

inline float3 tensor_to_float3_ray(const torch::Tensor& t) {
    TORCH_CHECK(t.dim() == 1 && t.size(0) == 3, "Tensor must be 1D with 3 elements");
    auto t_contig = t.contiguous().cpu().to(torch::kFloat32);
    float* ptr = t_contig.data_ptr<float>();
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

inline torch::Tensor float3_to_tensor_ray(const float3& f) {
    return torch::tensor({f.x, f.y, f.z}, torch::dtype(torch::kFloat32));
}

void bind_primitive_ray(py::module_& m) {
    py::class_<Ray>(m, "Ray")
        .def(py::init([](const torch::Tensor& origin, const torch::Tensor& dir) {
            return Ray(tensor_to_float3_ray(origin), tensor_to_float3_ray(dir));
        }), py::arg("origin"), py::arg("direction"))
        .def_property_readonly("origin", [](const Ray& self) { return float3_to_tensor_ray(self.origin); })
        .def_property_readonly("direction", [](const Ray& self) { return float3_to_tensor_ray(self.direction); })
        .def_property_readonly("inv_direction", [](const Ray& self) { return float3_to_tensor_ray(self.inv_direction); })
        .def_property_readonly("t_min", [](const Ray& self) { return self.t_min; })
        .def_property_readonly("t_max", [](const Ray& self) { return self.t_max; })
        .def("at", [](const Ray& self, float t) {
            return float3_to_tensor_ray(self.at(t));
        }, py::arg("t"))
        .def("is_intersect_aabb", [](const Ray& self, const torch::Tensor& aabb_min, const torch::Tensor& aabb_max) {
            float t_hit;
            bool hit = self.is_intersect_aabb(tensor_to_float3_ray(aabb_min), tensor_to_float3_ray(aabb_max), t_hit);
            return py::make_tuple(hit, t_hit);
        }, py::arg("aabb_min"), py::arg("aabb_max"));
}
