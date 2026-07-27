#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../ops/flood_fill.h"
#include "../../check.h"

namespace py = pybind11;

torch::Tensor compute_flood_fill_wrapper(
    torch::Tensor vertices,
    torch::Tensor triangles,
    torch::Tensor aabb_mins,
    torch::Tensor aabb_maxs,
    torch::Tensor bvh_children,
    torch::Tensor object_ids,
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> grid_res,
    int connectivity
) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(triangles);
    CHECK_INPUT(aabb_mins);
    CHECK_INPUT(aabb_maxs);
    CHECK_INPUT(bvh_children);
    CHECK_INPUT(object_ids);
    TORCH_CHECK(grid_min.size() == 3, "grid_min must have 3 elements.");
    TORCH_CHECK(grid_max.size() == 3, "grid_max must have 3 elements.");
    TORCH_CHECK(grid_res.size() == 3, "grid_res must have 3 elements.");
    return ops::compute_flood_fill(vertices, triangles, aabb_mins, aabb_maxs, bvh_children, object_ids, grid_min, grid_max, grid_res, connectivity);
}

void bind_ops_flood_fill(py::module_& m) {
    m.def("compute_flood_fill", &compute_flood_fill_wrapper, "Compute 3D Flood Fill on grid vertices using segment BVH intersection",
          py::arg("vertices"), py::arg("triangles"), py::arg("aabb_mins"), py::arg("aabb_maxs"),
          py::arg("bvh_children"), py::arg("object_ids"), py::arg("grid_min"), py::arg("grid_max"),
          py::arg("grid_res"), py::arg("connectivity") = 6);
}
