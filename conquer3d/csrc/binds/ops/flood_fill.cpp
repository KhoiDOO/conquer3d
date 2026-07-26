#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../ops/flood_fill.h"
#include "../../check.h"

namespace py = pybind11;

torch::Tensor compute_flood_fill_wrapper(
    torch::Tensor active_voxel_ids,
    std::vector<int64_t> res,
    int connectivity
) {
    CHECK_INPUT(active_voxel_ids);
    TORCH_CHECK(res.size() == 3, "res must have 3 elements.");
    return ops::compute_flood_fill(active_voxel_ids, res[0] - 1, res[1] - 1, res[2] - 1, connectivity);
}

void bind_ops_flood_fill(py::module_& m) {
    m.def("compute_flood_fill", &compute_flood_fill_wrapper, "Compute 3D Flood Fill from active voxel IDs",
          py::arg("active_voxel_ids"), py::arg("res"), py::arg("connectivity") = 6);
}
