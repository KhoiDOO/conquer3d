#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../data_structure/zcurve.h"
#include "../../check.h"

namespace py = pybind11;

torch::Tensor compute_zcurve_wrapper(torch::Tensor points)
{
    CHECK_INPUT(points);

    // We expect points to be [..., 3]
    auto num_points = points.numel() / 3;
    const float *p_points = points.data_ptr<float>();

    // Output is same shape as points but without the last dim
    auto sizes = points.sizes().vec();
    sizes.pop_back();

    torch::Tensor codes = torch::empty(sizes, points.options().dtype(torch::kInt64));
    int64_t *p_codes = codes.data_ptr<int64_t>();

    zcurve::compute_zcurve(p_points, num_points, p_codes);

    return codes;
}

void bind_ds_zcurve(py::module_ &m) {
    m.def("compute_zcurve", &compute_zcurve_wrapper,
          py::arg("points"),
          R"pbdoc(
          Computes 30-bit 3D Morton space-filling Z-curve codes for points in normalized [0, 1]^3.

          Args:
              points (torch.Tensor): (..., 3) float32 coordinates on CUDA with values in [0, 1].

          Returns:
              torch.Tensor: (...) int64 Morton space-filling codes with interleaved bits.

          Example:
              >>> import torch
              >>> from conquer3d._C import compute_zcurve
              >>> pts = torch.rand(1000, 3, device='cuda')
              >>> codes = compute_zcurve(pts)
          )pbdoc");
}
