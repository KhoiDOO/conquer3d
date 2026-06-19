#include <torch/extension.h>
#include "../../primitive/pgs.h"
#include "../../check.h"

#include <cuda_fp16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor> solve_pgs_cluster_tangency_radius_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &normals,
    const torch::Tensor &covis,
    const uint32_t k
)
{
    CHECK_INPUT(means);
    CHECK_INPUT(normals);
    CHECK_INPUT(covis);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(normals.scalar_type() == torch::kFloat32, "normals must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(normals.size(0) == num_gaussians && normals.size(1) == 3, "normals must have shape (N, 3)");
    TORCH_CHECK(covis.size(0) == num_gaussians && covis.size(1) == 6, "covis must have shape (N, 6)");

    uint32_t search_k = k + 1;

    TORCH_CHECK(search_k > 1, "k must be greater than 0");
    TORCH_CHECK(search_k <= 32, "Requested k exceeds the hardware register limit (MAX_K = 32).");
    TORCH_CHECK(search_k <= num_gaussians, "Requested k is larger than the number of points in the tree.");

    auto options = means.options();
    torch::Tensor isos = torch::empty({num_gaussians}, options.dtype(torch::kFloat32));
    torch::Tensor invalid_mask = torch::empty({num_gaussians}, options.dtype(torch::kBool));

    pgs::solve_pgs_cluster_tangency_radius(
        num_gaussians,
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float3 *>(normals.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        search_k,
        reinterpret_cast<float *>(isos.data_ptr<float>()),
        reinterpret_cast<bool *>(invalid_mask.data_ptr<bool>()));

    return std::make_tuple(isos, invalid_mask);
}

void bind_primitive_pgs(py::module_ &m)
{
    m.def("solve_pgs_cluster_tangency_radius_func", &solve_pgs_cluster_tangency_radius_wrapper,
          py::arg("means"),
          py::arg("normals"),
          py::arg("covis"),
          py::arg("k") = 16);
}