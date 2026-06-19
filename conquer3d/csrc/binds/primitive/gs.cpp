#include <torch/extension.h>
#include "../../primitive/gs.h"
#include "../../check.h"

#include <cuda_fp16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

torch::Tensor compute_gs_covi_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &rotations,
    const torch::Tensor &scales,
    const bool rotnorm,
    const float tol,
    const uint32_t level)
{
    CHECK_INPUT(means);
    CHECK_INPUT(rotations);
    CHECK_INPUT(scales);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(rotations.scalar_type() == torch::kFloat32, "rotations must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(rotations.size(1) == 4, "rotations must have shape (N, 4)");
    TORCH_CHECK(scales.size(1) == 3, "scales must have shape (N, 3)");

    auto options = means.options();
    torch::Tensor covi = torch::empty({num_gaussians, 6}, options);

    gs::compute_gs_covi(
        num_gaussians,
        reinterpret_cast<const float4 *>(rotations.data_ptr<float>()),
        reinterpret_cast<const float3 *>(scales.data_ptr<float>()),
        rotnorm,
        tol,
        level,
        reinterpret_cast<float *>(covi.data_ptr<float>()));
    
    return covi;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_gs_aabb_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &scales,
    const torch::Tensor &covis,
    const std::optional<torch::Tensor> &isos,
    const float iso,
    const float tol,
    const uint32_t level)
{
    CHECK_INPUT(means);
    CHECK_INPUT(scales);
    CHECK_INPUT(covis);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(scales.size(1) == 3, "scales must have shape (N, 3)");

    float *isos_ptr = nullptr;
    if (isos.has_value())
    {
        CHECK_INPUT(isos.value());
        TORCH_CHECK(isos.value().scalar_type() == torch::kFloat32, "isos must be float32");
        TORCH_CHECK(isos.value().size(0) == num_gaussians, "isos must have shape (N,)");
        isos_ptr = isos.value().data_ptr<float>();
    }

    auto options = means.options();
    torch::Tensor aabb_min = torch::empty({num_gaussians, 3}, options);
    torch::Tensor aabb_max = torch::empty({num_gaussians, 3}, options);
    torch::Tensor contact_points = torch::empty({num_gaussians, 9}, options);

    gs_aabb::compute_gs_aabb(
        num_gaussians,
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float3 *>(scales.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        isos_ptr,
        iso,
        tol,
        level,
        reinterpret_cast<float3 *>(aabb_min.data_ptr<float>()),
        reinterpret_cast<float3 *>(aabb_max.data_ptr<float>()),
        reinterpret_cast<float3 *>(contact_points.data_ptr<float>()));
    
    return std::make_tuple(aabb_min, aabb_max, contact_points);
}

torch::Tensor solve_gs_neighbor_mahalanobis_radius_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &covis,
    const int k)
{
    CHECK_INPUT(means);
    CHECK_INPUT(covis);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(covis.size(0) == num_gaussians && covis.size(1) == 6, "covis must have shape (N, 6)");

    uint32_t search_k = k + 1;

    TORCH_CHECK(search_k > 1, "k must be greater than 0");
    TORCH_CHECK(search_k <= 32, "Requested k exceeds the hardware register limit (MAX_K = 32).");
    TORCH_CHECK(search_k <= num_gaussians, "Requested k is larger than the number of points in the tree.");

    auto options = means.options();
    torch::Tensor isos = torch::empty({num_gaussians}, options.dtype(torch::kFloat32));

    gs::solve_gs_neighbor_mahalanobis_radius(
        num_gaussians,
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        search_k,
        reinterpret_cast<float *>(isos.data_ptr<float>()));

    return isos;
}

void bind_primitive_gs(py::module_ &m)
{
    m.def("compute_gs_covi_func", &compute_gs_covi_wrapper, "Compute covariance matrices for 3D Gaussians",
          py::arg("means"),
          py::arg("rotations"),
          py::arg("scales"),
          py::arg("rotnorm"),
          py::arg("tol"),
          py::arg("level"));
    m.def("solve_gs_neighbor_mahalanobis_radius_func", &solve_gs_neighbor_mahalanobis_radius_wrapper, "Solve Gaussian Neighbor Mahalanobis Radius",
          py::arg("means"),
          py::arg("covis"),
          py::arg("k"));
    m.def("compute_gs_aabb_func", &compute_gs_aabb_wrapper, "Compute AABB for 3D Gaussians",
          py::arg("means"),
          py::arg("scales"),
          py::arg("covis"),
          py::arg("isos"),
          py::arg("iso"),
          py::arg("tol"),
          py::arg("level"));
}