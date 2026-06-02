#include "../../primitive/gs.h"

#include <cuda_fp16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

// Helper macros to check tensor properties
#define CHECK_CUDA(x) \
    AT_ASSERTM((x).options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    AT_ASSERTM((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> get_aabb_wrapper(
    const torch::Tensor& means,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const float iso,
    const float tol,
    const uint32_t level,
    const bool rotnorm)
{
    // 1. Enforce memory contiguity and CUDA residency
    CHECK_INPUT(means);
    CHECK_INPUT(rotations);
    CHECK_INPUT(scales);

    // 2. Enforce Data Types (float3 maps to 3x float32)
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(rotations.scalar_type() == torch::kFloat32, "rotations must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");

    // 3. Extract dimensions and validate shapes
    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(rotations.size(1) == 4, "rotations must have shape (N, 4)");
    TORCH_CHECK(scales.size(1) == 3, "scales must have shape (N, 3)");

    // 4. Allocate Output Tensors directly on the same GPU as the inputs
    auto options = means.options();
    torch::Tensor aabb_min = torch::empty({num_gaussians, 3}, options);
    torch::Tensor aabb_max = torch::empty({num_gaussians, 3}, options);

    // 5. Launch the CUDA Kernel
    // We safely cast the flat float arrays into float3/float4 structs.
    gs::get_aabb(
        num_gaussians,
        reinterpret_cast<const float3*>(means.data_ptr<float>()),
        reinterpret_cast<const float4*>(rotations.data_ptr<float>()),
        reinterpret_cast<const float3*>(scales.data_ptr<float>()),
        iso,
        tol,
        level,
        rotnorm,
        reinterpret_cast<float3*>(aabb_min.data_ptr<float>()),
        reinterpret_cast<float3*>(aabb_max.data_ptr<float>())
    );

    // 6. Return as a Python Tuple
    return std::make_tuple(aabb_min, aabb_max);
}

void bind_primitive_gs(py::module_& m) {
    m.def("get_aabb_wrapper", &get_aabb_wrapper, "Compute AABB for 3D Gaussians",
        py::arg("means"),
        py::arg("rotations"),
        py::arg("scales"),
        py::arg("iso"),
        py::arg("tol"),
        py::arg("level"),
        py::arg("rotnorm") = false // Provides a nice Python default argument
    );
}