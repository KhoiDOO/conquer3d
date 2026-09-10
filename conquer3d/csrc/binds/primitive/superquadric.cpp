#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../primitive/superquadric.h"
#include "../../check.h"

#include <optional>
#include <tuple>

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for superquadric mesh tessellation.
 * @details Sits between pybind11 and the host dispatchers: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- validates the shared primitive
 * count and the resolution, allocates the output and scratch buffers, then chains the three
 * kernel launchers. Validating here keeps the launch path free of checks and gives Python
 * callers a clear error instead of a device fault.
 * @return Tuple of (vertices, triangles, labels), where labels is undefined unless requested.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_superquadric_mesh_wrapper(
    torch::Tensor scales,
    torch::Tensor exponents,
    torch::Tensor rotations,
    torch::Tensor translations,
    int64_t resolution,
    bool return_labels)
{
    CHECK_INPUT(scales);
    CHECK_INPUT(exponents);
    CHECK_INPUT(rotations);
    CHECK_INPUT(translations);

    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(exponents.scalar_type() == torch::kFloat32, "exponents must be float32");
    TORCH_CHECK(rotations.scalar_type() == torch::kFloat32, "rotations must be float32");
    TORCH_CHECK(translations.scalar_type() == torch::kFloat32, "translations must be float32");

    TORCH_CHECK(resolution >= 3, "resolution must be at least 3, got ", resolution);

    const int64_t num_quadrics = scales.size(0);
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3,
                "scales must have shape (K, 3)");
    TORCH_CHECK(exponents.dim() == 2 && exponents.size(0) == num_quadrics && exponents.size(1) == 2,
                "exponents must have shape (K, 2)");
    TORCH_CHECK(rotations.dim() == 3 && rotations.size(0) == num_quadrics && rotations.size(1) == 3 && rotations.size(2) == 3,
                "rotations must have shape (K, 3, 3)");
    TORCH_CHECK(translations.dim() == 2 && translations.size(0) == num_quadrics && translations.size(1) == 3,
                "translations must have shape (K, 3)");
    TORCH_CHECK(num_quadrics > 0, "at least one primitive is required to tessellate");

    const int64_t n = resolution;
    const int64_t verts_per_primitive = n * (n - 2) + 2;
    const int64_t tris_per_primitive = 2 * n * (n - 2);

    auto options_f32 = scales.options();
    auto options_i32 = scales.options().dtype(torch::kInt32);
    auto options_f64 = scales.options().dtype(torch::kFloat64);

    torch::Tensor vertices = torch::empty({num_quadrics * verts_per_primitive, 3}, options_f32);
    torch::Tensor triangles = torch::empty({num_quadrics * tris_per_primitive, 3}, options_i32);
    torch::Tensor labels;
    if (return_labels)
    {
        labels = torch::empty({num_quadrics * verts_per_primitive}, options_i32);
    }

    // The sampler's traversal depth is bounded by its sample count, so the stack lives in
    // global scratch rather than per-thread storage where the worst case would spill.
    const int64_t frame_doubles = sizeof(sq::SuperellipseFrame) / sizeof(double);
    torch::Tensor stack_scratch = torch::empty(
        {2 * num_quadrics * (n + 2) * frame_doubles}, options_f64);
    torch::Tensor azimuths = torch::empty({num_quadrics * (n + 1)}, options_f64);
    torch::Tensor polars = torch::empty({num_quadrics * n}, options_f64);

    sq::compute_superellipse_angles(
        (uint32_t)num_quadrics,
        (uint32_t)n,
        reinterpret_cast<const float3 *>(scales.data_ptr<float>()),
        reinterpret_cast<const float2 *>(exponents.data_ptr<float>()),
        reinterpret_cast<sq::SuperellipseFrame *>(stack_scratch.data_ptr<double>()),
        azimuths.data_ptr<double>(),
        polars.data_ptr<double>());

    sq::compute_superquadric_vertices(
        (uint32_t)num_quadrics,
        (uint32_t)n,
        reinterpret_cast<const float3 *>(scales.data_ptr<float>()),
        reinterpret_cast<const float2 *>(exponents.data_ptr<float>()),
        rotations.data_ptr<float>(),
        reinterpret_cast<const float3 *>(translations.data_ptr<float>()),
        azimuths.data_ptr<double>(),
        polars.data_ptr<double>(),
        return_labels,
        reinterpret_cast<float3 *>(vertices.data_ptr<float>()),
        return_labels ? labels.data_ptr<int32_t>() : nullptr);

    sq::compute_superquadric_faces(
        (uint32_t)num_quadrics,
        (uint32_t)n,
        reinterpret_cast<int3 *>(triangles.data_ptr<int32_t>()));

    return std::make_tuple(vertices, triangles, labels);
}

/**
 * @brief Registers superquadric primitive operators on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol
 * defined here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_primitive_superquadric(py::module_ &m)
{
    m.def("compute_superquadric_mesh_func", &compute_superquadric_mesh_wrapper,
          py::arg("scales"), py::arg("exponents"), py::arg("rotations"),
          py::arg("translations"), py::arg("resolution") = 30,
          py::arg("return_labels") = false,
          R"pbdoc(
          Tessellates a set of superquadrics into a single triangle mesh (CUDA).

          Each primitive is meshed analytically from Barr's parametric form on an
          `(resolution, resolution)` grid of angles spaced by approximately equal arc length,
          placed into world space, then packed into one vertex and triangle array.

          Args:
              scales (torch.Tensor): (K, 3) float32 semi-axes on CUDA, strictly positive.
              exponents (torch.Tensor): (K, 2) float32 shape exponents on CUDA.
              rotations (torch.Tensor): (K, 3, 3) float32 local-to-world rotations on CUDA.
              translations (torch.Tensor): (K, 3) float32 primitive centres on CUDA.
              resolution (int, optional): Angular samples along each axis. Each primitive
                  contributes `resolution * (resolution - 2) + 2` vertices and
                  `2 * resolution * (resolution - 2)` triangles. Defaults to 30.
              return_labels (bool, optional): If True, also returns the originating primitive
                  index of every vertex. Defaults to False.

          Returns:
              Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                  - vertices (torch.Tensor): (V, 3) float32 world coordinates.
                  - triangles (torch.Tensor): (F, 3) int32 vertex indices.
                  - labels (torch.Tensor): (V,) int32 primitive index per vertex, or None
                    when `return_labels` is False.

          Note:
              The result is a concatenation, not a union: one closed surface per primitive,
              self-intersecting wherever two overlap. Each primitive alone is watertight.

          Example:
              >>> from conquer3d._C import compute_superquadric_mesh_func
              >>> verts, tris, _ = compute_superquadric_mesh_func(
              ...     scales, exponents, rotations, translations, 40)
          )pbdoc");
}
