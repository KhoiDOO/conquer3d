#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "../../ops/chamfer.h"
#include "../../check.h"

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor> one_sided_chamfer_distance_wrapper(
    torch::Tensor query_points,
    torch::Tensor reference_points
) {
    CHECK_INPUT(query_points);
    CHECK_INPUT(reference_points);

    uint32_t num_query_points = query_points.size(0);
    uint32_t num_reference_points = reference_points.size(0);

    const float3* __restrict__ p_query_points = (float3*)query_points.data_ptr<float>();
    const float3* __restrict__ p_reference_points = (float3*)reference_points.data_ptr<float>();

    torch::Tensor distances = torch::empty({num_query_points}, query_points.options());
    torch::Tensor indices = torch::empty({num_query_points}, query_points.options().dtype(torch::kInt64));

    float* __restrict__ p_distances = distances.data_ptr<float>();
    int64_t* __restrict__ p_indices = indices.data_ptr<int64_t>();

    one_sided_chamfer_distance(
        num_query_points,
        p_query_points,
        num_reference_points,
        p_reference_points,
        p_distances,
        p_indices
    );

    return std::make_tuple(distances, indices);
}

void bind_ops_chamfer(py::module_& m) {
    m.def("one_sided_chamfer_distance", &one_sided_chamfer_distance_wrapper, "One Sided Chamfer Distance",
          py::arg("query_points"), py::arg("reference_points"));
}
