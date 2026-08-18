#ifndef MCA_H
#define MCA_H

#include <torch/extension.h>
#include "mc_data.h"
#include <tuple>
#include <optional>

namespace mca {

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> marching_cubes_asymptotic(
    const torch::Tensor &grid_vertices,
    const torch::Tensor &voxels,
    const torch::Tensor &sdf,
    const std::optional<torch::Tensor> &colors,
    float iso
);

std::tuple<torch::Tensor, std::optional<torch::Tensor>> marching_cubes_asymptotic_backward(
    const torch::Tensor &grad_vertices,
    const std::optional<torch::Tensor> &grad_colors,
    const torch::Tensor &grid_vertices,
    const torch::Tensor &unique_edges,
    const torch::Tensor &sdf,
    const std::optional<torch::Tensor> &colors,
    float iso
);

} // namespace mca

#endif // MCA_H
