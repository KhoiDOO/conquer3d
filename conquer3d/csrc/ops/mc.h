#ifndef MC_H
#define MC_H

#include <torch/extension.h>
#include "mc_data.h"
#include "../check.h"
#include "../constants.h"
#include "../primitive/edge.h"

#include <optional>

namespace mc{
    std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> marching_cubes(
        const uint32_t num_voxels,
        const float3* __restrict__ grid_vertices,
        const uint32_t* __restrict__ voxels,
        const float* __restrict__ voxel_values,
        const float3* __restrict__ grid_normals,
        const float iso,
        torch::TensorOptions vert_options,
        torch::TensorOptions tri_options
    );
}

#endif // MC_H