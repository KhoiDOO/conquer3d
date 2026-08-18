#ifndef MT_H
#define MT_H

#include <torch/extension.h>
#include "mt_data.h"
#include "../maths/maths.h"
#include "../check.h"
#include "../constants.h"
#include "../primitive/edge.h"

#include <optional>

namespace mt {
    std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>> marching_tetrahedra(
        const uint32_t num_tets,
        const float3* __restrict__ grid_vertices,
        const uint32_t* __restrict__ tets,
        const float* __restrict__ vert_values,
        const float3* __restrict__ grid_normals,
        const float3* __restrict__ grid_colors,
        const float iso,
        torch::TensorOptions vert_options,
        torch::TensorOptions tri_options,
        bool return_unique_edges = false
    );

    void backward(
        const uint32_t n_verts,
        const Edge *unique_edges,
        const float3 *grid_vertices,
        const float3 *grid_colors,
        const float *values,
        const float3 *adj_verts,
        const float3 *adj_colors,
        float *adj_values,
        float3 *adj_grid_colors,
        const float iso
    );
}

#endif // MT_H
