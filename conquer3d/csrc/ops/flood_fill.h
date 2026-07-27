#ifndef FLOOD_FILL_H
#define FLOOD_FILL_H

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace ops {
    torch::Tensor compute_flood_fill(
        const torch::Tensor& vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& aabb_mins,
        const torch::Tensor& aabb_maxs,
        const torch::Tensor& bvh_children,
        const torch::Tensor& object_ids,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> grid_res,
        int connectivity = 6
    );
}

#endif // FLOOD_FILL_H
