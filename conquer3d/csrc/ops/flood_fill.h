#ifndef FLOOD_FILL_H
#define FLOOD_FILL_H

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace ops {
    torch::Tensor compute_flood_fill(
        const torch::Tensor& active_voxel_ids,
        int64_t vx,
        int64_t vy,
        int64_t vz,
        int connectivity = 6
    );
}

#endif // FLOOD_FILL_H
