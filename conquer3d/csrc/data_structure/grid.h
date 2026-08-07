#pragma once
#include <vector>
#include <torch/extension.h>
#include "../maths/maths.h"

namespace grid {
    void get_active_voxel_ids_from_depth(
        const int num_pixels,
        const float* depth_image,
        const float4x4 c2w,
        const float3x3 intrinsics_inv,
        const int image_width,
        const int image_height,
        const float3 grid_min,
        const float3 grid_max,
        const int3 res,
        int64_t* out_voxel_ids,
        unsigned long long* valid_counter,
        bool activate_neighbor,
        float trunc_margin
    );
}
