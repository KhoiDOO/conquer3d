#ifndef VOLINT_H
#define VOLINT_H

#include "../maths/maths.h"
#include <cuda_runtime.h>

void single_view_volume_integral(
    const int num_vertices,
    const float3* grid_vertices,
    float* sdf,
    float* weight,
    float3* color,
    const float* depth_image,
    const float3* color_image,
    const int image_width,
    const int image_height,
    const float4x4 extrinsics,
    const float3x3 intrinsics,
    const float trunc_margin,
    const int mode
);

#endif // VOLINT_H