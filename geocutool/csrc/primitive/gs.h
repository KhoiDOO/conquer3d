#ifndef GS_H
#define GS_H

#include "base.h"
#include "../check.h"

#include <cuda_runtime.h>
#include <cstdint>


namespace gs
{
    __host__ void get_aabb(
        const uint32_t num_gaussians,
        const float3* __restrict__ means,
        const float4* __restrict__ rotations,
        const float3* __restrict__ scales,
        const float iso,
        const float tol,
        const uint32_t level,
        const bool rotnorm,
        float3* __restrict__ aabb_min,
        float3* __restrict__ aabb_max
    );
}

#endif // GS_H