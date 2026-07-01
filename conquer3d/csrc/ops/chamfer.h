#ifndef CHAMFER_H
#define CHAMFER_H

#include "../constants.h"
#include "../maths/maths.h"

#include <cuda_runtime.h>

void one_sided_chamfer_distance(
    const uint32_t num_query_points,
    const float3* __restrict__ query_points,
    const uint32_t num_reference_points,
    const float3* __restrict__ reference_points,
    float* __restrict__ distances,
    int64_t* __restrict__ indices
);

#endif // CHAMFER_H