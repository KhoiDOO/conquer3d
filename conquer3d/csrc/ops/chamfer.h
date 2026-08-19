/**
 * @file chamfer.h
 * @brief GPU KD-Tree accelerated one-sided Chamfer distance operator.
 */

#ifndef CHAMFER_H
#define CHAMFER_H

#include "../constants.h"
#include "../maths/maths.h"

#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief Computes the nearest-neighbor distance from each query point to a reference point set.
 * 
 * Builds an on-device KD-Tree over the reference point set and queries the nearest
 * neighbor for all query points in parallel.
 * 
 * @param[in]  num_query_points      Number of query points ($N$).
 * @param[in]  query_points          Pointer to (N, 3) float32 query coordinates in device memory.
 * @param[in]  num_reference_points  Number of reference points ($M$).
 * @param[in]  reference_points      Pointer to (M, 3) float32 reference coordinates in device memory.
 * @param[out] distances             Output device buffer of size $N$ for minimum squared Euclidean distances.
 * @param[out] indices               Output device buffer of size $N$ for nearest reference point indices.
 */
void one_sided_chamfer_distance(
    const uint32_t num_query_points,
    const float3* __restrict__ query_points,
    const uint32_t num_reference_points,
    const float3* __restrict__ reference_points,
    float* __restrict__ distances,
    int64_t* __restrict__ indices
);

#endif // CHAMFER_H