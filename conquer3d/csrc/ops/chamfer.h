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
 * @param[in] num_query_points Number of query points ($N$).
 * @param[in] query_points Pointer to (N, 3) float32 query coordinates in device memory.
 * @param[in] num_reference_points Number of reference points ($M$).
 * @param[in] reference_points Pointer to (M, 3) float32 reference coordinates in device memory.
 * @param[out] distances Output device buffer of size $N$ for minimum squared Euclidean distances.
 * @param[out] indices Output device buffer of size $N$ for nearest reference point indices.
 */
void one_sided_chamfer_distance(
    const uint32_t num_query_points,
    const float3* __restrict__ query_points,
    const uint32_t num_reference_points,
    const float3* __restrict__ reference_points,
    float* __restrict__ distances,
    int64_t* __restrict__ indices
);

/**
 * @brief Computes the analytical backward gradients for one-sided Chamfer distance.
 *
 * Computes the gradients of an upstream scalar loss with respect to query points and/or
 * reference points using the nearest-neighbor assignments obtained in the forward pass.
 * Query point gradients are written directly without atomic contention, while reference point
 * gradients are accumulated atomically across threads.
 *
 * @param[in] num_query_points Number of query points ($N$).
 * @param[in] query_points Pointer to (N, 3) float32 query coordinates in device memory.
 * @param[in] num_reference_points Number of reference points ($M$).
 * @param[in] reference_points Pointer to (M, 3) float32 reference coordinates in device memory.
 * @param[in] indices Pointer to (N,) int64 nearest-neighbor indices from forward pass.
 * @param[in] grad_distances Pointer to (N,) float32 incoming adjoint gradients w.r.t. distances.
 * @param[in] squared Whether the forward distance was squared Euclidean ($L_2^2$) or Euclidean ($L_2$).
 * @param[out] grad_query Output device buffer of size $N$ for query gradients (or nullptr if not requested).
 * @param[out] grad_reference Output device buffer of size $M$ for reference gradients (or nullptr if not requested).
 */
void one_sided_chamfer_distance_backward(
    const uint32_t num_query_points,
    const float3* __restrict__ query_points,
    const uint32_t num_reference_points,
    const float3* __restrict__ reference_points,
    const int64_t* __restrict__ indices,
    const float* __restrict__ grad_distances,
    const bool squared,
    float3* __restrict__ grad_query,
    float3* __restrict__ grad_reference
);

#endif // CHAMFER_H