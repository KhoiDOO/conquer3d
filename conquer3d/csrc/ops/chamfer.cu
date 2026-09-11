/**
 * @file chamfer.cu
 * @brief CUDA kernel implementations for GPU KD-Tree accelerated nearest neighbor Chamfer distance.
 */

#include "../data_structure/kdtree.h"

#include <torch/extension.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cfloat>

/**
 * @brief Degenerate-case kernel for a single-point reference set.
 * @details When the reference cloud holds exactly one point there is nothing to search:
 * the nearest neighbour is that point for every query. Specialising the case avoids
 * building a KD-tree whose traversal would cost more than the distance itself. One
 * thread per query point, fully coalesced.
 * @param[in] num_query_points Number of query points $N$.
 * @param[in] query_points Device array of $N$ query coordinates.
 * @param[in] reference_points Device array holding the single reference point.
 * @param[out] distances Device array of $N$ squared distances.
 * @param[out] indices Device array of $N$ neighbour indices, all zero.
 * @note Non-finite distances are clamped to `FLT_MAX` so downstream reductions stay
 * well defined.
 */
__global__ void one_sided_chamfer_single_point_kernel(const uint32_t num_query_points,
                                                      const float3 *__restrict__ query_points,
                                                      const float3 *__restrict__ reference_points,
                                                      float *__restrict__ distances, int64_t *__restrict__ indices)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_query_points)
        return;

    float3 qp = query_points[idx];
    float3 rp = reference_points[0];
    float3 d = qp - rp;
    float dist_sq = maths::dot(d, d);
    if (!isfinite(dist_sq))
        dist_sq = FLT_MAX;

    distances[idx] = dist_sq;
    indices[idx] = 0;
}

/**
 * @brief Nearest-neighbour search of every query point against a KD-tree.
 * @details One thread per query point, each walking the reference KD-tree independently
 * via `kdtree::query_kdtree_loop`. The per-thread priority queue is declared with the
 * compile-time bound `MAX_K` so it lives in registers rather than local memory, and only
 * the nearest entry is retained since Chamfer distance needs $k = 1$.
 * @param[in] num_query_points Number of query points $N$.
 * @param[in] query_points Device array of $N$ query coordinates.
 * @param[in] num_reference_points Number of reference points $M$.
 * @param[in] tree_points Device array of $M$ reference coordinates, KD-tree ordered.
 * @param[in] tree_inds Device array of $M$ permutation indices mapping tree order back to
 *     the caller's original ordering.
 * @param[out] distances Device array of $N$ squared distances to the nearest reference.
 * @param[out] indices Device array of $N$ original-order reference indices.
 * @warning Traversal is data dependent, so threads in a warp diverge whenever their
 * queries descend different branches; throughput therefore falls as the reference cloud
 * becomes less uniform.
 */
__global__ void one_sided_chamfer_distance_kernel(const uint32_t num_query_points,
                                                  const float3 *__restrict__ query_points,
                                                  const uint32_t num_reference_points,
                                                  const float3 *__restrict__ tree_points,
                                                  const int64_t *__restrict__ tree_inds, float *__restrict__ distances,
                                                  int64_t *__restrict__ indices)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_query_points)
        return;

    float3 query_point = query_points[idx];

    float best_dists[MAX_K];
    int64_t best_inds[MAX_K];

#pragma unroll
    for (int i = 0; i < MAX_K; i++)
    {
        best_dists[i] = FLT_MAX;
        best_inds[i] = -1;
    }

    kdtree::query_kdtree_loop(query_point, num_reference_points, tree_points, tree_inds, 1, best_dists, best_inds);

    distances[idx] = best_dists[0];
    indices[idx] = best_inds[0];
}

void one_sided_chamfer_distance(const uint32_t num_query_points, const float3 *__restrict__ query_points,
                                const uint32_t num_reference_points, const float3 *__restrict__ reference_points,
                                float *__restrict__ distances, int64_t *__restrict__ indices)
{
    if (num_query_points == 0)
        return;

    // Every enqueue below must go on PyTorch's current stream: the output buffers and the
    // scratch tensors are allocated by the caching allocator against that stream, and
    // kdtree::build already uses it. Mixing in the legacy default stream races, because
    // PyTorch creates its streams with cudaStreamNonBlocking and so gets no implicit sync.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (num_reference_points == 0)
    {
        // No reference points available: saturate distances and mark indices as -1.
        cudaMemsetAsync(distances, 0x7F, num_query_points * sizeof(float), stream);
        cudaMemsetAsync(indices, 0xFF, num_query_points * sizeof(int64_t), stream);
        return;
    }

    uint32_t threads = NTHREADS;
    uint32_t blocks = (num_query_points + threads - 1) / threads;

    if (num_reference_points == 1)
    {
        one_sided_chamfer_single_point_kernel<<<blocks, threads, 0, stream>>>(num_query_points, query_points,
                                                                              reference_points, distances, indices);
        return;
    }

    // Allocate PyTorch-backed memory for cloned points and permutation indices (zero cudaMalloc overhead)
    auto opt_f = torch::TensorOptions().device(torch::kCUDA, ::c10::cuda::current_device()).dtype(torch::kFloat32);
    auto opt_i = torch::TensorOptions().device(torch::kCUDA, ::c10::cuda::current_device()).dtype(torch::kInt64);

    auto cloned_ref_tensor = torch::empty({(int64_t)num_reference_points, 3}, opt_f);
    cudaMemcpyAsync(cloned_ref_tensor.data_ptr<float>(), reference_points, num_reference_points * sizeof(float3),
                    cudaMemcpyDeviceToDevice, stream);

    auto ref_indices_tensor = torch::arange((int64_t)num_reference_points, opt_i);

    float3 *p_cloned = (float3 *)cloned_ref_tensor.data_ptr<float>();
    int64_t *p_inds = ref_indices_tensor.data_ptr<int64_t>();

    kdtree::build(num_reference_points, p_cloned, p_inds);

    one_sided_chamfer_distance_kernel<<<blocks, threads, 0, stream>>>(
        num_query_points, query_points, num_reference_points, p_cloned, p_inds, distances, indices);
}

/**
 * @brief CUDA kernel computing analytical gradients for one-sided Chamfer distance.
 *
 * Each thread handles one query point. For each query point, it computes the gradient
 * contribution w.r.t. the query point (direct coalesced write) and atomically accumulates the
 * reaction gradient into the corresponding nearest reference point.
 *
 * @param[in] num_query_points Number of query points ($N$).
 * @param[in] query_points Device array of $N$ query coordinates.
 * @param[in] num_reference_points Number of reference points ($M$).
 * @param[in] reference_points Device array of $M$ reference coordinates.
 * @param[in] indices Device array of $N$ nearest reference point indices.
 * @param[in] grad_distances Device array of $N$ incoming adjoint gradients.
 * @param[in] squared Whether metric is squared Euclidean ($L_2^2$) or Euclidean ($L_2$).
 * @param[out] grad_query Device array of $N$ query point gradients (nullptr if not needed).
 * @param[out] grad_reference Device array of $M$ reference point gradients (nullptr if not needed).
 */
__global__ void one_sided_chamfer_distance_backward_kernel(
    const uint32_t num_query_points, const float3 *__restrict__ query_points, const uint32_t num_reference_points,
    const float3 *__restrict__ reference_points, const int64_t *__restrict__ indices,
    const float *__restrict__ grad_distances, const bool squared, float3 *__restrict__ grad_query,
    float3 *__restrict__ grad_reference)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_query_points)
        return;

    int64_t ref_idx = indices[idx];
    if (ref_idx < 0 || ref_idx >= static_cast<int64_t>(num_reference_points))
        return;

    float g = grad_distances[idx];
    float3 q = query_points[idx];
    float3 r = reference_points[ref_idx];

    float dx = q.x - r.x;
    float dy = q.y - r.y;
    float dz = q.z - r.z;

    float scale = 0.0f;
    if (squared)
    {
        scale = 2.0f * g;
    }
    else
    {
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        scale = (dist > 1e-12f) ? (g / dist) : 0.0f;
    }

    float gx = scale * dx;
    float gy = scale * dy;
    float gz = scale * dz;

    if (grad_query != nullptr)
    {
        grad_query[idx] = make_float3(gx, gy, gz);
    }

    if (grad_reference != nullptr)
    {
        atomicAdd(&(grad_reference[ref_idx].x), -gx);
        atomicAdd(&(grad_reference[ref_idx].y), -gy);
        atomicAdd(&(grad_reference[ref_idx].z), -gz);
    }
}

void one_sided_chamfer_distance_backward(const uint32_t num_query_points, const float3 *__restrict__ query_points,
                                         const uint32_t num_reference_points,
                                         const float3 *__restrict__ reference_points,
                                         const int64_t *__restrict__ indices, const float *__restrict__ grad_distances,
                                         const bool squared, float3 *__restrict__ grad_query,
                                         float3 *__restrict__ grad_reference)
{
    if (num_query_points == 0)
        return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (num_reference_points == 0)
    {
        if (grad_query != nullptr)
        {
            cudaMemsetAsync(grad_query, 0, num_query_points * sizeof(float3), stream);
        }
        return;
    }

    if (grad_query == nullptr && grad_reference == nullptr)
        return;

    uint32_t threads = NTHREADS;
    uint32_t blocks = (num_query_points + threads - 1) / threads;

    one_sided_chamfer_distance_backward_kernel<<<blocks, threads, 0, stream>>>(
        num_query_points, query_points, num_reference_points, reference_points, indices, grad_distances, squared,
        grad_query, grad_reference);
}