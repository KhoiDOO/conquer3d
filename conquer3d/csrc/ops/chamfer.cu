/**
 * @file chamfer.cu
 * @brief CUDA kernel implementations for GPU KD-Tree accelerated nearest neighbor Chamfer distance.
 */

#include "../data_structure/kdtree.h"

#include <torch/extension.h>
#include <cstdint>
#include <cfloat>

__global__ void one_sided_chamfer_single_point_kernel(
    const uint32_t num_query_points,
    const float3 *__restrict__ query_points,
    const float3 *__restrict__ reference_points,
    float *__restrict__ distances,
    int64_t *__restrict__ indices)
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

__global__ void one_sided_chamfer_distance_kernel(
    const uint32_t num_query_points,
    const float3 *__restrict__ query_points,
    const uint32_t num_reference_points,
    const float3 *__restrict__ tree_points,
    const int64_t *__restrict__ tree_inds,
    float *__restrict__ distances,
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

    kdtree::query_kdtree_loop(
        query_point,
        num_reference_points,
        tree_points,
        tree_inds,
        1,
        best_dists,
        best_inds);
    
    distances[idx] = best_dists[0];
    indices[idx] = best_inds[0];
}

void one_sided_chamfer_distance(
    const uint32_t num_query_points,
    const float3 *__restrict__ query_points,
    const uint32_t num_reference_points,
    const float3 *__restrict__ reference_points,
    float *__restrict__ distances,
    int64_t *__restrict__ indices)
{
    if (num_query_points == 0)
        return;

    if (num_reference_points == 0)
    {
        // No reference points available: initialize distances to infinity and indices to -1
        cudaMemsetAsync(distances, 0x7F, num_query_points * sizeof(float));
        cudaMemsetAsync(indices, 0xFF, num_query_points * sizeof(int64_t));
        return;
    }

    uint32_t threads = NTHREADS;
    uint32_t blocks = (num_query_points + threads - 1) / threads;

    if (num_reference_points == 1)
    {
        one_sided_chamfer_single_point_kernel<<<blocks, threads>>>(
            num_query_points,
            query_points,
            reference_points,
            distances,
            indices);
        return;
    }

    // Allocate PyTorch-backed memory for cloned points and permutation indices (zero cudaMalloc overhead)
    auto opt_f = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto opt_i = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);

    auto cloned_ref_tensor = torch::empty({(int64_t)num_reference_points, 3}, opt_f);
    cudaMemcpyAsync(
        cloned_ref_tensor.data_ptr<float>(),
        reference_points,
        num_reference_points * sizeof(float3),
        cudaMemcpyDeviceToDevice
    );

    auto ref_indices_tensor = torch::arange((int64_t)num_reference_points, opt_i);

    float3* p_cloned = (float3*)cloned_ref_tensor.data_ptr<float>();
    int64_t* p_inds = ref_indices_tensor.data_ptr<int64_t>();

    kdtree::build(
        num_reference_points,
        p_cloned,
        p_inds);

    one_sided_chamfer_distance_kernel<<<blocks, threads>>>(
        num_query_points,
        query_points,
        num_reference_points,
        p_cloned,
        p_inds,
        distances,
        indices);
}