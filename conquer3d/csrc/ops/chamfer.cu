#include "../data_structure/kdtree.h"

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

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
    thrust::device_vector<float3> cloned_ref_points(reference_points, reference_points + num_reference_points);
    thrust::device_vector<int64_t> reference_indices(num_reference_points);
    thrust::sequence(reference_indices.begin(), reference_indices.end());

    kdtree::build(
        num_reference_points,
        thrust::raw_pointer_cast(cloned_ref_points.data()),
        thrust::raw_pointer_cast(reference_indices.data()));

    uint32_t threads = NTHREADS;
    uint32_t blocks = (num_query_points + threads - 1) / threads;

    one_sided_chamfer_distance_kernel<<<blocks, threads>>>(
        num_query_points,
        query_points,
        num_reference_points,
        thrust::raw_pointer_cast(cloned_ref_points.data()),
        thrust::raw_pointer_cast(reference_indices.data()),
        distances,
        indices);
}