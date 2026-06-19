#include "gs.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

namespace gs
{
    __global__ void compute_gs_covi_kernel(
        const uint32_t num_gaussians,
        const float4 *__restrict__ rotations,
        const float3 *__restrict__ scales,
        const bool rotnorm,
        const float tol,
        const uint32_t level,
        float *__restrict__ covis)
    {
        uint32_t g_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (g_idx >= num_gaussians)
            return;

        float two_level = (float)(1U << level);
        float voxelSize = 2.0f / two_level;
        float min_scale = tol * voxelSize;
        float3 modified_scale = scales[g_idx];

        modified_scale.x = fmaxf(modified_scale.x, min_scale);
        modified_scale.y = fmaxf(modified_scale.y, min_scale);
        modified_scale.z = fmaxf(modified_scale.z, min_scale);

        float3x3 S_inv; // $S^{-1}$
        gs::compute_inverse_scale(modified_scale, S_inv);

        float3x3 R_T; // $R^T$
        gs::compute_rotation(rotations[g_idx], R_T, rotnorm, true);

        gs::compute_cov_inverse(S_inv, R_T, covis + (g_idx * 6)); // $(S^{-1} R^T)^T (S^{-1} R^T)$
    }

    void compute_gs_covi(
        const uint32_t num_gaussians,
        const float4 *__restrict__ rotations,
        const float3 *__restrict__ scales,
        const bool rotnorm,
        const float tol,
        const uint32_t level,
        float *__restrict__ covis)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_gaussians + threads - 1) / threads;

        compute_gs_covi_kernel<<<blocks, threads>>>(
            num_gaussians,
            rotations,
            scales,
            rotnorm,
            tol,
            level,
            covis);
    }

    __global__ void solve_gs_neighbor_mahalanobis_radius_kernel(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float3 *__restrict__ tree_points,
        const int64_t *__restrict__ tree_inds,
        const int k,
        float *__restrict__ isos)
    {
        uint32_t g_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (g_idx >= num_gaussians)
            return;

        float3 mean = means[g_idx];
        const float *covi = covis + (g_idx * 6);

        float best_dists[MAX_K];
        int64_t best_inds[MAX_K];

        #pragma unroll
        for (int i = 0; i < MAX_K; i++)
        {
            best_dists[i] = FLT_MAX;
            best_inds[i] = -1;
        }

        kdtree::query_kdtree_loop(mean, num_gaussians, tree_points, tree_inds, k, best_dists, best_inds);

        float sum_iso = 0.0f;
        int valid_count = 0;

        for (int i = 0; i < k; i++)
        {
            int64_t neighbor_idx = best_inds[i];

            if (neighbor_idx == -1)
                continue;
            if (neighbor_idx == g_idx)
                continue;

            float3 neighbor_mean = means[neighbor_idx];

            float iso;
            gs::compute_mahalanobis_distance(neighbor_mean, mean, covi, iso);

            sum_iso += iso;
            valid_count++;
        }

        if (valid_count > 0)
        {
            isos[g_idx] = sum_iso / (float)valid_count;
        }
        else
        {
            isos[g_idx] = 1.0f;
        }
    }

    void solve_gs_neighbor_mahalanobis_radius(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const int k,
        float *__restrict__ isos)
    {
        thrust::device_vector<float3> cloned_means(means, means + num_gaussians);
        thrust::device_vector<int64_t> oinds(num_gaussians);
        thrust::sequence(oinds.begin(), oinds.end());

        kdtree::build(
            num_gaussians,
            thrust::raw_pointer_cast(cloned_means.data()),
            thrust::raw_pointer_cast(oinds.data()));

        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_gaussians + threads - 1) / threads;

        solve_gs_neighbor_mahalanobis_radius_kernel<<<blocks, threads>>>(
            num_gaussians,
            means,
            covis,
            thrust::raw_pointer_cast(cloned_means.data()),
            thrust::raw_pointer_cast(oinds.data()),
            k,
            isos);
    }
}