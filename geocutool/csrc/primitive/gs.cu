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
        uint32_t threads = 256;
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
        if (g_idx >= num_gaussians) return;

        float3 mean = means[g_idx];
        const float *covi = covis + (g_idx * 6);

        float best_dists[MAX_K];
        int64_t best_inds[MAX_K];

        #pragma unroll
        for (int i = 0; i < MAX_K; i++) {
            best_dists[i] = FLT_MAX;
            best_inds[i] = -1;
        }

        kdtree::query_kdtree_loop(mean, num_gaussians, tree_points, tree_inds, k, best_dists, best_inds);

        float sum_iso = 0.0f;
        int valid_count = 0;

        for (int i = 0; i < k; i++) {
            int64_t neighbor_idx = best_inds[i];

            if (neighbor_idx == -1) continue;
            if (neighbor_idx == g_idx) continue;

            float3 neighbor_mean = means[neighbor_idx];

            float iso;
            gs::compute_mahalanobis_distance(neighbor_mean, mean, covi, iso);
            
            sum_iso += iso;
            valid_count++;
        }

        if (valid_count > 0) {
            isos[g_idx] = sum_iso / (float)valid_count;
        } else {
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
            thrust::raw_pointer_cast(oinds.data())
        );

        uint32_t threads = 256;
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

namespace gs_aabb
{
    __device__ __forceinline__ void compute_gs_single_aabb(
        const float3 &mean,
        const float3 &scale,
        const float *__restrict__ covi,
        const float &iso,
        const float &tol,
        const uint32_t level,
        float3 &out_min,
        float3 &out_max,
        float3 *contact_points)
    {
        float two_level = (float)(1U << level);
        float voxelSize = 2.0f / two_level;
        float min_scale = tol * voxelSize;
        float3 modified_scale = scale;

        modified_scale.x = fmaxf(scale.x, min_scale);
        modified_scale.y = fmaxf(scale.y, min_scale);
        modified_scale.z = fmaxf(scale.z, min_scale);

        double detS = ((double)modified_scale.x) * ((double)modified_scale.y) * ((double)modified_scale.z);

        double c0 = covi[0];
        double c1 = covi[1];
        double c2 = covi[2];
        double c3 = covi[3];
        double c4 = covi[4];
        double c5 = covi[5];

        double h0 = c3 * c5 - c4 * c4;
        double h1 = c2 * c4 - c1 * c5;
        double h2 = c1 * c4 - c2 * c3;
        double h3 = c0 * c5 - c2 * c2;
        double h4 = c1 * c2 - c0 * c4;
        double h5 = c0 * c3 - c1 * c1;

        double w[3];
        w[0] = detS * sqrt(iso / h0);
        w[1] = detS * sqrt(iso / h3);
        w[2] = detS * sqrt(iso / h5);

        double3 Q[3];
        Q[0] = make_double3(h0, h1, h2);
        Q[1] = make_double3(h1, h3, h4);
        Q[2] = make_double3(h2, h4, h5);

        float3 P[6];
        #pragma unroll
        for (int i = 0; i < 3; i++)
        {
            P[2 * i] = make_float3((float)(w[i] * Q[i].x), (float)(w[i] * Q[i].y), (float)(w[i] * Q[i].z));
            P[2 * i + 1] = -1.0f * P[2 * i];
        }

        contact_points[0] = P[0];
        contact_points[1] = P[2];
        contact_points[2] = P[4];

        float3 Pmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 Pmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        #pragma unroll
        for (int i = 0; i < 6; i++)
        {
            Pmin.x = fminf(Pmin.x, P[i].x);
            Pmin.y = fminf(Pmin.y, P[i].y);
            Pmin.z = fminf(Pmin.z, P[i].z);
            Pmax.x = fmaxf(Pmax.x, P[i].x);
            Pmax.y = fmaxf(Pmax.y, P[i].y);
            Pmax.z = fmaxf(Pmax.z, P[i].z);
        }

        out_min = mean + Pmin;
        out_max = mean + Pmax;
    }

    template <bool multiple_isos>
    __global__ void compute_gs_aabb_kernel(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float3 *__restrict__ scales,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        const float tol,
        const uint32_t level,
        float3 *__restrict__ aabb_min,
        float3 *__restrict__ aabb_max,
        float3 *__restrict__ contact_points)
    {
        uint32_t g_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (g_idx >= num_gaussians)
            return;

        float __iso = multiple_isos ? isos[g_idx] : iso;

        compute_gs_single_aabb(
            means[g_idx],
            scales[g_idx],
            covis + (g_idx * 6),
            __iso,
            tol,
            level,
            aabb_min[g_idx],
            aabb_max[g_idx],
            contact_points + (g_idx * 3));
    }

    void compute_gs_aabb(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float3 *__restrict__ scales,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        const float tol,
        const uint32_t level,
        float3 *__restrict__ aabb_min,
        float3 *__restrict__ aabb_max,
        float3 *__restrict__ contact_points
    ) {

        uint32_t threads = 256;
        uint32_t blocks = (num_gaussians + threads - 1) / threads;

        if (isos != nullptr)
        {
            compute_gs_aabb_kernel<true><<<blocks, threads>>>(
                num_gaussians,
                means,
                scales,
                covis,
                isos,
                iso,
                tol,
                level,
                aabb_min,
                aabb_max,
                contact_points);
        }
        else
        {
            compute_gs_aabb_kernel<false><<<blocks, threads>>>(
                num_gaussians,
                means,
                scales,
                covis,
                isos,
                iso,
                tol,
                level,
                aabb_min,
                aabb_max,
                contact_points);
        }
    }
    
    __device__ __forceinline__ void compute_gs_single_aabb_w_covi(
        const float3 &mean,
        const float4 &rot,
        const float3 &scale,
        const float &iso,
        const float &tol,
        const uint32_t level,
        const bool rotnorm,
        float3 &out_min,
        float3 &out_max,
        float3 *contact_points,
        float *covi)
    {
        float two_level = (float)(0x1 << level);
        float voxelSize = 2.0f / two_level;
        float min_scale = tol * voxelSize;
        float3 modified_scale = scale;

        modified_scale.x = fmaxf(scale.x, min_scale);
        modified_scale.y = fmaxf(scale.y, min_scale);
        modified_scale.z = fmaxf(scale.z, min_scale);

        double detS = ((double)modified_scale.x) * ((double)modified_scale.y) * ((double)modified_scale.z);

        float3x3 S_inv; // $S^{-1}$
        gs::compute_inverse_scale(modified_scale, S_inv);

        float3x3 R_T; // $R^T$
        gs::compute_rotation(rot, R_T, rotnorm, true);

        gs::compute_cov_inverse(S_inv, R_T, covi); // $(S^{-1} R^T)^T (S^{-1} R^T)$

        double c0 = covi[0];
        double c1 = covi[1];
        double c2 = covi[2];
        double c3 = covi[3];
        double c4 = covi[4];
        double c5 = covi[5];

        double h0 = c3 * c5 - c4 * c4;
        double h1 = c2 * c4 - c1 * c5;
        double h2 = c1 * c4 - c2 * c3;
        double h3 = c0 * c5 - c2 * c2;
        double h4 = c1 * c2 - c0 * c4;
        double h5 = c0 * c3 - c1 * c1;

        double w[3];
        w[0] = detS * sqrt(iso / h0);
        w[1] = detS * sqrt(iso / h3);
        w[2] = detS * sqrt(iso / h5);

        double3 Q[3];
        Q[0] = make_double3(h0, h1, h2);
        Q[1] = make_double3(h1, h3, h4);
        Q[2] = make_double3(h2, h4, h5);

        float3 P[6];
        for (int i = 0; i < 3; i++)
        {
            P[2 * i] = make_float3((float)(w[i] * Q[i].x), (float)(w[i] * Q[i].y), (float)(w[i] * Q[i].z));
            P[2 * i + 1] = -1.0f * P[2 * i];
        }

        contact_points[0] = P[0];
        contact_points[1] = P[2];
        contact_points[2] = P[4];

        float3 Pmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 Pmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int i = 0; i < 6; i++)
        {
            Pmin.x = fminf(Pmin.x, P[i].x);
            Pmin.y = fminf(Pmin.y, P[i].y);
            Pmin.z = fminf(Pmin.z, P[i].z);
            Pmax.x = fmaxf(Pmax.x, P[i].x);
            Pmax.y = fmaxf(Pmax.y, P[i].y);
            Pmax.z = fmaxf(Pmax.z, P[i].z);
        }

        out_min = mean + Pmin;
        out_max = mean + Pmax;
    }

    template <bool multiple_isos>
    __global__ void compute_gs_aabb_w_covi_kernel(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float4 *__restrict__ rotations,
        const float3 *__restrict__ scales,
        const float *__restrict__ isos,
        const float iso,
        const float tol,
        const uint32_t level,
        const bool rotnorm,
        float3 *__restrict__ aabb_min,
        float3 *__restrict__ aabb_max,
        float3 *__restrict__ contact_points,
        float *__restrict__ covis)
    {
        uint32_t g_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (g_idx >= num_gaussians)
            return;

        float __iso = multiple_isos ? isos[g_idx] : iso;

        compute_gs_single_aabb_w_covi(
            means[g_idx],
            rotations[g_idx],
            scales[g_idx],
            __iso,
            tol,
            level,
            rotnorm,
            aabb_min[g_idx],
            aabb_max[g_idx],
            contact_points + (g_idx * 3),
            covis + (g_idx * 6));
    }

    void compute_gs_aabb_w_covi(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float4 *__restrict__ rotations,
        const float3 *__restrict__ scales,
        const float *__restrict__ isos,
        const float iso,
        const float tol,
        const uint32_t level,
        const bool rotnorm,
        float3 *__restrict__ aabb_min,
        float3 *__restrict__ aabb_max,
        float3 *__restrict__ contact_points,
        float *__restrict__ covis)
    {
        uint32_t threads = 256;
        uint32_t blocks = (num_gaussians + threads - 1) / threads;

        if (isos != nullptr)
        {
            compute_gs_aabb_w_covi_kernel<true><<<blocks, threads>>>(
                num_gaussians,
                means,
                rotations,
                scales,
                isos,
                iso,
                tol,
                level,
                rotnorm,
                aabb_min,
                aabb_max,
                contact_points,
                covis);
        }
        else
        {
            compute_gs_aabb_w_covi_kernel<false><<<blocks, threads>>>(
                num_gaussians,
                means,
                rotations,
                scales,
                isos,
                iso,
                tol,
                level,
                rotnorm,
                aabb_min,
                aabb_max,
                contact_points,
                covis);
        }
    }

    __device__ __forceinline__ bool test_gs_aabb_inside_voxel(
        const float3 &gs_ab_min,
        const float3 &gs_ab_max,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max)
    {
        return (
            vx_ab_min.x <= gs_ab_min.x &&
            vx_ab_min.y <= gs_ab_min.y &&
            vx_ab_min.z <= gs_ab_min.z &&
            (vx_ab_max.x) >= gs_ab_max.x &&
            (vx_ab_max.y) >= gs_ab_max.y &&
            (vx_ab_max.z) >= gs_ab_max.z);
    }

    __device__ __forceinline__ bool test_gs_intersect_voxel_edge(
        const float3 &mean,
        const float *covi,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max,
        const float iso)
    {

        float3 p = vx_ab_min - mean;             // move to local space of Gaussian
        float vsize = vx_ab_max.x - vx_ab_min.x; // assume cubic voxel

        // if edge crossings, true true
        for (int i = 0; i < 4; i++)
        {
            float dummy_t0, dummy_t1;

            if (gs::test_gs_segment(
                    covi[0], covi[1], covi[2], covi[3], covi[4], covi[5], iso,
                    make_float3(p.x, p.y + (i / 2 ? vsize : 0.0), p.z + (i % 2 ? vsize : 0.0)),
                    make_float3(p.x + vsize, p.y + (i / 2 ? vsize : 0.0), p.z + (i % 2 ? vsize : 0.0)),
                    false, dummy_t0, dummy_t1))
            {
                return true;
            }

            if (gs::test_gs_segment(
                    covi[0], covi[1], covi[2], covi[3], covi[4], covi[5], iso,
                    make_float3(p.x + (i / 2 ? vsize : 0.0), p.y, p.z + (i % 2 ? vsize : 0.0)),
                    make_float3(p.x + (i / 2 ? vsize : 0.0), p.y + vsize, p.z + (i % 2 ? vsize : 0.0)),
                    false, dummy_t0, dummy_t1))
            {
                return true;
            }

            if (gs::test_gs_segment(
                    covi[0], covi[1], covi[2], covi[3], covi[4], covi[5], iso,
                    make_float3(p.x + (i / 2 ? vsize : 0.0), p.y + (i % 2 ? vsize : 0.0), p.z),
                    make_float3(p.x + (i / 2 ? vsize : 0.0), p.y + (i % 2 ? vsize : 0.0), p.z + vsize),
                    false, dummy_t0, dummy_t1))
            {
                return true;
            }
        }

        return false;
    }

    __device__ __forceinline__ bool test_gs_vx_face(
        const float *p,
        float *q,
        const float vsize,
        const int i,
        const int j,
        const int k,
        const float vs)
    {
        float t = (q[i] - (p[i] + vs)) / (2.0 * q[i]);

        if (0.0f <= t && t <= 1.0f)
        {
            float h[3];
            for (int l = 0; l < 3; l++)
                h[l] = (1.0 - 2.0 * t) * q[l];

            return p[j] <= h[j] && h[j] <= (p[j] + vsize) && p[k] <= h[k] && h[k] <= (p[k] + vsize);
        }

        return false;
    }

    __device__ __forceinline__ bool test_gs_intersect_voxel_face(
        const float3 &mean,
        float3 cp0,
        float3 cp1,
        float3 cp2,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max)
    {
        float3 p = vx_ab_min - mean;
        float vsize = vx_ab_max.x - vx_ab_min.x; // assume cubic voxel
        bool b[6];

        // need to index float3 components, so cast to float[]
        b[0] = test_gs_vx_face((float *)(&p), (float *)(&cp0), vsize, 0, 1, 2, 0.0);
        b[1] = test_gs_vx_face((float *)(&p), (float *)(&cp0), vsize, 0, 1, 2, vsize);

        b[2] = test_gs_vx_face((float *)(&p), (float *)(&cp1), vsize, 1, 0, 2, 0.0);
        b[3] = test_gs_vx_face((float *)(&p), (float *)(&cp1), vsize, 1, 0, 2, vsize);

        b[4] = test_gs_vx_face((float *)(&p), (float *)(&cp2), vsize, 2, 0, 1, 0.0);
        b[5] = test_gs_vx_face((float *)(&p), (float *)(&cp2), vsize, 2, 0, 1, vsize);

        return (b[0] || b[1] || b[2] || b[3] || b[4] || b[5]);
    }

    __device__ __forceinline__ bool test_gs_intersect_voxel(
        const uint64_t gaus_idx,
        const float3 &mean,
        const float *covi,
        const float3 &gs_ab_min,
        const float3 &gs_ab_max,
        const float3 &cp0,
        const float3 &cp1,
        const float3 &cp2,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max,
        const float iso)
    {

        if (aabb::test_aabb_inside(gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max))
            return true;

        if (!aabb::test_aabb_overlap(gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max))
            return false;

        if (test_gs_intersect_voxel_face(mean, cp0, cp1, cp2, vx_ab_min, vx_ab_max))
            return true;

        return test_gs_intersect_voxel_edge(mean, covi, vx_ab_min, vx_ab_max, iso);
    }

    __device__ __forceinline__ void compute_overlap_metrics(
        const float3 &gs_ab_min,
        const float3 &gs_ab_max,
        const float3 &vx_ab_min,
        const float3 &vx_ab_max,
        const float3 &mean,
        const float *covi,
        const float opacity,
        const bool return_centroids,
        float3 &out_centroid,
        float &out_density,
        float &out_aspect_ratio,
        float &out_penetration)
    {
        // 1. Calculate Overlap Box boundaries
        float3 overlap_min;
        float3 overlap_max;
        aabb::compute_aabb_overlap(gs_ab_min, gs_ab_max, vx_ab_min, vx_ab_max, overlap_min, overlap_max);

        if (return_centroids)
        {
            aabb::compute_aabb_centroid(overlap_min, overlap_max, out_centroid);
            gs::compute_density(out_centroid, mean, covi, opacity, out_density);
        }

        // 3. Get the physical dimensions of the overlap box
        float3 dims;
        aabb::compute_aabb_dim_size(overlap_min, overlap_max, dims);

        // 4. Find the smallest and largest dimensions
        float min_dim = fminf(dims.x, fminf(dims.y, dims.z));
        float max_dim = fmaxf(dims.x, fmaxf(dims.y, dims.z));

        // Assume cubic voxels: size is max - min on any axis
        float voxel_size = vx_ab_max.x - vx_ab_min.x;

        // 5. Calculate Metrics (with safeguards against divide-by-zero)
        out_aspect_ratio = (max_dim > 1e-6f) ? (min_dim / max_dim) : 0.0f;
        out_penetration = (voxel_size > 1e-6f) ? (min_dim / voxel_size) : 0.0f;
    }

    template <bool multiple_isos>
    __global__ void query_gs_voxel_pair_intersection_brute_force_kernel(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float *__restrict__ opacities,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
        const float3 *__restrict__ contact_points,
        const float *__restrict__ isos,
        const float iso,
        const float ar_threshold,
        const float p_threshold,
        const bool return_centroids,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_voxel_ids,
        int64_t *__restrict__ out_gaus_ids,
        float3 *__restrict__ centroids,
        float *__restrict__ densities,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity)
    {
        uint32_t v_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= num_voxels)
            return;

        float3 vx_ab_min = vx_aabb_mins[v_idx];
        float3 vx_ab_max = vx_aabb_maxs[v_idx];

        bool any_hit = false;

        for (uint32_t g_idx = 0; g_idx < num_gaussians; g_idx++)
        {
            float3 mean = means[g_idx];
            const float *covi = covis + (g_idx * 6);
            float opacity = opacities[g_idx];
            float3 gs_ab_min = gs_aabb_mins[g_idx];
            float3 gs_ab_max = gs_aabb_maxs[g_idx];
            float3 cp0 = contact_points[g_idx * 3 + 0];
            float3 cp1 = contact_points[g_idx * 3 + 1];
            float3 cp2 = contact_points[g_idx * 3 + 2];
            float __iso = multiple_isos ? isos[g_idx] : iso;

            bool hit = test_gs_intersect_voxel(
                g_idx,
                mean,
                covi,
                gs_ab_min,
                gs_ab_max,
                cp0,
                cp1,
                cp2,
                vx_ab_min,
                vx_ab_max,
                __iso);

            if (hit)
            {
                float3 centroid;
                float density;
                float aspect_ratio;
                float penetration;

                // Calculate the metrics of the intersection
                compute_overlap_metrics(
                    gs_ab_min,
                    gs_ab_max,
                    vx_ab_min,
                    vx_ab_max,
                    mean,
                    covi,
                    opacity,
                    return_centroids,
                    centroid,
                    density,
                    aspect_ratio,
                    penetration);

                if (aspect_ratio >= ar_threshold && penetration >= p_threshold)
                {
                    any_hit = true;
                    uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int *)global_counter, 1ULL);

                    if (write_idx < max_capacity)
                    {
                        out_voxel_ids[write_idx] = v_idx;
                        out_gaus_ids[write_idx] = g_idx;

                        if (return_centroids)
                        {
                            centroids[write_idx] = centroid;
                            densities[write_idx] = density;
                        }
                    }
                }
            }
        }

        hit_mask[v_idx] = any_hit;
    }

    void query_gs_voxel_pair_intersection_brute_force(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float *__restrict__ opacities,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
        const float3 *__restrict__ contact_points,
        const float *__restrict__ isos,
        const float iso,
        const float ar_threshold,
        const float p_threshold,
        const bool return_centroids,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_voxel_ids,
        int64_t *__restrict__ out_gaus_ids,
        float3 *__restrict__ centroids,
        float *__restrict__ densities,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity)
    {
        uint32_t threads = 256;
        uint32_t blocks = (num_voxels + threads - 1) / threads;

        if (isos != nullptr)
        {
            query_gs_voxel_pair_intersection_brute_force_kernel<true><<<blocks, threads>>>(
                num_voxels,
                num_gaussians,
                vx_aabb_mins,
                vx_aabb_maxs,
                means,
                covis,
                opacities,
                gs_aabb_mins,
                gs_aabb_maxs,
                contact_points,
                isos,
                iso,
                ar_threshold,
                p_threshold,
                return_centroids,
                hit_mask,
                out_voxel_ids,
                out_gaus_ids,
                centroids,
                densities,
                global_counter,
                max_capacity);
        }
        else
        {
            query_gs_voxel_pair_intersection_brute_force_kernel<false><<<blocks, threads>>>(
                num_voxels,
                num_gaussians,
                vx_aabb_mins,
                vx_aabb_maxs,
                means,
                covis,
                opacities,
                gs_aabb_mins,
                gs_aabb_maxs,
                contact_points,
                isos,
                iso,
                ar_threshold,
                p_threshold,
                return_centroids,
                hit_mask,
                out_voxel_ids,
                out_gaus_ids,
                centroids,
                densities,
                global_counter,
                max_capacity);
        }
    }

    template <bool multiple_isos>
    __global__ void query_gs_edge_pair_intersection_brute_force_kernel(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_edge_ids,
        int64_t *__restrict__ out_gaus_ids,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity)
    {
        uint32_t e_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (e_idx >= num_edges)
            return;

        float3 edge_start = edge_starts[e_idx];
        float3 edge_end = edge_ends[e_idx];

        bool any_hit = false;

        for (uint32_t g_idx = 0; g_idx < num_gaussians; g_idx++)
        {
            float3 mean = means[g_idx];
            const float *covi = covis + (g_idx * 6);

            float3 local_start = edge_start - mean;
            float3 local_end = edge_end - mean;

            float dummy_t_entry;
            float dummy_t_exit;
            float __iso = multiple_isos ? isos[g_idx] : iso;

            bool hit = gs::test_gs_segment(
                covi[0], covi[1], covi[2], covi[3], covi[4], covi[5],
                __iso,
                local_start,
                local_end,
                false,
                dummy_t_entry,
                dummy_t_exit);

            if (hit)
            {
                any_hit = true;
                uint64_t write_idx = (uint64_t)atomicAdd((unsigned long long int *)global_counter, 1ULL);

                if (write_idx < max_capacity)
                {
                    out_edge_ids[write_idx] = e_idx;
                    out_gaus_ids[write_idx] = g_idx;
                }
            }
        }

        hit_mask[e_idx] = any_hit;
    }

    void query_gs_edge_pair_intersection_brute_force(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_edge_ids,
        int64_t *__restrict__ out_gaus_ids,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity)
    {
        uint32_t threads = 256;
        uint32_t blocks = (num_edges + threads - 1) / threads;

        if (isos != nullptr)
        {
            query_gs_edge_pair_intersection_brute_force_kernel<true><<<blocks, threads>>>(
                num_edges,
                num_gaussians,
                edge_starts,
                edge_ends,
                means,
                covis,
                isos,
                iso,
                hit_mask,
                out_edge_ids,
                out_gaus_ids,
                global_counter,
                max_capacity);
        }
        else
        {
            query_gs_edge_pair_intersection_brute_force_kernel<false><<<blocks, threads>>>(
                num_edges,
                num_gaussians,
                edge_starts,
                edge_ends,
                means,
                covis,
                isos,
                iso,
                hit_mask,
                out_edge_ids,
                out_gaus_ids,
                global_counter,
                max_capacity);
        }
    }

    template <bool multiple_isos>
    __global__ void query_gs_edge_intersection_brute_force_kernel(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ means,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids)
    {
        uint32_t e_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (e_idx >= num_edges)
            return;

        float3 edge_start = edge_starts[e_idx];
        float3 edge_end = edge_ends[e_idx];

        bool any_hit = false;

        float max_density = -1.0f;
        int best_gs = -1;

        for (uint32_t g_idx = 0; g_idx < num_gaussians; g_idx++)
        {
            float3 mean = means[g_idx];
            const float *covi = covis + (g_idx * 6);

            float3 local_start = edge_start - mean;
            float3 local_end = edge_end - mean;

            float t_entry;
            float t_exit;
            float __iso = multiple_isos ? isos[g_idx] : iso;

            bool hit = gs::test_gs_segment(
                covi[0], covi[1], covi[2], covi[3], covi[4], covi[5],
                __iso,
                local_start,
                local_end,
                true,
                t_entry,
                t_exit);

            if (hit)
            {
                any_hit = true;
                float t_mid = (fmaxf(t_entry, 0.0f) + fminf(t_exit, 1.0f)) * 0.5f;

                float3 p_mid = local_start + t_mid * (local_end - local_start);

                float density;
                gs::compute_density_local(p_mid, covi, opacities[g_idx], density);

                if (density > max_density)
                {
                    max_density = density;
                    best_gs = g_idx;
                }
            }
        }

        out_gaus_ids[e_idx] = best_gs;
        hit_mask[e_idx] = any_hit;
    }

    void query_gs_edge_intersection_brute_force(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ means,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids)
    {
        uint32_t threads = 256;
        uint32_t blocks = (num_edges + threads - 1) / threads;

        if (isos != nullptr)
        {
            query_gs_edge_intersection_brute_force_kernel<true><<<blocks, threads>>>(
                num_edges,
                num_gaussians,
                edge_starts,
                edge_ends,
                means,
                opacities,
                covis,
                isos,
                iso,
                hit_mask,
                out_gaus_ids);
        }
        else
        {
            query_gs_edge_intersection_brute_force_kernel<false><<<blocks, threads>>>(
                num_edges,
                num_gaussians,
                edge_starts,
                edge_ends,
                means,
                opacities,
                covis,
                isos,
                iso,
                hit_mask,
                out_gaus_ids);
        }
    }
}