/**
 * @file pgs_math.cuh
 * @brief High-performance CUDA device inline mathematical routines for Periodic Gaussian Splatting (PGS).
 * @details Implements planar disc segment intersections and pairwise tangency contact radius solvers.
 */

#ifndef PGS_MATH_CUH
#define PGS_MATH_CUH

#include "../maths/maths.h"
#include "gs_math.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace pgs
{
    /**
     * @brief Tests line segment intersection against a Periodic Gaussian disc in device register space.
     * 
     * @param[in]  mean          Gaussian centroid coordinate $\mu$.
     * @param[in]  normal        Principal normal axis $n$ defining the tangent plane.
     * @param[in]  covi          Upper-triangular packed inverse covariance array $\Sigma^{-1}$.
     * @param[in]  iso           Mahalanobis radius squared threshold $r^2$.
     * @param[in]  segment_start Segment start position $P_0$.
     * @param[in]  segment_end   Segment end position $P_1$.
     * @param[out] t_hit         Parametric hit location $t \in [0, 1]$ along segment.
     * 
     * @return bool True if the segment pierces the planar Gaussian disc within the Mahalanobis radius.
     */
    __device__ __forceinline__ bool test_pgs_segment(
        const float3 &mean,
        const float3 &normal,
        const float *covi,
        const float iso,
        const float3 &segment_start,
        const float3 &segment_end,
        float &t_hit)
    {
        float3 dir = segment_end - segment_start;

        float denom = maths::dot(dir, normal);
        if (fabsf(denom) < 1e-6f) return false; 

        float num = maths::dot(mean - segment_start, normal);
        float t = num / denom;

        if (t >= 0.0f && t <= 1.0f) 
        {
            float3 p_hit = segment_start + t * dir;

            float dist_sq;
            gs::compute_mahalanobis_distance(p_hit, mean, covi, dist_sq);

            if (dist_sq <= iso) 
            {
                t_hit = t;
                return true; 
            }
        }

        return false;
    }

    /**
     * @brief Solves the exact pairwise tangency contact radius between two adjacent planar Gaussians.
     * 
     * @details Finds the intersection line of two tangent planes $n_i, n_j$ and minimizes the
     * Mahalanobis distance along the line to find the critical contact isosurface radius $r^2$.
     * 
     * @param[in]  mean            First Gaussian centroid $\mu_i$.
     * @param[in]  normal          First Gaussian normal axis $n_i$.
     * @param[in]  covi            First Gaussian inverse covariance $\Sigma_i^{-1}$.
     * @param[in]  neighbor_mean   Second Gaussian centroid $\mu_j$.
     * @param[in]  neighbor_normal Second Gaussian normal axis $n_j$.
     * @param[out] out_iso         Computed tangency radius squared.
     * 
     * @return bool True if planes intersect non-parallelly and a valid tangency radius exists.
     */
    __device__ __forceinline__ bool solve_pgs_pair_tangency_radius(
        const float3 &mean,
        const float3 &normal,
        const float *covi,
        const float3 &neighbor_mean,
        const float3 &neighbor_normal,
        float &out_iso
    )
    {
        float3 n_i = maths::normalize(normal);
        float3 n_j = maths::normalize(neighbor_normal);

        float3 d = maths::cross(n_i, n_j);

        float d_norm = maths::norm(d);

        if (d_norm < 1e-6f) return false;

        float3x3 m = make_float3x3(
            n_i.x, n_i.y, n_i.z,
            n_j.x, n_j.y, n_j.z,
            d.x, d.y, d.z);

        float n_mu_i = maths::dot(n_i, mean);
        float n_mu_j = maths::dot(n_j, neighbor_mean);

        float3x3 m_inv;
        bool invertible = maths::invert(m, m_inv);

        if (!invertible) return false;
        float3 p = m_inv * make_float3(n_mu_i, n_mu_j, 0.0f);

        float3 q = p - mean;

        float a = q.x * (covi[0] * q.x + covi[1] * q.y + covi[2] * q.z) +
        q.y * (covi[1] * q.x + covi[3] * q.y + covi[4] * q.z) +
        q.z * (covi[2] * q.x + covi[4] * q.y + covi[5] * q.z);
        
        float b = d.x * (covi[0] * q.x + covi[1] * q.y + covi[2] * q.z) +
        d.y * (covi[1] * q.x + covi[3] * q.y + covi[4] * q.z) +
        d.z * (covi[2] * q.x + covi[4] * q.y + covi[5] * q.z);

        float b2 = b * b;

        float c = d.x * (covi[0] * d.x + covi[1] * d.y + covi[2] * d.z) +
        d.y * (covi[1] * d.x + covi[3] * d.y + covi[4] * d.z) +
        d.z * (covi[2] * d.x + covi[4] * d.y + covi[5] * d.z);
        
        float iso_sq = fmaxf(a - b2 / c, 0.0f);
        out_iso = iso_sq;

        return true;
    }
}

#endif // PGS_MATH_CUH