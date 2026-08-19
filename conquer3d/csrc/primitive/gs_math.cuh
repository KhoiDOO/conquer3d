/**
 * @file gs_math.cuh
 * @brief High-performance CUDA device inline mathematical routines for 3D Gaussian Splatting.
 * @details Implements Mahalanobis distance evaluations, exponential density queries,
 * quaternion-to-rotation conversions, inverse covariance formulations, and analytical
 * ellipsoid-segment quadratic intersection tests.
 */

#ifndef GS_MATH_CUH
#define GS_MATH_CUH

#pragma once
#include "../maths/maths.h"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace gs
{

    /**
     * @brief Computes squared Mahalanobis distance $d^2 = (p - \mu)^T \Sigma^{-1} (p - \mu)$ on device.
     * 
     * @param[in]  point         Query point coordinate in world space.
     * @param[in]  mean          Gaussian centroid coordinate $\mu$.
     * @param[in]  covi          Upper-triangular packed inverse covariance array $\Sigma^{-1}$ of length 6.
     * @param[out] out_distance  Computed squared Mahalanobis distance.
     */
    __device__ __forceinline__ void compute_mahalanobis_distance(
        const float3 &point,
        const float3 &mean,
        const float *covi,
        float &out_distance)
    {
        float3 d = point - mean;

        out_distance = 
            d.x * (d.x * covi[0] + d.y * covi[1] + d.z * covi[2]) + 
            d.y * (d.x * covi[1] + d.y * covi[3] + d.z * covi[4]) +
            d.z * (d.x * covi[2] + d.y * covi[4] + d.z * covi[5]);
    }

    /**
     * @brief Evaluates Gaussian volumetric density $\rho(p) = \alpha \exp(-\frac{1}{2} d^2)$ at point $p$.
     * 
     * @param[in]  point       Query point in world space.
     * @param[in]  mean        Gaussian centroid $\mu$.
     * @param[in]  covi        Upper-triangular packed inverse covariance entries $\Sigma^{-1}$.
     * @param[in]  opacity     Gaussian opacity scaling factor $\alpha$.
     * @param[out] out_density Computed scalar density clamped to zero for $d^2 > 30$.
     */
    __device__ __forceinline__ void compute_density(
        const float3 &point,
        const float3 &mean,
        const float *covi,
        const float opacity,
        float &out_density)
    {
        float mahal_dist;
        compute_mahalanobis_distance(point, mean, covi, mahal_dist);

        float power = -0.5f * mahal_dist;

        if (power > 0.0f || power < -15.0f)
        {
            out_density = 0.0f;
        }
        else
        {
            out_density = opacity * expf(power);
        }
    }

    /**
     * @brief Evaluates local-origin Gaussian density $\rho(p) = \alpha \exp(-\frac{1}{2} p^T \Sigma^{-1} p)$.
     */
    __device__ __forceinline__ void compute_density_local(
        const float3 &point,
        const float *covi,
        const float opacity,
        float &out_density)
    {
        float mahal_dist;
        compute_mahalanobis_distance(point, make_float3(0.0f, 0.0f, 0.0f), covi, mahal_dist);
        float power = -0.5f * mahal_dist;

        if (power > 0.0f || power < -15.0f)
        {
            out_density = 0.0f;
        }
        else
        {
            out_density = opacity * expf(power);
        }
    }

    /**
     * @brief Computes diagonal inverse scale matrix $S^{-1} = \text{diag}(1/s_x, 1/s_y, 1/s_z)$.
     */
    __device__ __forceinline__ void compute_inverse_scale(
        const float3 &scale, 
        float3x3 &out_inv_scale
    ) {
        out_inv_scale = make_float3x3(
            1.0f / scale.x, 0.0f, 0.0f,
            0.0f, 1.0f / scale.y, 0.0f,
            0.0f, 0.0f, 1.0f / scale.z);
    }

    /**
     * @brief Converts unit quaternion $(r, x, y, z)$ into 3x3 orthonormal rotation matrix $R$.
     * 
     * @param[in]  rot           Quaternion representation $(r, x, y, z)$.
     * @param[out] out_rotation  Resulting 3x3 orthonormal rotation matrix.
     * @param[in]  rotnorm       Whether to normalize quaternion before conversion.
     * @param[in]  transpose     Whether to output transposed rotation matrix $R^T$.
     */
    __device__ __forceinline__ void compute_rotation(
        const float4 &rot,
        float3x3 &out_rotation,
        const bool rotnorm,
        const bool transpose
    ) {
        float4 q = rot;
        float r = q.x;
        float x = q.y;
        float y = q.z;
        float z = q.w;

        if (rotnorm)
        {
            float inv_norm = rsqrtf(r * r + x * x + y * y + z * z);
            r *= inv_norm;
            x *= inv_norm;
            y *= inv_norm;
            z *= inv_norm;
        }

        if (transpose)
        {
            out_rotation = make_float3x3(
                1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
                2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
                2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y));
        } else {
            out_rotation = make_float3x3(
                1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
                2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
        }
    }

    /**
     * @brief Computes 3D inverse covariance matrix $\Sigma^{-1} = R S^{-2} R^T$.
     */
    __device__ __forceinline__ void compute_cov_inverse(
        const float3x3 &inv_scale,
        const float3x3 &rotation_transpose,
        float *covi)
    {
        // M = S^{-1} R^T
        float3x3 M = inv_scale * rotation_transpose;

        // M^T M
        // = (S^{-1} R^T)^T (S^{-1} R^T)
        // = R S^{-T} S^{-1} R^T
        // = R S^{-2} R^T
        // = \Sigma^{-1}
        float3x3 out_cov_inv = maths::transpose(M) * M;

        covi[0] = out_cov_inv.m[0][0];
        covi[1] = out_cov_inv.m[0][1];
        covi[2] = out_cov_inv.m[0][2];
        covi[3] = out_cov_inv.m[1][1];
        covi[4] = out_cov_inv.m[1][2];
        covi[5] = out_cov_inv.m[2][2];
    }

    /**
     * @brief Analytical ray/segment intersection test against 3D Gaussian ellipsoid isosurface.
     * 
     * @details Solves the quadratic equation $a t^2 + b t + c = 0$ along the ray $p(t) = P_0 + t (P_1 - P_0)$.
     * 
     * @param[in]  c0..c5        Six upper-triangular components of $\Sigma^{-1}$.
     * @param[in]  iso           Mahalanobis radius squared threshold $r^2$.
     * @param[in]  segment_start Segment start position $P_0$.
     * @param[in]  segment_end   Segment end position $P_1$.
     * @param[in]  return_t      Whether to compute and clamp entry/exit parameters $t \in [0, 1]$.
     * @param[out] t_entry       Segment parametric entry position $t_{\text{entry}}$.
     * @param[out] t_exit        Segment parametric exit position $t_{\text{exit}}$.
     * 
     * @return bool True if segment intersects the ellipsoid within $t \in [0, 1]$.
     */
    __device__ __forceinline__ bool test_gs_segment(
        const float c0, const float c1, const float c2,
        const float c3, const float c4, const float c5,
        const float iso,
        const float3 &segment_start,
        const float3 &segment_end,
        const bool return_t,
        float &t_entry, float &t_exit
    )
    {
        float3 d = segment_end - segment_start;

        float3 v_d = make_float3(
            c0 * d.x + c1 * d.y + c2 * d.z,
            c1 * d.x + c3 * d.y + c4 * d.z,
            c2 * d.x + c4 * d.y + c5 * d.z);

        float3 v_p0 = make_float3(
            c0 * segment_start.x + c1 * segment_start.y + c2 * segment_start.z,
            c1 * segment_start.x + c3 * segment_start.y + c4 * segment_start.z,
            c2 * segment_start.x + c4 * segment_start.y + c5 * segment_start.z);

        float a = maths::dot(d, v_d);
        float b = 2.0f * maths::dot(segment_start, v_d);
        float c = maths::dot(segment_start, v_p0) - iso;

        float dcrm = fmaxf(b * b - 4.0f * a * c, 0.0f);

        if (dcrm > 0.0f)
        {
            float rdcrm = sqrtf(dcrm) / (2.0f * a);
            float b2a = -b / (2.0f * a);

            t_entry = b2a - rdcrm;
            t_exit = b2a + rdcrm;

            if (return_t)
            {
                t_entry = fmaxf(t_entry, 0.0f);
                t_exit = fminf(t_exit, 1.0f);
            }

            return !(1.0f <= t_entry || t_exit <= 0.0f);
        }
        
        if (return_t)
        {
            t_entry = -1.0f;
            t_exit = -1.0f;
        }

        return false;
    }
}

#endif // GS_MATH_CUH