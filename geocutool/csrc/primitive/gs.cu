#include "gs.h"

#include <math_constants.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

namespace gs 
{
    __device__ __forceinline__ void compute_single_aabb(
        const float3& mean,
        const float4& rot,
        const float3& scale,
        const float& iso,
        const float& tol,
        const uint32_t level,
        const bool rotnorm,
        float3& out_min,
        float3& out_max)
    {
        float two_level = (float)(0x1 << level);
        float voxelSize = 2.0f/two_level;
        float min_scale = tol*voxelSize;
        float3 modified_scale = scale;

        modified_scale.x = fmaxf(scale.x, min_scale);
        modified_scale.y = fmaxf(scale.y, min_scale);
        modified_scale.z = fmaxf(scale.z, min_scale);

        double detS = ((double)modified_scale.x) * ((double)modified_scale.y) * ((double)modified_scale.z);

        float3x3 S_inv = make_float3x3(
            1.0f/modified_scale.x, 0.0f, 0.0f,
            0.0f, 1.0f/modified_scale.y, 0.0f,
            0.0f, 0.0f, 1.0f/modified_scale.z
        ); // $S^{-1}$

        float4 q = rot;
        float r = q.x;
        float x = q.y;
        float y = q.z;
        float z = q.w;

        if (rotnorm) 
        {
            float inv_norm = rsqrtf(r*r + x*x + y*y + z*z);
            r *= inv_norm;
            x *= inv_norm;
            y *= inv_norm;
            z *= inv_norm;
        }

        float3x3 R_T = make_float3x3(
            1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
            2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
            2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y)
        ); // $R^T$

        // M = S^{-1} R^T
        float3x3 M = S_inv * R_T;
        
        // M^T M 
        // = (S^{-1} R^T)^T (S^{-1} R^T) 
        // = R S^{-T} S^{-1} R^T 
        // = R S^{-2} R^T
        // = \Sigma
        float3x3 Sigma_inv = transpose(M) * M;

        double c0 = Sigma_inv.m[0][0];
        double c1 = Sigma_inv.m[0][1];
        double c2 = Sigma_inv.m[0][2];
        double c3 = Sigma_inv.m[1][1];
        double c4 = Sigma_inv.m[1][2];
        double c5 = Sigma_inv.m[2][2];

        double h0 = c3*c5 - c4*c4;
        double h1 = c2*c4 - c1*c5;
        double h2 = c1*c4 - c2*c3;
        double h3 = c0*c5 - c2*c2;
        double h4 = c1*c2 - c0*c4;
        double h5 = c0*c3 - c1*c1;

        double w[3];
        w[0] = detS*sqrt(iso/h0);
        w[1] = detS*sqrt(iso/h3);
        w[2] = detS*sqrt(iso/h5);

        double3 Q[3];
        Q[0] = make_double3(h0, h1, h2);
        Q[1] = make_double3(h1, h3, h4);
        Q[2] = make_double3(h2, h4, h5);

        float3 P[6];
        for (int i = 0; i < 3; i++)
        {
            P[2*i] = make_float3((float)(w[i]*Q[i].x), (float)(w[i]*Q[i].y), (float)(w[i]*Q[i].z));
            P[2*i+1] = -1.0f*P[2*i];
        }

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

    __global__ void compute_aabb_kernel(
        const uint32_t num_gaussians,
        const float3* __restrict__ means,
        const float4* __restrict__ rotations,
        const float3* __restrict__ scales,
        const float iso,
        const float tol,
        const uint32_t level,
        const bool rotnorm,
        float3* __restrict__ aabb_min,
        float3* __restrict__ aabb_max)
    {
        uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

        if (tidx >= num_gaussians) return;

        compute_single_aabb(
            means[tidx],
            rotations[tidx],
            scales[tidx],
            iso,
            tol,
            level,
            rotnorm,
            aabb_min[tidx],
            aabb_max[tidx]
        );
    }

    void get_aabb(
        const uint32_t num_gaussians,
        const float3* __restrict__ means,
        const float4* __restrict__ rotations,
        const float3* __restrict__ scales,
        const float iso,
        const float tol,
        const uint32_t level,
        const bool rotnorm,
        float3* __restrict__ aabb_min,
        float3* __restrict__ aabb_max)
    {
        uint32_t threads = 256;
        uint32_t blocks = (num_gaussians + threads - 1) / threads;

        compute_aabb_kernel<<<blocks, threads>>>(
            num_gaussians,
            means,
            rotations,
            scales,
            iso,
            tol,
            level,
            rotnorm,
            aabb_min,
            aabb_max
        );
    }
}