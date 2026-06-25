#ifndef TRIANGLES_H
#define TRIANGLES_H

#include "../maths/maths.h"

namespace triangle
{

    __device__ __inline__ float3 compute_normal(const float3 &v0, const float3 &v1, const float3 &v2)
    {
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        return maths::normalize(maths::cross(edge1, edge2));
    }

    __device__ __inline__ float compute_area(const float3 &v0, const float3 &v1, const float3 &v2)
    {
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        return 0.5f * maths::norm(maths::cross(edge1, edge2));
    }

    __device__ __inline__ void compute_aabb(const float3 &v0, const float3 &v1, const float3 &v2, float3 &aabb_min, float3 &aabb_max)
    {
        aabb_min = make_float3(fminf(v0.x, fminf(v1.x, v2.x)), fminf(v0.y, fminf(v1.y, v2.y)), fminf(v0.z, fminf(v1.z, v2.z)));
        aabb_max = make_float3(fmaxf(v0.x, fmaxf(v1.x, v2.x)), fmaxf(v0.y, fmaxf(v1.y, v2.y)), fmaxf(v0.z, fmaxf(v1.z, v2.z)));
    }

    // Tests if two triangles intersect
    // Uses a robust variation of the Moller / Devillers & Guigue interval overlap check
    __device__ __inline__ bool test_intersection(const float3& V0, const float3& V1, const float3& V2,
                                                 const float3& U0, const float3& U1, const float3& U2)
    {
        float3 E1 = V1 - V0;
        float3 E2 = V2 - V0;
        float3 N1 = maths::cross(E1, E2);
        
        float dU0 = maths::dot(N1, U0 - V0);
        float dU1 = maths::dot(N1, U1 - V0);
        float dU2 = maths::dot(N1, U2 - V0);
        
        // Early rejection if all vertices of T2 are strictly on one side of T1
        if (dU0 * dU1 > 0.0f && dU0 * dU2 > 0.0f) return false;
        
        float3 D1 = U1 - U0;
        float3 D2 = U2 - U0;
        float3 N2 = maths::cross(D1, D2);
        
        float dV0 = maths::dot(N2, V0 - U0);
        float dV1 = maths::dot(N2, V1 - U0);
        float dV2 = maths::dot(N2, V2 - U0);
        
        // Early rejection if all vertices of T1 are strictly on one side of T2
        if (dV0 * dV1 > 0.0f && dV0 * dV2 > 0.0f) return false;
        
        // Direction of the intersection line of the two planes
        float3 D = maths::cross(N1, N2);
        
        // Projections of the vertices onto the intersection line
        float pV0 = maths::dot(D, V0);
        float pV1 = maths::dot(D, V1);
        float pV2 = maths::dot(D, V2);
        
        float pU0 = maths::dot(D, U0);
        float pU1 = maths::dot(D, U1);
        float pU2 = maths::dot(D, U2);
        
        float isect1[2], isect2[2];
        
        // Find intersection interval of T1 on the line
        if (dV0 * dV1 > 0.0f) {
            isect1[0] = pV2 + (pV0 - pV2) * dV2 / (dV2 - dV0);
            isect1[1] = pV2 + (pV1 - pV2) * dV2 / (dV2 - dV1);
        } else if (dV0 * dV2 > 0.0f) {
            isect1[0] = pV1 + (pV0 - pV1) * dV1 / (dV1 - dV0);
            isect1[1] = pV1 + (pV2 - pV1) * dV1 / (dV1 - dV2);
        } else {
            isect1[0] = pV0 + (pV1 - pV0) * dV0 / (dV0 - dV1);
            isect1[1] = pV0 + (pV2 - pV0) * dV0 / (dV0 - dV2);
        }
        
        // Find intersection interval of T2 on the line
        if (dU0 * dU1 > 0.0f) {
            isect2[0] = pU2 + (pU0 - pU2) * dU2 / (dU2 - dU0);
            isect2[1] = pU2 + (pU1 - pU2) * dU2 / (dU2 - dU1);
        } else if (dU0 * dU2 > 0.0f) {
            isect2[0] = pU1 + (pU0 - pU1) * dU1 / (dU1 - dU0);
            isect2[1] = pU1 + (pU2 - pU1) * dU1 / (dU1 - dU2);
        } else {
            isect2[0] = pU0 + (pU1 - pU0) * dU0 / (dU0 - dU1);
            isect2[1] = pU0 + (pU2 - pU0) * dU0 / (dU0 - dU2);
        }
        
        // Sort intervals
        if (isect1[0] > isect1[1]) { float tmp = isect1[0]; isect1[0] = isect1[1]; isect1[1] = tmp; }
        if (isect2[0] > isect2[1]) { float tmp = isect2[0]; isect2[0] = isect2[1]; isect2[1] = tmp; }
        
        // Test for overlap
        if (isect1[1] < isect2[0] || isect2[1] < isect1[0]) return false;
        
        return true;
    }

} // namespace triangle

#endif // TRIANGLES_H