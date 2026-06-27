#ifndef TRIANGLES_H
#define TRIANGLES_H

#include "../maths/maths.h"

#include "ray.h"

struct Triangle {
    float3 v0, v1, v2;

    __host__ __device__ __forceinline__ Triangle() 
        : v0(make_float3(0.0f, 0.0f, 0.0f)), v1(make_float3(0.0f, 0.0f, 0.0f)), v2(make_float3(0.0f, 0.0f, 0.0f)) {}

    __host__ __device__ __forceinline__ Triangle(const float3& a, const float3& b, const float3& c)
        : v0(a), v1(b), v2(c) {}

    __host__ __device__ __forceinline__ bool is_intersect_ray(const Ray& ray, float& t_hit, float& u, float& v) const {
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        float3 h = maths::cross(ray.direction, edge2);
        float a = maths::dot(edge1, h);

        // If a is near zero, ray is parallel to triangle
        if (a > -1e-6f && a < 1e-6f) return false; 

        float f = 1.0f / a;
        float3 s = ray.origin - v0;
        u = f * maths::dot(s, h);

        if (u < 0.0f || u > 1.0f) return false;

        float3 q = maths::cross(s, edge1);
        v = f * maths::dot(ray.direction, q);

        if (v < 0.0f || u + v > 1.0f) return false;

        float t = f * maths::dot(edge2, q);
        
        // Ensure hit is within ray's valid range
        if (t >= ray.t_min && t <= ray.t_max) { 
            t_hit = t;
            return true;
        }
        return false;
    }

    __host__ __device__ __forceinline__ float3 compute_closest_point(const float3& p) const {
        float3 ab = v1 - v0;
        float3 ac = v2 - v0;
        float3 ap = p - v0;

        float d1 = maths::dot(ab, ap);
        float d2 = maths::dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f) return v0;

        float3 bp = p - v1;
        float d3 = maths::dot(ab, bp);
        float d4 = maths::dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) return v1;

        float vc = d1*d4 - d3*d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            float v = d1 / (d1 - d3);
            return v0 + v * ab;
        }

        float3 cp = p - v2;
        float d5 = maths::dot(ab, cp);
        float d6 = maths::dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6) return v2;

        float vb = d5*d2 - d1*d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            float w = d2 / (d2 - d6);
            return v0 + w * ac;
        }

        float va = d3*d6 - d5*d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return v1 + w * (v2 - v1);
        }

        float denom = 1.0f / (va + vb + vc);
        float v = vb * denom;
        float w = vc * denom;
        return v0 + ab * v + ac * w;
    }

    __host__ __device__ __forceinline__ float3 compute_normal() const {
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        return maths::normalize(maths::cross(edge1, edge2));
    }

    __host__ __device__ __forceinline__ float compute_area() const {
        float3 edge1 = v1 - v0;
        float3 edge2 = v2 - v0;
        return 0.5f * maths::norm(maths::cross(edge1, edge2));
    }

    __host__ __device__ __forceinline__ float3 sample_point(float r1, float r2) const {
        float sqrt_r1 = sqrtf(r1);
        float u = 1.0f - sqrt_r1;
        float v = r2 * sqrt_r1;
        float w = 1.0f - u - v;
        return v0 * u + v1 * v + v2 * w;
    }

    __host__ __device__ __forceinline__ void compute_aabb(float3 &aabb_min, float3 &aabb_max) const {
        aabb_min = make_float3(fminf(v0.x, fminf(v1.x, v2.x)), fminf(v0.y, fminf(v1.y, v2.y)), fminf(v0.z, fminf(v1.z, v2.z)));
        aabb_max = make_float3(fmaxf(v0.x, fmaxf(v1.x, v2.x)), fmaxf(v0.y, fmaxf(v1.y, v2.y)), fmaxf(v0.z, fmaxf(v1.z, v2.z)));
    }

    __device__ __inline__ bool test_intersection(const Triangle& T2) const {
        const float3& V0 = v0;
        const float3& V1 = v1;
        const float3& V2 = v2;
        const float3& U0 = T2.v0;
        const float3& U1 = T2.v1;
        const float3& U2 = T2.v2;
        float3 E1 = V1 - V0;
        float3 E2 = V2 - V0;
        float3 N1 = maths::cross(E1, E2);
        
        float dU0 = maths::dot(N1, U0 - V0);
        float dU1 = maths::dot(N1, U1 - V0);
        float dU2 = maths::dot(N1, U2 - V0);
        
        if (dU0 * dU1 > 0.0f && dU0 * dU2 > 0.0f) return false;
        
        float3 D1 = U1 - U0;
        float3 D2 = U2 - U0;
        float3 N2 = maths::cross(D1, D2);
        
        float dV0 = maths::dot(N2, V0 - U0);
        float dV1 = maths::dot(N2, V1 - U0);
        float dV2 = maths::dot(N2, V2 - U0);
        
        if (dV0 * dV1 > 0.0f && dV0 * dV2 > 0.0f) return false;
        
        float3 D = maths::cross(N1, N2);
        
        float pV0 = maths::dot(D, V0);
        float pV1 = maths::dot(D, V1);
        float pV2 = maths::dot(D, V2);
        
        float pU0 = maths::dot(D, U0);
        float pU1 = maths::dot(D, U1);
        float pU2 = maths::dot(D, U2);
        
        float isect1[2], isect2[2];
        
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
        
        if (isect1[0] > isect1[1]) { float tmp = isect1[0]; isect1[0] = isect1[1]; isect1[1] = tmp; }
        if (isect2[0] > isect2[1]) { float tmp = isect2[0]; isect2[0] = isect2[1]; isect2[1] = tmp; }
        
        if (isect1[1] < isect2[0] || isect2[1] < isect1[0]) return false;
        
        return true;
    }
};

namespace triangle
{
    __device__ __inline__ float3 compute_normal(const float3 &v0, const float3 &v1, const float3 &v2)
    {
        return Triangle(v0, v1, v2).compute_normal();
    }

    __device__ __inline__ float compute_area(const float3 &v0, const float3 &v1, const float3 &v2)
    {
        return Triangle(v0, v1, v2).compute_area();
    }

    __device__ __inline__ void compute_aabb(const float3 &v0, const float3 &v1, const float3 &v2, float3 &aabb_min, float3 &aabb_max)
    {
        Triangle(v0, v1, v2).compute_aabb(aabb_min, aabb_max);
    }

    __device__ __inline__ bool test_intersection(const Triangle& T1, const Triangle& T2)
    {
        return T1.test_intersection(T2);
    }

    __device__ __inline__ float3 sample_point(const float3 &v0, const float3 &v1, const float3 &v2, float r1, float r2)
    {
        return Triangle(v0, v1, v2).sample_point(r1, r2);
    }
} // namespace triangle

#endif // TRIANGLES_H