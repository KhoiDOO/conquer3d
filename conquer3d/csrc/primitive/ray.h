#ifndef RAY_H
#define RAY_H

#include "../maths/maths.h"

#include <cuda_runtime.h>

struct Ray {
    float3 origin;
    float3 direction;
    float3 inv_direction; // Precomputed for fast AABB intersection (Slab method)
    float t_min;
    float t_max;

    __host__ __device__ __forceinline__ Ray() 
        : origin(make_float3(0.0f, 0.0f, 0.0f)), 
          direction(make_float3(0.0f, 0.0f, 1.0f)), 
          inv_direction(make_float3(1e30f, 1e30f, 1.0f)), 
          t_min(0.0f), 
          t_max(1e30f) {}

    __host__ __device__ __forceinline__ Ray(const float3& o, const float3& d, float tmin = 0.0f, float tmax = 1e30f) 
        : origin(o), direction(d), t_min(tmin), t_max(tmax) {
        
        // Precompute inverse direction, avoiding divide-by-zero
        inv_direction.x = (d.x == 0.0f) ? 1e30f : 1.0f / d.x;
        inv_direction.y = (d.y == 0.0f) ? 1e30f : 1.0f / d.y;
        inv_direction.z = (d.z == 0.0f) ? 1e30f : 1.0f / d.z;
    }

    __host__ __device__ __forceinline__ float3 at(float t) const {
        return origin + t * direction;
    }

    __host__ __device__ __forceinline__ bool is_intersect_aabb(const float3& aabb_min, const float3& aabb_max, float& t_hit) const {
        float tx1 = (aabb_min.x - origin.x) * inv_direction.x;
        float tx2 = (aabb_max.x - origin.x) * inv_direction.x;
        
        float tmin = fminf(tx1, tx2);
        float tmax = fmaxf(tx1, tx2);

        float ty1 = (aabb_min.y - origin.y) * inv_direction.y;
        float ty2 = (aabb_max.y - origin.y) * inv_direction.y;

        tmin = fmaxf(tmin, fminf(ty1, ty2));
        tmax = fminf(tmax, fmaxf(ty1, ty2));

        float tz1 = (aabb_min.z - origin.z) * inv_direction.z;
        float tz2 = (aabb_max.z - origin.z) * inv_direction.z;

        tmin = fmaxf(tmin, fminf(tz1, tz2));
        tmax = fminf(tmax, fmaxf(tz1, tz2));

        if (tmax >= tmin && tmax >= this->t_min && tmin <= this->t_max) {
            t_hit = (tmin >= this->t_min) ? tmin : tmax;
            return true;
        }
        return false;
    }
};

#endif // RAY_H
