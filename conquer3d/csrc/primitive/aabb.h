#ifndef AABB_H
#define AABB_H

#include "../maths/maths.h"

#include <cuda_runtime.h>

namespace aabb
{
    __device__ __forceinline__ void compute_aabb_centroid(
        const float3 &aabb_min,
        const float3 &aabb_max,
        float3 &out_centroid)
    {
        out_centroid = make_float3(
            (aabb_min.x + aabb_max.x) * 0.5f,
            (aabb_min.y + aabb_max.y) * 0.5f,
            (aabb_min.z + aabb_max.z) * 0.5f);
    }

    __device__ __forceinline__ void compute_aabb_overlap(
        const float3 &query_aabb_min,
        const float3 &query_aabb_max,
        const float3 &target_aabb_min,
        const float3 &target_aabb_max,
        float3 &overlap_min,
        float3 &overlap_max)
    {
        overlap_min = make_float3(
            fmaxf(query_aabb_min.x, target_aabb_min.x),
            fmaxf(query_aabb_min.y, target_aabb_min.y),
            fmaxf(query_aabb_min.z, target_aabb_min.z));

        overlap_max = make_float3(
            fminf(query_aabb_max.x, target_aabb_max.x),
            fminf(query_aabb_max.y, target_aabb_max.y),
            fminf(query_aabb_max.z, target_aabb_max.z));
    }

    __device__ __forceinline__ void compute_aabb_union(
        const float3 &query_aabb_min,
        const float3 &query_aabb_max,
        const float3 &target_aabb_min,
        const float3 &target_aabb_max,
        float3 &out_union_min,
        float3 &out_union_max)
    {
        out_union_min = make_float3(
            fminf(query_aabb_min.x, target_aabb_min.x),
            fminf(query_aabb_min.y, target_aabb_min.y),
            fminf(query_aabb_min.z, target_aabb_min.z));

        out_union_max = make_float3(
            fmaxf(query_aabb_max.x, target_aabb_max.x),
            fmaxf(query_aabb_max.y, target_aabb_max.y),
            fmaxf(query_aabb_max.z, target_aabb_max.z));
    }

    __device__ __forceinline__ void compute_aabb_extent(
        const float3 &aabb_min,
        const float3 &aabb_max,
        float3 &out_extent)
    {
        out_extent = make_float3(
            fmaxf(1e-6f, aabb_max.x - aabb_min.x),
            fmaxf(1e-6f, aabb_max.y - aabb_min.y),
            fmaxf(1e-6f, aabb_max.z - aabb_min.z));
    }

    __device__ __forceinline__ void compute_aabb_dim_size(
        const float3 &aabb_min,
        const float3 &aabb_max,
        float3 &out_dim_size)
    {
        out_dim_size = make_float3(
            fmaxf(0.0f, aabb_max.x - aabb_min.x),
            fmaxf(0.0f, aabb_max.y - aabb_min.y),
            fmaxf(0.0f, aabb_max.z - aabb_min.z));
    }

    __device__ __forceinline__ void compute_aabb_volume(
        const float3 &aabb_min,
        const float3 &aabb_max,
        float &out_volume)
    {
        float3 dim_size;
        compute_aabb_dim_size(aabb_min, aabb_max, dim_size);
        out_volume = dim_size.x * dim_size.y * dim_size.z;
    }

    __device__ __forceinline__ void compute_aabb_surface_area(
        const float3 &aabb_min,
        const float3 &aabb_max,
        float &out_surface_area)
    {
        float3 dim_size;
        compute_aabb_dim_size(aabb_min, aabb_max, dim_size);
        out_surface_area = 2.0f * (dim_size.x * dim_size.y + dim_size.x * dim_size.z + dim_size.y * dim_size.z);
    }

    __device__ __forceinline__ void compute_aabb_volume_from_dims(
        const float3 &dim_size,
        float &out_volume)
    {
        out_volume = dim_size.x * dim_size.y * dim_size.z;
    }

    __device__ __forceinline__ void compute_aabb_surface_area_from_dims(
        const float3 &dim_size,
        float &out_surface_area)
    {
        out_surface_area = 2.0f * (dim_size.x * dim_size.y + dim_size.x * dim_size.z + dim_size.y * dim_size.z);
    }

    __device__ __forceinline__ bool test_aabb_overlap(
        const float3 &query_aabb_min,
        const float3 &query_aabb_max,
        const float3 &target_aabb_min,
        const float3 &target_aabb_max)
    {
        if (query_aabb_max.x < target_aabb_min.x)
            return false;
        if (query_aabb_max.y < target_aabb_min.y)
            return false;
        if (query_aabb_max.z < target_aabb_min.z)
            return false;
        if (query_aabb_min.x > (target_aabb_max.x))
            return false;
        if (query_aabb_min.y > (target_aabb_max.y))
            return false;
        if (query_aabb_min.z > (target_aabb_max.z))
            return false;

        return true;
    }

    __device__ __forceinline__ bool test_aabb_inside(
        const float3 &query_aabb_min,
        const float3 &query_aabb_max,
        const float3 &target_aabb_min,
        const float3 &target_aabb_max
    )
    {
        return (
            target_aabb_min.x <= query_aabb_min.x &&
            target_aabb_min.y <= query_aabb_min.y &&
            target_aabb_min.z <= query_aabb_min.z &&
            (target_aabb_max.x) >= query_aabb_max.x &&
            (target_aabb_max.y) >= query_aabb_max.y &&
            (target_aabb_max.z) >= query_aabb_max.z);
    }
}

#endif // AABB_H