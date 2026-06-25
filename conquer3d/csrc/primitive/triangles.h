#include "../maths/maths.h"

__device__ float3 compute_normal(const float3 &v0, const float3 &v1, const float3 &v2)
{
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    return maths::normalize(maths::cross(edge1, edge2));
}

__device__ float compute_area(const float3 &v0, const float3 &v1, const float3 &v2)
{
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    return 0.5f * maths::norm(maths::cross(edge1, edge2));
}