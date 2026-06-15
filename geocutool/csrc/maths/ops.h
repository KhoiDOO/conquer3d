#ifndef OPS_H
#define OPS_H

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

#ifndef __CUDACC__
static inline __host__ __device__ double rsqrt(double a) {
    return 1. / sqrt(a);
}

static inline __host__ __device__ float rsqrtf(float a) {
    return 1. / sqrtf(a);
}
#endif

// static __host__ __device__ __forceinline__ float project_edge(
//     const float3& vertex, const float3& edge, const float3& point) {
//     const float3 point_vec = point - vertex;
//     const float length = dot2(edge);
//     return dot(point_vec, edge) / length;
// }

// static __host__ __device__ __forceinline__ float3 project_plane(
//     const float3& vertex, const float3& normal, const float3& point) {
//     const float3 unit_normal = normalize(normal);
//     const float dist = (point.x - vertex.x) * unit_normal.x + \
//                         (point.y - vertex.y) * unit_normal.y + \
//                         (point.z - vertex.z) * unit_normal.z;
//     return point - (unit_normal * dist);
// }

// static __host__ __device__ __forceinline__ bool is_not_above(
//     const float3& vertex, const float3& edge, const float3& normal, const float3& point) {
//     const float3 edge_normal = cross(normal, edge);
//     return dot(edge_normal, point - vertex) <= 0;
// }

// static __host__ __device__ __forceinline__ float3 point_at(
//     const float3& vertex, const float3& edge, const float& t) {
//     return vertex + (edge * t);
// }

#endif // OPS_H