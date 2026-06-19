#ifndef F3x1_H
#define F3x1_H

#include "ops.h"

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

static inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline __host__ __device__ void operator+=(float3 &a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

static inline __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline __host__ __device__ void operator-=(float3 &a, float3 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

static inline __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

static inline __host__ __device__ float3 operator*(float b, float3 a) {
    return make_float3(b * a.x, b * a.y, b * a.z);
}

static inline __host__ __device__ void operator*=(float3 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}

static inline __host__ __device__ float3 operator/(float3 a, const float b) {
    float inv = 1.0f / b;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

static inline __host__ __device__ void operator/=(float3 &a, float b) {
    float inv = 1.0f / b;
    a.x *= inv; a.y *= inv; a.z *= inv;
}

namespace maths
{
    static inline __host__ __device__ float dot(float3 a, float3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static inline __host__ __device__ float dot2(float3 a) {
        return dot(a, a);
    }

    static inline __host__ __device__ float norm(float3 a) {
        return sqrtf(dot2(a));
    }

    static inline __host__ __device__ float3 cross(float3 a, float3 b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    static inline __host__ __device__ float3 normalize(float3 v) {
        float invLen = rsqrtf(dot2(v));
        return v * invLen;
    }

    static inline __host__ __device__ bool equals(float3 a, float3 b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
}

#endif // F3x1_H