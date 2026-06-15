#ifndef F4x1_H
#define F4x1_H

#include "ops.h"

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

static inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static inline __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

static inline __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

static inline __host__ __device__ void operator-=(float4 &a, float4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

static inline __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

static inline __host__ __device__ float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

static inline __host__ __device__ void operator*=(float4 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

static inline __host__ __device__ float4 operator/(float4 a, const float b) {
    float inv = 1.0f / b;
    return make_float4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
}

static inline __host__ __device__ void operator/=(float4 &a, float b) {
    float inv = 1.0f / b;
    a.x *= inv; a.y *= inv; a.z *= inv; a.w *= inv;
}

namespace maths
{
    static inline __host__ __device__ float dot(float4 a, float4 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    static inline __host__ __device__ float dot2(float4 a) {
        return dot(a, a);
    }

    static inline __host__ __device__ float norm(float4 a) {
        return sqrtf(dot2(a));
    }

    static inline __host__ __device__ float4 normalize(float4 v) {
        float invLen = rsqrtf(dot2(v));
        return v * invLen;
    }

    static inline __host__ __device__ bool equals(float4 a, float4 b) {
        return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
    }
}

#endif // F4x1_H