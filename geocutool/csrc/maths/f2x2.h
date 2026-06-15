#ifndef F2X2_H
#define F2X2_H

#include "ops.h"

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

typedef struct
{
    float m[2][2];
} float2x2;

static __inline__ __host__ __device__ float2x2 make_float2x2(
    float a00, float a01,
    float a10, float a11) {
    float2x2 a;
    a.m[0][0] = a00; a.m[0][1] = a01;
    a.m[1][0] = a10; a.m[1][1] = a11;
    return a;
}

// [2, 2] x [2, 2] = [2, 2]
static __inline__ __host__ __device__ float2x2 operator* (const float2x2& a, const float2x2& b)
{
    float2x2 c;

    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1];
    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1];

    return c;
}

// [2, 1]^T x [2, 2] = [2, 1]^T
static __inline__ __host__ __device__ float2 operator*(const float2& a, const float2x2& m) {
    return make_float2(
        a.x * m.m[0][0] + a.y * m.m[1][0],
        a.x * m.m[0][1] + a.y * m.m[1][1]
    );
}

// [2, 2] x [2, 1] = [2, 1]
static __inline__ __host__ __device__ float2 operator*(const float2x2& m, const float2& a) {
    return make_float2(
        m.m[0][0] * a.x + m.m[0][1] * a.y,
        m.m[1][0] * a.x + m.m[1][1] * a.y
    );
}

namespace maths
{
    static __inline__ __host__ __device__ float2x2 transpose(const float2x2& a) {
        float2x2 b;
        b.m[0][0] = a.m[0][0]; b.m[0][1] = a.m[1][0];
        b.m[1][0] = a.m[0][1]; b.m[1][1] = a.m[1][1];
        return b;
    }

    static __inline__ __host__ __device__ float det(const float2x2& a) {
        return a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0];
    }

    static inline __host__ __device__ void copy(float2x2 &a, float2x2 b) {
        a.m[0][0] = b.m[0][0]; a.m[0][1] = b.m[0][1];
        a.m[1][0] = b.m[1][0]; a.m[1][1] = b.m[1][1];
    }

    static __host__ __device__ __forceinline__ bool invert(const float2x2& m, float2x2& out_inv) 
    {
        float det_val = m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0];

        if (fabsf(det_val) < 1e-8f) return false;

        float inv_det = 1.0f / det_val;

        out_inv.m[0][0] =  m.m[1][1] * inv_det;
        out_inv.m[0][1] = -m.m[0][1] * inv_det;
        out_inv.m[1][0] = -m.m[1][0] * inv_det;
        out_inv.m[1][1] =  m.m[0][0] * inv_det;

        return true;
    }
}

#endif // F2X2_H