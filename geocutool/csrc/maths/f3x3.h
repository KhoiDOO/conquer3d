#ifndef F3X3_H
#define F3X3_H

#include "f2x2.h"
#include "ops.h"

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

typedef struct
{
    float m[3][3];
} float3x3;

static __inline__ __host__ __device__ float3x3 make_float3x3(
    float a00, float a01, float a02,
    float a10, float a11, float a12,
    float a20, float a21, float a22) {
    float3x3 a;
    a.m[0][0] = a00; a.m[0][1] = a01; a.m[0][2] = a02;
    a.m[1][0] = a10; a.m[1][1] = a11; a.m[1][2] = a12;
    a.m[2][0] = a20; a.m[2][1] = a21; a.m[2][2] = a22;
    return a;
}

// [3, 3] x [3, 3] = [3, 3]
static __inline__ __host__ __device__ float3x3 operator* (const float3x3& a, const float3x3& b)
{
    float3x3 c;

    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1];
    c.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2];

    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1];
    c.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2];

    c.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0];
    c.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1];
    c.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2];

    return c;
}

// [3, 1]^T x [3, 3] = [3, 1]^T
static __inline__ __host__ __device__ float3 operator*(const float3& a, const float3x3& m) {
    return make_float3(
        a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0],
        a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1],
        a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2]
    );
}

// [3, 3] x [3, 1] = [3, 1]
static __inline__ __host__ __device__ float3 operator*(const float3x3& m, const float3& a) {
    return make_float3(
        m.m[0][0] * a.x + m.m[0][1] * a.y + m.m[0][2] * a.z,
        m.m[1][0] * a.x + m.m[1][1] * a.y + m.m[1][2] * a.z,
        m.m[2][0] * a.x + m.m[2][1] * a.y + m.m[2][2] * a.z
    );
}

namespace maths
{
    static __inline__ __host__ __device__ float3x3 transpose(const float3x3& a) {
        float3x3 b;
        b.m[0][0] = a.m[0][0]; b.m[0][1] = a.m[1][0]; b.m[0][2] = a.m[2][0];
        b.m[1][0] = a.m[0][1]; b.m[1][1] = a.m[1][1]; b.m[1][2] = a.m[2][1];
        b.m[2][0] = a.m[0][2]; b.m[2][1] = a.m[1][2]; b.m[2][2] = a.m[2][2];
        return b;
    }

    static __inline__ __host__ __device__ float det(const float3x3& a) {
        return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1])
            - a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0])
            + a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
    }

    static inline __host__ __device__ void copy(float3x3 &a, float3x3 b) {
        a.m[0][0] = b.m[0][0]; a.m[0][1] = b.m[0][1]; a.m[0][2] = b.m[0][2];
        a.m[1][0] = b.m[1][0]; a.m[1][1] = b.m[1][1]; a.m[1][2] = b.m[1][2];
        a.m[2][0] = b.m[2][0]; a.m[2][1] = b.m[2][1]; a.m[2][2] = b.m[2][2];
    }

    static __host__ __device__ __forceinline__ bool invert(const float3x3& m, float3x3& out_inv) 
    {
        float a = m.m[0][0], b = m.m[0][1], c = m.m[0][2];
        float d = m.m[1][0], e = m.m[1][1], f = m.m[1][2];
        float g = m.m[2][0], h = m.m[2][1], i = m.m[2][2];

        float A = e * i - f * h;
        float B = f * g - d * i;
        float C = d * h - e * g;

        float det_val = a * A + b * B + c * C;

        if (fabsf(det_val) < 1e-8f) return false; 

        float inv_det = 1.0f / det_val;

        out_inv.m[0][0] = A * inv_det;
        out_inv.m[0][1] = (c * h - b * i) * inv_det;
        out_inv.m[0][2] = (b * f - c * e) * inv_det;
        out_inv.m[1][0] = B * inv_det;
        out_inv.m[1][1] = (a * i - c * g) * inv_det;
        out_inv.m[1][2] = (c * d - a * f) * inv_det;
        out_inv.m[2][0] = C * inv_det;
        out_inv.m[2][1] = (b * g - a * h) * inv_det;
        out_inv.m[2][2] = (a * e - b * d) * inv_det;

        return true;
    }
}
#endif // F3X3_H