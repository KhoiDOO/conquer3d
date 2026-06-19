#ifndef F4x4_H
#define F4x4_H

#include "f3x4.h"

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

typedef struct
{
    float m[4][4];
} float4x4;

static __inline__ __host__ __device__ float4x4 make_float4x4(
    float a00, float a01, float a02, float a03,
    float a10, float a11, float a12, float a13,
    float a20, float a21, float a22, float a23,
    float a30, float a31, float a32, float a33) {
    float4x4 a;
    a.m[0][0] = a00; a.m[0][1] = a01; a.m[0][2] = a02; a.m[0][3] = a03;
    a.m[1][0] = a10; a.m[1][1] = a11; a.m[1][2] = a12; a.m[1][3] = a13;
    a.m[2][0] = a20; a.m[2][1] = a21; a.m[2][2] = a22; a.m[2][3] = a23;
    a.m[3][0] = a30; a.m[3][1] = a31; a.m[3][2] = a32; a.m[3][3] = a33;
    return a;
}

// [4, 4] x [4, 4] = [4, 4]
static __inline__ __host__ __device__ float4x4  operator*(const float4x4& a, const float4x4& b)
{
    float4x4 c;

    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0] + a.m[0][3] * b.m[3][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1] + a.m[0][3] * b.m[3][1];
    c.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2] + a.m[0][3] * b.m[3][2];
    c.m[0][3] = a.m[0][0] * b.m[0][3] + a.m[0][1] * b.m[1][3] + a.m[0][2] * b.m[2][3] + a.m[0][3] * b.m[3][3];

    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0] + a.m[1][3] * b.m[3][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1] + a.m[1][3] * b.m[3][1];
    c.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2] + a.m[1][3] * b.m[3][2];
    c.m[1][3] = a.m[1][0] * b.m[0][3] + a.m[1][1] * b.m[1][3] + a.m[1][2] * b.m[2][3] + a.m[1][3] * b.m[3][3];

    c.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0] + a.m[2][3] * b.m[3][0];
    c.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1] + a.m[2][3] * b.m[3][1];
    c.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2] + a.m[2][3] * b.m[3][2];
    c.m[2][3] = a.m[2][0] * b.m[0][3] + a.m[2][1] * b.m[1][3] + a.m[2][2] * b.m[2][3] + a.m[2][3] * b.m[3][3];

    c.m[3][0] = a.m[3][0] * b.m[0][0] + a.m[3][1] * b.m[1][0] + a.m[3][2] * b.m[2][0] + a.m[3][3] * b.m[3][0];
    c.m[3][1] = a.m[3][0] * b.m[0][1] + a.m[3][1] * b.m[1][1] + a.m[3][2] * b.m[2][1] + a.m[3][3] * b.m[3][1];
    c.m[3][2] = a.m[3][0] * b.m[0][2] + a.m[3][1] * b.m[1][2] + a.m[3][2] * b.m[2][2] + a.m[3][3] * b.m[3][2];
    c.m[3][3] = a.m[3][0] * b.m[0][3] + a.m[3][1] * b.m[1][3] + a.m[3][2] * b.m[2][3] + a.m[3][3] * b.m[3][3];

    return c;
}

// [4, 4] x [4, 1] = [4, 1]
static __inline__ __host__ __device__ float4 operator* (const float4& a, const float4x4& m)
{
    return make_float4(
        a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0] + a.w * m.m[3][0],
        a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1] + a.w * m.m[3][1],
        a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2] + a.w * m.m[3][2],
        a.x * m.m[0][3] + a.y * m.m[1][3] + a.z * m.m[2][3] + a.w * m.m[3][3]
    );
}

static __inline__ __host__ __device__ float4 operator* (const float4x4& m, const float4& a)
{
    return make_float4(
        a.x * m.m[0][0] + a.y * m.m[0][1] + a.z * m.m[0][2] + a.w * m.m[0][3],
        a.x * m.m[1][0] + a.y * m.m[1][1] + a.z * m.m[1][2] + a.w * m.m[1][3],
        a.x * m.m[2][0] + a.y * m.m[2][1] + a.z * m.m[2][2] + a.w * m.m[2][3],
        a.x * m.m[3][0] + a.y * m.m[3][1] + a.z * m.m[3][2] + a.w * m.m[3][3]
    );
}

// [3, 4] x [4, 1] = [3, 1]
static __inline__ __host__ __device__ float3 operator*(const float3& a, const float4x4& m) {
    return make_float3(
        a.x * m.m[0][0] + a.y * m.m[1][0] + a.z * m.m[2][0] + m.m[3][0],
        a.x * m.m[0][1] + a.y * m.m[1][1] + a.z * m.m[2][1] + m.m[3][1],
        a.x * m.m[0][2] + a.y * m.m[1][2] + a.z * m.m[2][2] + m.m[3][2]
    );
}

// [3, 4] x [4, 4] = [3, 4]
static __inline__ __host__ __device__ float3x4 operator* (const float3x4& a, const float4x4& b)
{
    float3x4 c;

    c.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0] + a.m[0][3] * b.m[3][0];
    c.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1] + a.m[0][3] * b.m[3][1];
    c.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2] + a.m[0][3] * b.m[3][2];
    c.m[0][3] = a.m[0][0] * b.m[0][3] + a.m[0][1] * b.m[1][3] + a.m[0][2] * b.m[2][3] + a.m[0][3] * b.m[3][3];

    c.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0] + a.m[1][3] * b.m[3][0];
    c.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1] + a.m[1][3] * b.m[3][1];
    c.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2] + a.m[1][3] * b.m[3][2];
    c.m[1][3] = a.m[1][0] * b.m[0][3] + a.m[1][1] * b.m[1][3] + a.m[1][2] * b.m[2][3] + a.m[1][3] * b.m[3][3];

    c.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0] + a.m[2][3] * b.m[3][0];
    c.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1] + a.m[2][3] * b.m[3][1];
    c.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2] + a.m[2][3] * b.m[3][2];
    c.m[2][3] = a.m[2][0] * b.m[0][3] + a.m[2][1] * b.m[1][3] + a.m[2][2] * b.m[2][3] + a.m[2][3] * b.m[3][3];

    return c;
}

namespace maths
{
    static __inline__ __host__ __device__ float4x4 transpose(const float4x4& a) {
        float4x4 b;
        b.m[0][0] = a.m[0][0]; b.m[0][1] = a.m[1][0]; b.m[0][2] = a.m[2][0]; b.m[0][3] = a.m[3][0];
        b.m[1][0] = a.m[0][1]; b.m[1][1] = a.m[1][1]; b.m[1][2] = a.m[2][1]; b.m[1][3] = a.m[3][1];
        b.m[2][0] = a.m[0][2]; b.m[2][1] = a.m[1][2]; b.m[2][2] = a.m[2][2]; b.m[2][3] = a.m[3][2];
        b.m[3][0] = a.m[0][3]; b.m[3][1] = a.m[1][3]; b.m[3][2] = a.m[2][3]; b.m[3][3] = a.m[3][3];
        return b;
    }

    static inline __host__ __device__ void copy(float4x4 &a, float4x4 b) {
        a.m[0][0] = b.m[0][0]; a.m[0][1] = b.m[0][1]; a.m[0][2] = b.m[0][2]; a.m[0][3] = b.m[0][3];
        a.m[1][0] = b.m[1][0]; a.m[1][1] = b.m[1][1]; a.m[1][2] = b.m[1][2]; a.m[1][3] = b.m[1][3];
        a.m[2][0] = b.m[2][0]; a.m[2][1] = b.m[2][1]; a.m[2][2] = b.m[2][2]; a.m[2][3] = b.m[2][3];
        a.m[3][0] = b.m[3][0]; a.m[3][1] = b.m[3][1]; a.m[3][2] = b.m[3][2]; a.m[3][3] = b.m[3][3]; 
    }
}

#endif // F4x4_H