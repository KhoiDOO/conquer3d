#ifndef F3x4_H
#define F3x4_H

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>

typedef struct
{
    float m[3][4];
} float3x4;

static __inline__ __host__ __device__ float3 mul3x4(float3x4 m, float4 a) {
    return make_float3(
        a.x * m.m[0][0] + a.y * m.m[0][1] + a.z * m.m[0][2] + a.w * m.m[0][3],
        a.x * m.m[1][0] + a.y * m.m[1][1] + a.z * m.m[1][2] + a.w * m.m[1][3],
        a.x * m.m[2][0] + a.y * m.m[2][1] + a.z * m.m[2][2] + a.w * m.m[2][3]
    );
}

static __inline__ __host__ __device__ float3 operator* (const float3x4 m, const float4 a)
{
    return mul3x4(m, a);
}

#endif // F3x4_H