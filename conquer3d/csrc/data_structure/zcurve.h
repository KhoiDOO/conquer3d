#ifndef ZCURVE_H
#define ZCURVE_H

#include "../maths/maths.h"
#include "cuda_runtime.h"

namespace zcurve
{
    inline __device__ unsigned int expandBits(unsigned int v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

    inline __device__ unsigned int morton3D(float x, float y, float z)
    {
        x = x * 1024.0f;
        y = y * 1024.0f;
        z = z * 1024.0f;
        x = x < 0.0f ? 0.0f : (x > 1023.0f ? 1023.0f : x);
        y = y < 0.0f ? 0.0f : (y > 1023.0f ? 1023.0f : y);
        z = z < 0.0f ? 0.0f : (z > 1023.0f ? 1023.0f : z);
        unsigned int xx = expandBits((unsigned int)x);
        unsigned int yy = expandBits((unsigned int)y);
        unsigned int zz = expandBits((unsigned int)z);
        return xx * 4 + yy * 2 + zz;
    }

    void compute_zcurve(const float *points, uint32_t num_points, int64_t *codes);
}

#endif // ZCURVE_H