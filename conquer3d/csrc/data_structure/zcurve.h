/**
 * @file zcurve.h
 * @brief 3D Morton code (Lebesgue Z-curve) spatial bit interleaving utilities.
 */

#ifndef ZCURVE_H
#define ZCURVE_H

#include "../maths/maths.h"
#include "cuda_runtime.h"
#include <cstdint>

namespace zcurve
{
    /**
     * @brief Expands a 10-bit integer by inserting 2 zeros between each bit.
     * 
     * @param[in] v Input 10-bit unsigned integer in range [0, 1023].
     * @return 30-bit integer with spaced bits.
     */
    inline __device__ unsigned int expandBits(unsigned int v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

    /**
     * @brief Computes the 30-bit 3D Morton code for normalized coordinates in [0, 1]^3.
     * 
     * @param[in] x Normalized X-coordinate in [0, 1].
     * @param[in] y Normalized Y-coordinate in [0, 1].
     * @param[in] z Normalized Z-coordinate in [0, 1].
     * @return 30-bit interleaved Morton code.
     */
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

    /**
     * @brief Computes Morton codes for a batch of 3D points on GPU.
     * 
     * @param[in]  points     Pointer to (N, 3) float32 coordinates on CUDA.
     * @param[in]  num_points Number of points ($N$).
     * @param[out] codes      Output device buffer of size $N$ for 64-bit Morton codes.
     */
    void compute_zcurve(const float *points, uint32_t num_points, int64_t *codes);
}

#endif // ZCURVE_H