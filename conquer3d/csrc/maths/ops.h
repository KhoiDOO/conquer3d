#ifndef OPS_H
#define OPS_H

#include <stdint.h>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
/**
 * @file ops.h
 * @brief Host fallbacks for CUDA math intrinsics, and scalar interpolation helpers.
 *
 * @details `rsqrt` and `rsqrtf` are device intrinsics, so host-only translation units get
 * the fallbacks below and shared headers compile either way without `#ifdef` at every call
 * site. The scalar helpers overload the same names as their `float3` counterparts in
 * f3x1.h, and live here because this header is included first and must not depend on the
 * vector operators.
 */

#include <math_constants.h>

#ifndef __CUDACC__
/**
 * @brief Reciprocal square root, double precision (host fallback).
 * @param[in] a Value whose reciprocal square root is taken; must be positive.
 * @return The value $1 / \sqrt{a}$.
 * @note Defined only when not compiling with nvcc, which supplies the intrinsic.
 */
static inline __host__ __device__ double rsqrt(double a) {
    return 1. / sqrt(a);
}

/**
 * @brief Reciprocal square root, single precision (host fallback).
 * @param[in] a Value whose reciprocal square root is taken; must be positive.
 * @return The value $1 / \sqrt{a}$.
 * @note Defined only when not compiling with nvcc. The device intrinsic is an
 * approximation, so host and device results may differ in the last bits.
 */
static inline __host__ __device__ float rsqrtf(float a) {
    return 1. / sqrtf(a);
}
#endif

namespace maths
{
    /**
     * @brief Confines a scalar to a range.
     * @param[in] v Value to clamp.
     * @param[in] min_val Lower bound.
     * @param[in] max_val Upper bound.
     * @return @p v confined to $[\text{min\_val}, \text{max\_val}]$.
     * @note Ordered `fmaxf(lo, fminf(hi, v))`. Irrelevant for finite input, but it decides
     * which bound a NaN collapses to, so the order is deliberate.
     */
    static inline __host__ __device__ float clamp(float v, float min_val, float max_val) {
        return fmaxf(min_val, fminf(max_val, v));
    }

    /**
     * @brief Confines a scalar to the unit interval.
     * @details The common case of clamp(), for interpolation parameters and normalised
     * cell coordinates that must not escape $[0, 1]$ through rounding.
     * @param[in] v Value to clamp.
     * @return @p v confined to $[0, 1]$.
     */
    static inline __host__ __device__ float saturate(float v) {
        return fmaxf(0.0f, fminf(1.0f, v));
    }

    /**
     * @brief Linear interpolation between two scalars.
     * @param[in] a Value returned at $t = 0$.
     * @param[in] b Value returned at $t = 1$.
     * @param[in] t Interpolation parameter; not clamped.
     * @return $a + t\,(b - a)$.
     */
    static inline __host__ __device__ float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }
}

#endif // OPS_H