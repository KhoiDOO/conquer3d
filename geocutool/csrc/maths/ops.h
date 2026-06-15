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

#endif // OPS_H