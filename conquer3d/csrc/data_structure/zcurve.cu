#include "zcurve.h"
#include "../constants.h"
#include <cuda_runtime.h>

namespace zcurve
{

    __global__ void compute_zcurve_kernel(const float3 *__restrict__ points, uint32_t num_points, int64_t *__restrict__ codes)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points)
            return;

        float3 p = points[idx];
        unsigned int code = morton3D(p.x, p.y, p.z);
        codes[idx] = (int64_t)code;
    }

    void compute_zcurve(const float *points, uint32_t num_points, int64_t *codes)
    {
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_points + threads - 1) / threads;

        compute_zcurve_kernel<<<blocks, threads>>>((const float3 *)points, num_points, codes);
    }

}
