#ifndef CHECK_H
#define CHECK_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

inline bool check_cuda_result(cudaError_t code, const char *file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

#define CHECK_CUDA(code) check_cuda_result((code), __FILE__, __LINE__)

#endif // CHECK_H