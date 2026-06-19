#ifndef CHECK_H
#define CHECK_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Helper macros to check tensor properties
// These require <torch/extension.h> or <ATen/ATen.h> to be included in the translation unit
#define CHECK_CUDA(x) \
    TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

inline bool check_cuda_result(cudaError_t code, const char *file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

#define CUDA_CHECK(code) check_cuda_result((code), __FILE__, __LINE__)

#endif // CHECK_H