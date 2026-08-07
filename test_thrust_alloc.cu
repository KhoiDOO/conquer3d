#include <ATen/cuda/ThrustAllocator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

int main() {
    at::cuda::ThrustAllocator allocator;
    // How to pass allocator to thrust::cuda::par?
    // Try thrust::cuda::par.with(thrust::cuda::allocator(allocator)) or something
    auto policy1 = thrust::cuda::par(allocator);
    return 0;
}
