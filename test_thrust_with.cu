#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <iostream>

struct MyAllocator {
    typedef char value_type;
    char* allocate(std::ptrdiff_t size) { return nullptr; }
    void deallocate(char* p, size_t size) {}
};

int main() {
    MyAllocator alloc;
    auto policy = thrust::cuda::par(alloc);
    return 0;
}
