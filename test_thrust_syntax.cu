#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

struct MyAllocator {
    typedef char value_type;
    char* allocate(std::ptrdiff_t size) { return nullptr; }
    void deallocate(char* p, size_t size) {}
};

int main() {
    MyAllocator alloc;
    
    // Test different syntax for thrust execution policy with custom allocator
    // auto policy1 = thrust::cuda::par.allocator(alloc); // This failed for the user
    auto policy2 = thrust::cuda::par(alloc); // Old syntax?
    // thrust::cuda::allocator exists?
    
    return 0;
}
