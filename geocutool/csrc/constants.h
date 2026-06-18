#ifndef CONSTANTS_H
#define CONSTANTS_H

// ==========================================
// Number of Threads per Block
// ==========================================
#define NTHREADS 256

// ==========================================
// KD-Tree Hardware Limits
// ==========================================
// Defines the maximum number of neighbors the Priority Queue can hold.
// 32 is highly recommended to prevent CUDA register spilling.
#define MAX_K 32

// ==========================================
// BVH Stack Size
// ==========================================
#define BVH_STACK_SIZE 64
// ==========================================
// BVH Max Capacity
// ==========================================
#define BVH_MAX_CAPACITY 10000000


#endif // CONSTANTS_H