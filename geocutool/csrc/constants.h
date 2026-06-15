#ifndef CONSTANTS_H
#define CONSTANTS_H

// ==========================================
// KD-Tree Hardware Limits
// ==========================================
// Defines the maximum number of neighbors the Priority Queue can hold.
// 32 is highly recommended to prevent CUDA register spilling.
#define MAX_K 32

// You can add other global math/physics constants here in the future!
// #define PGS_EPSILON 1e-6f

#endif // CONSTANTS_H