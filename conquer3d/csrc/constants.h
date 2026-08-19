/**
 * @file constants.h
 * @brief Global hardware execution constants, register limits, and numerical tolerances.
 */

#ifndef CONSTANTS_H
#define CONSTANTS_H

// ==========================================
// CUDA Execution Model
// ==========================================
/**
 * @def NTHREADS
 * @brief Default 1D CUDA thread block dimension (256 threads per block).
 */
#define NTHREADS 256

// ==========================================
// KD-Tree Constants
// ==========================================
/**
 * @def MAX_K
 * @brief Maximum k-NN nearest neighbors supported in thread register allocation.
 */
#define MAX_K 32

// ==========================================
// BVH Traversal Constants
// ==========================================
/**
 * @def BVH_STACK_SIZE
 * @brief Fixed local stack depth for non-recursive GPU BVH traversal (supports up to $2^{64}$ primitives).
 */
#define BVH_STACK_SIZE 64

/**
 * @def BVH_MAX_CAPACITY
 * @brief Maximum global collision hit buffer size.
 */
#define BVH_MAX_CAPACITY 10000000

// ==========================================
// 3D Gaussian Splatting Constants
// ==========================================
/**
 * @def ISO
 * @brief Default Mahalanobis isosurface threshold radius squared ($r^2 \approx 11.345$).
 */
#define ISO 11.345

/**
 * @def TOL
 * @brief Numerical tolerance for ellipsoid bounding box containment tests.
 */
#define TOL 0.125

/**
 * @def EPS
 * @brief General numerical epsilon for geometric orientation and division-by-zero guards.
 */
#define EPS 1e-5f

#endif // CONSTANTS_H