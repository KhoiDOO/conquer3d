#ifndef MCA_DATA_H
#define MCA_DATA_H

#include <cuda_runtime.h>
#include <stdint.h>

namespace mca {

// 6 Faces in conquer3d CCW outward order:
// Face 0: -Z (Bottom), Face 1: +Z (Top), Face 2: -Y (Front),
// Face 3: +Y (Back), Face 4: -X (Left), Face 5: +X (Right)
static __constant__ int v_face[6][4] = {
    {0, 3, 2, 1}, // -Z
    {4, 5, 6, 7}, // +Z
    {0, 1, 5, 4}, // -Y
    {3, 7, 6, 2}, // +Y
    {0, 4, 7, 3}, // -X
    {1, 2, 6, 5}  // +X
};

static __constant__ int e_face[6][4] = {
    {3, 2, 1, 0}, // -Z
    {4, 5, 6, 7}, // +Z
    {0, 9, 4, 8}, // -Y
    {11, 6, 10, 2}, // +Y
    {8, 7, 11, 3}, // -X
    {1, 10, 5, 9}  // +X
};

// Bitmask of which faces are ambiguous for each of the 256 cube cases.
// Bit f (0..5) is 1 if face f has diagonally opposite corners with same sign.
static __constant__ uint8_t cubeFaceAmbigMask[256] = {
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 4, 0, 0, 1, 4, 0, 16, 0, 21, 0, 16, 0, 20, 0,
    0, 4, 0, 0, 32, 37, 0, 0, 0, 4, 1, 0, 32, 36, 0, 0,
    0, 0, 0, 0, 32, 33, 0, 0, 16, 0, 17, 0, 48, 32, 16, 0,
    0, 0, 32, 32, 0, 1, 0, 0, 8, 8, 41, 40, 0, 0, 0, 0,
    2, 2, 38, 34, 2, 3, 6, 2, 26, 10, 63, 42, 18, 2, 22, 2,
    0, 4, 0, 0, 0, 5, 0, 0, 8, 12, 9, 8, 0, 4, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 24, 8, 25, 8, 16, 0, 16, 0,
    0, 16, 0, 16, 8, 25, 8, 24, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 4, 0, 8, 9, 12, 8, 0, 0, 5, 0, 0, 0, 4, 0,
    2, 22, 2, 18, 42, 63, 10, 26, 2, 6, 3, 2, 34, 38, 2, 2,
    0, 0, 0, 0, 40, 41, 8, 8, 0, 0, 1, 0, 32, 32, 0, 0,
    0, 16, 32, 48, 0, 17, 0, 16, 0, 0, 33, 32, 0, 0, 0, 0,
    0, 0, 36, 32, 0, 1, 4, 0, 0, 0, 37, 32, 0, 0, 4, 0,
    0, 20, 0, 16, 0, 21, 0, 16, 0, 4, 1, 0, 0, 4, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
};

} // namespace mca

#endif // MCA_DATA_H
