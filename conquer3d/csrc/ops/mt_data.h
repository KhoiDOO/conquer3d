#ifndef MT_DATA_H
#define MT_DATA_H

#include <cuda_runtime.h>
#include <stdint.h>

namespace mt {

// 6 edges of a tetrahedron connecting its 4 local vertices (0-3)
__constant__ int tetEdgeConnection[6][2] = {
    {0, 1}, // Edge 0
    {0, 2}, // Edge 1
    {0, 3}, // Edge 2
    {1, 2}, // Edge 3
    {1, 3}, // Edge 4
    {2, 3}  // Edge 5
};

// Active edges bitmask for each of the 16 cases of a tetrahedron
__constant__ int tetEdgeTable[16] = {
    0x00, // 0: None
    0x07, // 1 (0001: v0): edges 0, 1, 2
    0x19, // 2 (0010: v1): edges 0, 3, 4
    0x1E, // 3 (0011: v0, v1): edges 1, 2, 3, 4
    0x2A, // 4 (0100: v2): edges 1, 3, 5
    0x2D, // 5 (0101: v0, v2): edges 0, 2, 3, 5
    0x33, // 6 (0110: v1, v2): edges 0, 1, 4, 5
    0x34, // 7 (0111: v0, v1, v2): edges 2, 4, 5
    0x34, // 8 (1000: v3): edges 2, 4, 5
    0x33, // 9 (1001: v0, v3): edges 0, 1, 4, 5
    0x2D, // 10 (1010: v1, v3): edges 0, 2, 3, 5
    0x2A, // 11 (1011: v0, v1, v3): edges 1, 3, 5
    0x1E, // 12 (1100: v2, v3): edges 1, 2, 3, 4
    0x19, // 13 (1101: v0, v2, v3): edges 0, 3, 4
    0x07, // 14 (1110: v1, v2, v3): edges 0, 1, 2
    0x00  // 15: None
};

// Number of triangles for each of the 16 cases
__constant__ int tetNumTris[16] = {
    0, 1, 1, 2, 1, 2, 2, 1,
    1, 2, 2, 1, 2, 1, 1, 0
};

// Prefix counts of triangle vertices (3 * triangles) for each case
__constant__ int tetTriNumTable[17] = {
    0, 0, 3, 6, 12, 15, 21, 27, 30, 33, 39, 45, 48, 54, 57, 60, 60
};

// Triangle configurations (up to 2 triangles, 6 edge indices, terminated by -1)
__constant__ int tetTriTable[16][7] = {
    {-1, -1, -1, -1, -1, -1, -1}, // 0x00
    { 0,  1,  2, -1, -1, -1, -1}, // 0x01
    { 0,  4,  3, -1, -1, -1, -1}, // 0x02
    { 2,  1,  4,  4,  3,  1, -1}, // 0x03
    { 1,  3,  5, -1, -1, -1, -1}, // 0x04
    { 0,  5,  2,  0,  3,  5, -1}, // 0x05
    { 0,  4,  5,  0,  1,  5, -1}, // 0x06
    { 2,  5,  4, -1, -1, -1, -1}, // 0x07
    { 2,  4,  5, -1, -1, -1, -1}, // 0x08
    { 0,  5,  4,  0,  1,  5, -1}, // 0x09
    { 0,  2,  5,  0,  5,  3, -1}, // 0x0A
    { 1,  5,  3, -1, -1, -1, -1}, // 0x0B
    { 2,  4,  1,  4,  1,  3, -1}, // 0x0C
    { 0,  3,  4, -1, -1, -1, -1}, // 0x0D
    { 0,  2,  1, -1, -1, -1, -1}, // 0x0E
    {-1, -1, -1, -1, -1, -1, -1}  // 0x0F
};

} // namespace mt

#endif // MT_DATA_H
