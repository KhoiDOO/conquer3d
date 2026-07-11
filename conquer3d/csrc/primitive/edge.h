#ifndef EDGE_H
#define EDGE_H

#include <cuda_runtime.h>
#include <cstdint>
#include "../maths/f3x1.h"

struct Edge {
    uint32_t v0;
    uint32_t v1;

    __host__ __device__ Edge() : v0(0), v1(0) {}
    __host__ __device__ Edge(uint32_t a, uint32_t b) {
        v0 = a < b ? a : b;
        v1 = a > b ? a : b;
    }

    __host__ __device__ bool operator==(const Edge& other) const {
        return v0 == other.v0 && v1 == other.v1;
    }

    __host__ __device__ bool operator!=(const Edge& other) const {
        return v0 != other.v0 || v1 != other.v1;
    }

    __host__ __device__ bool operator<(const Edge& other) const {
        if (v0 != other.v0) return v0 < other.v0;
        return v1 < other.v1;
    }

    __host__ __device__ float3 compute_midpoint(const float3* vertices) const {
        float3 p0 = vertices[v0];
        float3 p1 = vertices[v1];
        return (p0 + p1) * 0.5f;
    }
};

#endif // EDGE_H
