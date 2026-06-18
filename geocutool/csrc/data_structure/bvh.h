#ifndef BVH_H
#define BVH_H

#include "../maths/maths.h"
#include "../constants.h"
#include "../primitive/aabb.h"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

class BVH
{
protected:
    uint32_t num_objects;
    uint32_t num_nodes; // Will always be 2N - 1

    torch::Tensor aabb_mins; // Size: [2N - 1, 3]
    torch::Tensor aabb_maxs; // Size: [2N - 1, 3]
    torch::Tensor bvh_children; // Size: [2N - 1, 2] -> x = left_child, y = right_child
    torch::Tensor bvh_parents; // Size: [2N - 1] -> index of parent node
    torch::Tensor object_ids; // Size: [N] -> Maps the sorted leaf index back to the original Gaussian index

public:
    BVH(const torch::Tensor &in_aabb_mins, const torch::Tensor &in_aabb_maxs);

    std::tuple<torch::Tensor, torch::Tensor> query(
        const torch::Tensor &query_aabb_mins,
        const torch::Tensor &query_aabb_maxs);
};

namespace bvh
{
    __host__ void build(
        const uint32_t num_objects,
        const uint32_t num_nodes,
        const float3 *__restrict__ in_aabb_mins, // Unsorted input
        const float3 *__restrict__ in_aabb_maxs, // Unsorted input
        float3 *__restrict__ bvh_aabb_mins,      // Size 2N-1
        float3 *__restrict__ bvh_aabb_maxs,      // Size 2N-1
        int2 *__restrict__ bvh_children,         // Size 2N-1
        int *__restrict__ bvh_parents,           // Size 2N-1
        int *__restrict__ object_ids             // Size N
    );

    __host__ void query(
        const uint32_t num_queries,
        const uint32_t num_objects,
        const float3 *__restrict__ query_mins,
        const float3 *__restrict__ query_maxs,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        int64_t *__restrict__ out_query_ids,
        int64_t *__restrict__ out_object_ids,
        int64_t *__restrict__ hit_counter,
        const int64_t max_capacity);
}

#endif // BVH_H