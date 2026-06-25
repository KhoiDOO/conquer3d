#ifndef MESH_BVH_H
#define MESH_BVH_H

#include "bvh.h"
#include <torch/extension.h>

class MeshBVH : public BVH
{
public:
    using BVH::BVH;
    using BVH::query;
    using BVH::query_self;

    // Returns a [N, 2] tensor of intersecting triangle index pairs
    torch::Tensor get_self_intersection(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    bool is_self_intersection(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);
};

#endif // MESH_BVH_H
