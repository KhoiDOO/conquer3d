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

    using BVH::query_ray;

    // Returns a [N, 2] tensor of intersecting triangle index pairs
    torch::Tensor get_self_intersection(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    bool is_self_intersection(
        const torch::Tensor &vertices,
        const torch::Tensor &triangles);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_ray_intersection(
        const torch::Tensor &ray_origins,
        const torch::Tensor &ray_dirs,
        const torch::Tensor &vertices,
        const torch::Tensor &triangles,
        bool return_distance = false);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> query_point(
        const torch::Tensor &query_points,
        const torch::Tensor &vertices,
        const torch::Tensor &triangles,
        bool return_sdf = false,
        bool return_prj_pts = true,
        int sign_mode = 0);
};

#endif // MESH_BVH_H
