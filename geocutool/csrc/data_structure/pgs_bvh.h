#ifndef PGS_BVH_H
#define PGS_BVH_H

#include "bvh.h"
#include "../primitive/pgs.h"

class PGSBVH : public BVH
{
public:
    // Inherit the base constructor (this automatically builds the tree!)
    using BVH::BVH;
    using BVH::query;

    // Add Gaussian-specific queries
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> query_voxel_pair(
        const torch::Tensor &vx_aabb_mins,
        const torch::Tensor &vx_aabb_maxs,
        const torch::Tensor &means,
        const torch::Tensor &normals,
        const torch::Tensor &covis,
        const torch::Tensor &gs_aabb_mins,
        const torch::Tensor &gs_aabb_maxs,
        const std::optional<torch::Tensor> &isos,
        const float iso,
        const bool return_centroids,
        const bool return_centroid_densities,
        const int64_t max_capacity);

    std::tuple<torch::Tensor, torch::Tensor> query_edge(
        const torch::Tensor &edge_starts,
        const torch::Tensor &edge_ends,
        const torch::Tensor &means,
        const torch::Tensor &normals,
        const torch::Tensor &opacities,
        const torch::Tensor &covis,
        const std::optional<torch::Tensor> &isos,
        const float iso);
};

#endif // PGS_BVH_H