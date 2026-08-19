/**
 * @file pgs_bvh.h
 * @brief High-performance Bounding Volume Hierarchy for Periodic Gaussian Splatting (PGS).
 */

#ifndef PGS_BVH_H
#define PGS_BVH_H

#include "bvh.h"
#include "../primitive/pgs.h"

#include <variant>
#include <tuple>
#include <optional>

/**
 * @brief Bounding Volume Hierarchy specialized for Periodic Gaussians and directional radiance fields.
 */
class PGSBVH : public BVH
{
public:
    using BVH::BVH;
    using BVH::query;

    /**
     * @brief Queries intersecting Periodic Gaussian-voxel pairs against voxel bounding boxes.
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> query_voxel_pair(
        const torch::Tensor &vx_aabb_mins,
        const torch::Tensor &vx_aabb_maxs,
        const torch::Tensor &means,
        const torch::Tensor &normals,
        const torch::Tensor &covis,
        const torch::Tensor &gs_aabb_mins,
        const torch::Tensor &gs_aabb_maxs,
        const std::variant<float, torch::Tensor> &isos,
        const bool return_centroids,
        const bool return_centroid_densities,
        const int64_t max_capacity);

    /**
     * @brief Queries Periodic Gaussian intersections and directional densities along line segments.
     */
    std::tuple<torch::Tensor, torch::Tensor> query_edge(
        const torch::Tensor &edge_starts,
        const torch::Tensor &edge_ends,
        const torch::Tensor &means,
        const torch::Tensor &normals,
        const torch::Tensor &opacities,
        const torch::Tensor &covis,
        const std::variant<float, torch::Tensor> &isos);
};

#endif // PGS_BVH_H