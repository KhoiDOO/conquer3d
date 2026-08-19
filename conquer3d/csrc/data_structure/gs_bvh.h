/**
 * @file gs_bvh.h
 * @brief High-performance Bounding Volume Hierarchy for 3D Gaussian Splatting (3DGS).
 */

#ifndef GS_BVH_H
#define GS_BVH_H

#include "bvh.h"
#include "../primitive/gs.h"

#include <variant>
#include <tuple>
#include <optional>

/**
 * @brief Bounding Volume Hierarchy specialized for 3D Gaussian primitives and volumetric voxel/edge queries.
 */
class GSBVH : public BVH
{
public:
    using BVH::BVH;
    using BVH::query;

    /**
     * @brief Queries intersecting Gaussian-voxel pairs against voxel bounding boxes.
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> query_voxel_pair(
        const torch::Tensor &vx_aabb_mins,
        const torch::Tensor &vx_aabb_maxs,
        const torch::Tensor &means,
        const torch::Tensor &covis,
        const torch::Tensor &opacities,
        const torch::Tensor &gs_aabb_mins,
        const torch::Tensor &gs_aabb_maxs,
        const torch::Tensor &contact_points,
        const std::variant<float, torch::Tensor> &isos,
        const float ar_threshold,
        const float p_threshold,
        const bool return_centroids,
        const int64_t max_capacity);

    /**
     * @brief Queries 3D Gaussian intersections with line segments / voxel edges.
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> query_edge_pair(
        const torch::Tensor &edge_starts,
        const torch::Tensor &edge_ends,
        const torch::Tensor &means,
        const torch::Tensor &covis,
        const std::variant<float, torch::Tensor> &isos,
        const int64_t max_capacity);

    /**
     * @brief Queries closest Gaussian intersections and scalar density along line segments.
     */
    std::tuple<torch::Tensor, torch::Tensor> query_edge(
        const torch::Tensor &edge_starts,
        const torch::Tensor &edge_ends,
        const torch::Tensor &means,
        const torch::Tensor &opacities,
        const torch::Tensor &covis,
        const std::variant<float, torch::Tensor> &isos);
};

#endif // GS_BVH_H