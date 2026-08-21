/**
 * @file flood_fill_dilated.h
 * @brief High-performance GPU 2-Level Hierarchical Boundary-Aware Dilated Flood-Fill.
 * 
 * Implements the 3-stage boundary-aware flood-fill algorithm (AssetGen / arXiv:2605.26137)
 * on top of a 2-level coarse-to-fine hierarchy (< 15 MB VRAM at 1024^3).
 * 
 * Features:
 * - Morphological boundary dilation to seal open holes and cracks
 * - Coarse-to-fine wavefront flood fill from outer bounding domain
 * - Cavity pruning for tight concavities
 * - 26-neighbor distance-weighted consensus relabeling to eliminate geometric inflation
 */

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include "flood_fill_cf.h"

namespace ops {

    /**
     * @brief Computes 2-Level Hierarchical Boundary-Aware Dilated Flood Fill on GPU.
     *
     * @param[in] vertices         (V, 3) float32 mesh vertex tensor.
     * @param[in] triangles        (F, 3) int32 triangle index tensor.
     * @param[in] aabb_mins        (2F-1, 3) BVH lower box coordinates.
     * @param[in] aabb_maxs        (2F-1, 3) BVH upper box coordinates.
     * @param[in] bvh_children     (2F-1, 2) BVH child node indices.
     * @param[in] object_ids       (F,) leaf-to-triangle map.
     * @param[in] grid_min         3D lower coordinate bounds [x_min, y_min, z_min].
     * @param[in] grid_max         3D upper coordinate bounds [x_max, y_max, z_max].
     * @param[in] grid_res         3D fine grid resolution [rx, ry, rz].
     * @param[in] dilation_radius  Morphological dilation radius k in voxels (default 1).
     * @param[in] min_cavity_size  Minimum voxel volume for interior cavities (default 64).
     * @param[in] block_size       Optional macro-block size [bx, by, bz]. If empty, computed dynamically.
     * @param[in] connectivity     Neighbor connectivity for consensus voting (6, 18, 26, default 26).
     * @return CFFloodFillResult struct holding coarse mask and fine boundary masks.
     */
    CFFloodFillResult compute_flood_fill_dilated_cf(
        const torch::Tensor& vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& aabb_mins,
        const torch::Tensor& aabb_maxs,
        const torch::Tensor& bvh_children,
        const torch::Tensor& object_ids,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> grid_res,
        int dilation_radius = 1,
        int min_cavity_size = 64,
        std::vector<int64_t> block_size = {},
        int connectivity = 26
    );

} // namespace ops
