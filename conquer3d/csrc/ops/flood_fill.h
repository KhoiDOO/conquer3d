/**
 * @file flood_fill.h
 * @brief GPU 3D Volumetric Flood-Fill for robust inside/outside topological sign determination.
 */

#ifndef FLOOD_FILL_H
#define FLOOD_FILL_H

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace ops {
    /**
     * @brief Computes a 3D volumetric binary flood-fill occupancy mask on GPU.
     * 
     * Identifies boundary-intersecting voxels using BVH collision queries, then performs
     * parallel frontier-expansion flood fill from grid boundaries to segment exterior vs interior space.
     * 
     * @param[in] vertices      (V, 3) float32 mesh vertex tensor.
     * @param[in] triangles     (F, 3) int32 triangle index tensor.
     * @param[in] aabb_mins     (2F-1, 3) BVH lower box coordinates.
     * @param[in] aabb_maxs     (2F-1, 3) BVH upper box coordinates.
     * @param[in] bvh_children  (2F-1, 2) BVH child node indices.
     * @param[in] object_ids    (F,) leaf-to-triangle map.
     * @param[in] grid_min      3D lower coordinate bounds `[x_min, y_min, z_min]`.
     * @param[in] grid_max      3D upper coordinate bounds `[x_max, y_max, z_max]`.
     * @param[in] grid_res      3D grid resolution `[rx, ry, rz]`.
     * @param[in] connectivity  Voxel neighbor connectivity (6, 18, or 26). Defaults to 6.
     * 
     * @return torch.Tensor: 3D int8 mask of shape `(rx, ry, rz)` (0: exterior, 1: surface, -1: interior).
     */
    torch::Tensor compute_flood_fill(
        const torch::Tensor& vertices,
        const torch::Tensor& triangles,
        const torch::Tensor& aabb_mins,
        const torch::Tensor& aabb_maxs,
        const torch::Tensor& bvh_children,
        const torch::Tensor& object_ids,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> grid_res,
        int connectivity = 6
    );
}

#endif // FLOOD_FILL_H
