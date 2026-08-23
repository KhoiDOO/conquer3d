/**
 * @file grid.h
 * @brief High-performance voxel grid filtering and active surface extraction from depth frames.
 */

#pragma once
#include <vector>
#include <torch/extension.h>
#include "../maths/maths.h"

namespace grid {
    /**
     * @brief Unprojects a depth map into 3D space and collects intersecting 1D voxel indices.
     * 
     * @param[in]  num_pixels        Total number of pixels in the depth map ($W \times H$).
     * @param[in]  depth_image       Pointer to device array of float32 depth values in meters.
     * @param[in]  c2w               Camera-to-World 4x4 rigid transformation matrix.
     * @param[in]  intrinsics_inv    Inverse 3x3 camera intrinsic matrix.
     * @param[in]  image_width       Image width in pixels ($W$).
     * @param[in]  image_height      Image height in pixels ($H$).
     * @param[in]  grid_min          Lower bounding coordinate float3.
     * @param[in]  grid_max          Upper bounding coordinate float3.
     * @param[in]  res               Grid resolution int3 `(rx, ry, rz)`.
     * @param[out] out_voxel_ids     Output device buffer for linear voxel IDs.
     * @param[out] valid_counter     Atomic counter tracking number of active voxels written.
     * @param[in]  activate_neighbor If true, activates 26-connected adjacent neighbors.
     * @param[in]  trunc_margin      Depth truncation margin in meters.
     */
    void get_active_voxel_ids_from_depth(
        const int num_pixels,
        const float* depth_image,
        const float4x4 c2w,
        const float3x3 intrinsics_inv,
        const int image_width,
        const int image_height,
        const float3 grid_min,
        const float3 grid_max,
        const int3 res,
        int64_t* out_voxel_ids,
        unsigned long long* valid_counter,
        bool activate_neighbor,
        float trunc_margin
    );
    
    /**
     * @brief Filters active voxel IDs to retain only voxels containing mesh vertices inside their 3D cell bounds.
     * 
     * @param[in] active_voxel_ids  (M,) int64 active sparse voxel IDs.
     * @param[in] vertices          (V, 3) float32 mesh vertex coordinates.
     * @param[in] grid_min          Lower 3D bounds float3.
     * @param[in] grid_max          Upper 3D bounds float3.
     * @param[in] res               Grid resolution int3 (rx, ry, rz).
     * @return torch.Tensor: Filtered active voxel IDs containing at least 1 mesh vertex.
     */
    torch::Tensor filter_voxels_containing_vertices(
        const torch::Tensor& active_voxel_ids,
        const torch::Tensor& vertices,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> res
    );

    /**
     * @brief Generates 8 vertex-centered 3D voxel corner coordinates for each mesh vertex on GPU.
     * 
     * @param[in] vertices  (N, 3) float32 mesh vertex tensor on CUDA.
     * @param[in] grid_min  Lower 3D bounds float3.
     * @param[in] grid_max  Upper 3D bounds float3.
     * @param[in] res       Grid resolution int3 (rx, ry, rz).
     * @return std::tuple<torch::Tensor, torch::Tensor>: Tuple of (raw_corners (N*8, 3), spacing_tensor (3,)).
     */
    std::tuple<torch::Tensor, torch::Tensor> create_voxel_cloud_corners(
        const torch::Tensor& vertices,
        std::vector<float> grid_min,
        std::vector<float> grid_max,
        std::vector<int64_t> res
    );
}
