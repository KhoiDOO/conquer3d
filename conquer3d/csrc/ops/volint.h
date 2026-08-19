/**
 * @file volint.h
 * @brief GPU TSDF Volume Integration for RGB-D camera streams.
 */

#ifndef VOLINT_H
#define VOLINT_H

#include "../maths/maths.h"
#include <cuda_runtime.h>

/**
 * @brief Integrates a single RGB-D depth and color frame into a 3D volumetric TSDF grid.
 * 
 * Projects each 3D grid vertex into camera image space using extrinsics and intrinsics,
 * computes the signed distance difference to the measured depth, and updates the running
 * weighted average TSDF and RGB values in-place.
 * 
 * @param[in]     num_vertices  Number of grid vertices ($N$).
 * @param[in]     grid_vertices Pointer to (N, 3) float32 coordinates in world space.
 * @param[in,out] sdf           Pointer to (N,) float32 TSDF values updated in-place.
 * @param[in,out] weight        Pointer to (N,) float32 running average weights updated in-place.
 * @param[in,out] color         Optional pointer to (N, 3) float32 RGB colors updated in-place.
 * @param[in]     depth_image   Pointer to (H, W) float32 depth map in meters.
 * @param[in]     color_image   Optional pointer to (H, W, 3) float32 RGB color image.
 * @param[in]     image_width   Width of the depth/color image in pixels ($W$).
 * @param[in]     image_height  Height of the depth/color image in pixels ($H$).
 * @param[in]     extrinsics    4x4 World-to-Camera extrinsic transformation matrix.
 * @param[in]     intrinsics    3x3 Camera intrinsic matrix.
 * @param[in]     trunc_margin  Truncation threshold $\mu$ in meters.
 * @param[in]     mode          Integration mode (1 for true Euclidean distance, 0 for projective distance).
 */
void single_view_volume_integral(
    const int num_vertices,
    const float3* grid_vertices,
    float* sdf,
    float* weight,
    float3* color,
    const float* depth_image,
    const float3* color_image,
    const int image_width,
    const int image_height,
    const float4x4 extrinsics,
    const float3x3 intrinsics,
    const float trunc_margin,
    const int mode
);

#endif // VOLINT_H