/**
 * @file mc.h
 * @brief High-performance CUDA implementation of the Marching Cubes algorithm.
 * 
 * Provides forward isosurface extraction and analytical backward differentiation
 * w.r.t. grid vertex scalar values and vertex color features.
 */

#ifndef MC_H
#define MC_H

#include <torch/extension.h>
#include "mc_data.h"
#include "../maths/maths.h"
#include "../check.h"
#include "../constants.h"
#include "../primitive/edge.h"

#include <optional>

namespace mc {

/**
 * @brief Extracts an isosurface mesh from a voxel grid using CUDA Marching Cubes.
 * 
 * @param[in]  num_voxels          Number of voxel cells in the grid.
 * @param[in]  grid_vertices       Pointer to device array of (V, 3) float32 coordinates.
 * @param[in]  voxels              Pointer to device array of (M, 8) uint32 corner indices.
 * @param[in]  voxel_values        Pointer to device array of (V,) float32 scalar values.
 * @param[in]  grid_normals        Optional pointer to device array of (V, 3) vertex normals.
 * @param[in]  grid_colors         Optional pointer to device array of (V, 3) vertex colors.
 * @param[in]  iso                 Isosurface threshold value.
 * @param[in]  vert_options        TensorOptions for allocating output vertex tensors.
 * @param[in]  tri_options         TensorOptions for allocating output face tensors.
 * @param[in]  return_unique_edges If true, returns unique active edges for autograd backward.
 * 
 * @return std::tuple containing:
 *         - vertices (torch.Tensor): Extracted mesh vertices (M, 3).
 *         - triangles (torch.Tensor): Extracted triangle face indices (T, 3).
 *         - normals (std::optional<torch.Tensor>): Interpolated vertex normals.
 *         - colors (std::optional<torch.Tensor>): Interpolated vertex colors.
 *         - unique_edges (std::optional<torch.Tensor>): Active intersected edges.
 */
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>> marching_cubes(
    const uint32_t num_voxels,
    const float3* __restrict__ grid_vertices,
    const uint32_t* __restrict__ voxels,
    const float* __restrict__ voxel_values,
    const float3* __restrict__ grid_normals,
    const float3* __restrict__ grid_colors,
    const float iso,
    torch::TensorOptions vert_options,
    torch::TensorOptions tri_options,
    bool return_unique_edges = false
);

/**
 * @brief Analytical backward pass of Marching Cubes w.r.t. scalar values and colors.
 * 
 * @param[in]  n_verts          Number of extracted surface vertices.
 * @param[in]  unique_edges     Pointer to device array of active intersected edge pairs.
 * @param[in]  grid_vertices    Pointer to device array of grid coordinates.
 * @param[in]  grid_colors      Optional pointer to device array of grid colors.
 * @param[in]  values           Pointer to device array of scalar values on grid corners.
 * @param[in]  adj_verts        Upstream adjoint gradients w.r.t. extracted vertices.
 * @param[in]  adj_colors       Optional upstream adjoint gradients w.r.t. extracted colors.
 * @param[out] adj_values       Output device buffer accumulating scalar field gradients.
 * @param[out] adj_grid_colors  Output device buffer accumulating color field gradients.
 * @param[in]  iso              Isosurface extraction threshold.
 */
void backward(
    const uint32_t n_verts,
    const Edge *unique_edges,
    const float3 *grid_vertices,
    const float3 *grid_colors,
    const float *values,
    const float3 *adj_verts,
    const float3 *adj_colors,
    float *adj_values,
    float3 *adj_grid_colors,
    const float iso
);

} // namespace mc

#endif // MC_H