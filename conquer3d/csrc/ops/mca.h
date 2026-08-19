/**
 * @file mca.h
 * @brief CUDA implementation of Marching Cubes with Asymptotic Deciders (MCA).
 * 
 * Resolves topological ambiguities on bilinear cell faces using Nielson & Hamann's
 * saddle point evaluation, guaranteeing watertight 2-manifold surface topology.
 */

#ifndef MCA_H
#define MCA_H

#include <torch/extension.h>
#include "mc_data.h"
#include <tuple>
#include <optional>

namespace mca {

/**
 * @brief Forward pass of Marching Cubes with Asymptotic Deciders.
 * 
 * @param[in]  grid_vertices  (N, 3) float32 coordinates of grid vertices on CUDA.
 * @param[in]  voxels         (M, 8) int32 corner indices per voxel cell on CUDA.
 * @param[in]  sdf            (N,) float32 scalar SDF values on grid vertices.
 * @param[in]  colors         Optional (N, C) float32 vertex color/feature tensor.
 * @param[in]  iso            Isosurface threshold value.
 * 
 * @return std::tuple containing:
 *         - vertices (torch.Tensor): Extracted (V, 3) surface vertices.
 *         - faces (torch.Tensor): Extracted (F, 3) triangle face indices.
 *         - out_colors (std::optional<torch.Tensor>): (V, C) interpolated colors.
 * 
 * @note All tensors must be contiguous and reside on the same CUDA device.
 */
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> marching_cubes_asymptotic(
    const torch::Tensor &grid_vertices,
    const torch::Tensor &voxels,
    const torch::Tensor &sdf,
    const std::optional<torch::Tensor> &colors,
    float iso
);

/**
 * @brief Backward pass of Marching Cubes with Asymptotic Deciders.
 * 
 * @param[in]  grad_vertices  Upstream gradient w.r.t. extracted vertices (V, 3).
 * @param[in]  grad_colors    Optional upstream gradient w.r.t. extracted colors.
 * @param[in]  grid_vertices  (N, 3) float32 grid corner coordinates.
 * @param[in]  unique_edges   (V, 2) int64 corner index pairs for extracted surface vertices.
 * @param[in]  sdf            (N,) float32 scalar SDF field.
 * @param[in]  colors         Optional (N, C) float32 color features.
 * @param[in]  iso            Isosurface extraction threshold.
 * 
 * @return std::tuple containing (grad_sdf, grad_colors).
 */
std::tuple<torch::Tensor, std::optional<torch::Tensor>> marching_cubes_asymptotic_backward(
    const torch::Tensor &grad_vertices,
    const std::optional<torch::Tensor> &grad_colors,
    const torch::Tensor &grid_vertices,
    const torch::Tensor &unique_edges,
    const torch::Tensor &sdf,
    const std::optional<torch::Tensor> &colors,
    float iso
);

} // namespace mca

#endif // MCA_H
