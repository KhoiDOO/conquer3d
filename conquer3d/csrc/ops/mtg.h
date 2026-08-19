/**
 * @file mtg.h
 * @brief CUDA implementation of Marching Tetrahedra Grid (MTG) for structured cubic voxel grids.
 */

#ifndef MTG_H
#define MTG_H

#include <torch/extension.h>
#include "mtg_data.h"
#include "../check.h"
#include "../constants.h"
#include "../primitive/edge.h"

#include <optional>
#include <tuple>
#include <cstdint>

namespace mtg {
    /**
     * @brief Forward pass of Marching Tetrahedra on structured voxel grids on GPU.
     * 
     * Internally decomposes each cubic voxel into 6 tetrahedral sub-cells.
     * 
     * @param[in]  num_voxels           Total number of voxel cells ($M$).
     * @param[in]  grid_vertices        Pointer to (N, 3) float32 coordinates on CUDA.
     * @param[in]  voxels               Pointer to (M, 8) uint32 voxel corner indices.
     * @param[in]  voxel_values         Pointer to (N,) float32 scalar SDF values on vertices.
     * @param[in]  grid_normals         Optional pointer to (N, 3) float32 vertex normals.
     * @param[in]  grid_colors          Optional pointer to (N, 3) float32 vertex colors.
     * @param[in]  iso                  Isosurface extraction threshold.
     * @param[in]  vert_options         PyTorch tensor allocation options for output vertices.
     * @param[in]  tri_options          PyTorch tensor allocation options for output triangle indices.
     * @param[in]  return_unique_edges  If true, tracks and returns unique edge index pairs for backward pass.
     * 
     * @return Tuple containing (vertices, triangles, out_normals, out_colors, unique_edges).
     */
    std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>> marching_tetrahedra_grid(
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
     * @brief Analytical backward pass of Marching Tetrahedra Grid on GPU.
     * 
     * @param[in]  n_verts          Number of extracted surface vertices ($V$).
     * @param[in]  unique_edges     Pointer to (V, 2) endpoint edge indices.
     * @param[in]  grid_vertices    Pointer to (N, 3) vertex coordinates.
     * @param[in]  grid_colors      Optional pointer to (N, 3) vertex colors.
     * @param[in]  values           Pointer to (N,) scalar SDF values.
     * @param[in]  adj_verts        Upstream adjoint gradient w.r.t. surface vertices (V, 3).
     * @param[in]  adj_colors       Optional upstream adjoint gradient w.r.t. surface colors (V, 3).
     * @param[out] adj_values       Output gradient buffer for scalar values $\partial L / \partial sdfs$ of size $N$.
     * @param[out] adj_grid_colors  Optional output gradient buffer for grid colors of size $N \times 3$.
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
}

#endif // MTG_H