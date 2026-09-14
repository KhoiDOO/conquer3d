/**
 * @file flood_fill_band.h
 * @brief Leak-resistant band flood fill on the coarse-fine layout, the backend of `sign_mode` 6.
 *
 * @details The segment-test fills (modes 3 and 5) let water through any gap a lattice segment can
 * thread, so a single small hole floods a whole interior. Following the data preparation of
 * AssetGen (arXiv 2605.26137, section 5.1), this fill instead builds a wall of lattice vertices:
 * the corners of every cell a triangle touches, dilated by a Chebyshev radius so that holes
 * narrower than the dilation are sealed. Water floods from the grid boundary and stops at the wall,
 * small cavities the dilation created are released, and the wall itself is resolved by
 * neighbour agreement. Storage follows mode 5 -- one label per coarse block plus fine labels only
 * for blocks near the surface -- so memory scales with surface area rather than volume, and the
 * result is read by the same sign lookup.
 */

#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ops
{

    /**
     * @brief Result container for the band flood fill.
     * @details Shares mode 5's layout so one device lookup serves both modes: a coarse label per
     * block, a dense lookup from coarse index to band-block slot, and fine labels for the band
     * blocks only.
     */
    struct BandFloodFillResult
    {
        torch::Tensor coarse_mask;       ///< (Cx, Cy, Cz) int8 labels: 2 exterior, -1 interior, 1 band block.
        torch::Tensor band_block_coords; ///< (N, 3) int32 coarse coordinates of the band blocks.
        torch::Tensor band_block_lookup; ///< (Cx, Cy, Cz) int32 slot of each band block, -1 elsewhere.
        torch::Tensor fine_masks;        ///< (N, 8, 8, 8) int8 vertex labels in band blocks: 2 exterior, -1 interior.
        std::vector<int64_t> block_size; ///< Fine vertices per coarse block, always [8, 8, 8].
        std::vector<int64_t> coarse_res; ///< Coarse grid resolution, [Cx, Cy, Cz] = ceil(res / 8).
        std::map<std::string, int64_t> stats; ///< Build diagnostics, keyed by name.
    };

    /**
     * @brief Computes the leak-resistant band flood fill on GPU.
     * @details The lattice has `grid_res` vertices per axis spanning `[grid_min, grid_max]`, so vertex
     * `i` sits at `grid_min + i * (grid_max - grid_min) / (grid_res - 1)`, the convention every flood
     * fill mode shares. The pipeline classifies coarse blocks, marks the surface band `S0` and the
     * dilated band `D` inside band blocks, floods the exterior from the grid boundary with `D` as a
     * wall, releases interior components of at most @p cavity_max_voxels vertices, and resolves the
     * band by neighbour agreement in three layers (`D \ S0` with `S0` as a barrier, then `S0`, then
     * any remainder). Every vertex ends labelled exterior or interior.
     * @param[in] vertices (V, 3) float32 mesh vertices.
     * @param[in] triangles (F, 3) int32 triangle indices.
     * @param[in] aabb_mins (2F-1, 3) float32 BVH node lower bounds.
     * @param[in] aabb_maxs (2F-1, 3) float32 BVH node upper bounds.
     * @param[in] bvh_children (2F-1, 2) int32 BVH child indices.
     * @param[in] object_ids (F,) int32 leaf-to-triangle map.
     * @param[in] grid_min Lower grid corner [x, y, z].
     * @param[in] grid_max Upper grid corner [x, y, z].
     * @param[in] grid_res Vertices per axis [rx, ry, rz].
     * @param[in] dilation_radius Chebyshev dilation radius `r` in lattice spacings; holes up to about
     *     `2r + 3` spacings across are sealed.
     * @param[in] cavity_max_voxels Largest interior component, in vertices, released back to the band
     *     for resolution; -1 selects `(2r + 1)^3` and 0 disables the release.
     * @return BandFloodFillResult holding the labels and the diagnostics `num_band_blocks`,
     *     `flood_rounds`, `num_released_cavity_voxels`, `resolution_rounds`,
     *     `num_unresolved_defaulted` and `bvh_stack_overflows`.
     * @warning The wall trades thin detail for leak resistance. Solid parts thinner than about
     *     `2r + 3` spacings resolve exterior and vanish, slots narrower than 3 spacings fill in, and a
     *     pocket the dilation seals off that holds more than @p cavity_max_voxels free vertices stays
     *     interior. Raise the resolution or lower @p dilation_radius to keep such features.
     * @throws std::runtime_error If the mesh lies closer than `dilation_radius + 2` spacings to the
     *     grid boundary, where the band would touch the seeds and water could not reach around it;
     *     the message names the padding required.
     */
    BandFloodFillResult compute_flood_fill_band(const torch::Tensor &vertices, const torch::Tensor &triangles,
                                                const torch::Tensor &aabb_mins, const torch::Tensor &aabb_maxs,
                                                const torch::Tensor &bvh_children, const torch::Tensor &object_ids,
                                                std::vector<float> grid_min, std::vector<float> grid_max,
                                                std::vector<int64_t> grid_res, int dilation_radius = 2,
                                                int64_t cavity_max_voxels = -1);

} // namespace ops
