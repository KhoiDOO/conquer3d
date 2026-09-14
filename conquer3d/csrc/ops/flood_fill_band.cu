/**
 * @file flood_fill_band.cu
 * @brief CUDA kernels and host pipeline for the leak-resistant band flood fill (`sign_mode` 6).
 *
 * @details All geometry is mapped to lattice units, `q = (p - grid_min) / spacing`, before any test
 * runs, so vertex `i` sits at the integer `i` exactly and the only tolerance is a few float ULPs of
 * the largest coordinate. The result is independent of where the grid sits in world space and of
 * its scale, which the segment tests of modes 3 and 5 are not.
 *
 * Label codes, shared by the coarse and fine arrays:
 *   - `-2` unknown (free and not yet reached by water),
 *   - `3` surface band `S0` (fine only),
 *   - `1` dilated band `D \ S0` on fine vertices, or a band block on the coarse grid,
 *   - `2` exterior, `-1` interior,
 *   - `5` released cavity awaiting commit, `4` / `-4` exterior / interior awaiting commit.
 */

#include "flood_fill_band.h"
#include "flood_fill_common.cuh"
#include "../data_structure/bvh_traverse.cuh"
#include "../constants.h"
#include "../primitive/triangle.h"
#include "../maths/maths.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <sstream>
#include <stdexcept>

namespace ops
{
    namespace band_fill
    {
        constexpr int B = 8;    ///< Fine vertices per coarse block along each axis.
        constexpr int B3 = 512; ///< Fine vertices per coarse block, one full CUDA block of threads.

        constexpr int8_t UNKNOWN = -2;          ///< Free vertex or block not yet reached by water.
        constexpr int8_t INTERIOR = -1;         ///< Resolved interior.
        constexpr int8_t DILATED = 1;           ///< Fine vertex in the dilated band `D \ S0`.
        constexpr int8_t BAND_BLOCK = 1;        ///< Coarse block holding fine labels.
        constexpr int8_t EXTERIOR = 2;          ///< Resolved exterior.
        constexpr int8_t SURFACE = 3;           ///< Fine vertex in the surface band `S0`.
        constexpr int8_t PENDING_EXTERIOR = 4;  ///< Exterior vote cast this resolution round.
        constexpr int8_t PENDING_INTERIOR = -4; ///< Interior vote cast this resolution round.
        constexpr int8_t RELEASED = 5;          ///< Interior vertex of a released cavity, before commit.

        /**
         * @brief Fine and coarse extents of the lattice, passed to every kernel.
         */
        struct BandGrid
        {
            int rx; ///< Fine vertices along x.
            int ry; ///< Fine vertices along y.
            int rz; ///< Fine vertices along z.
            int cx; ///< Coarse blocks along x, `ceil(rx / 8)`.
            int cy; ///< Coarse blocks along y, `ceil(ry / 8)`.
            int cz; ///< Coarse blocks along z, `ceil(rz / 8)`.
        };

        /**
         * @brief Smaller of two integers, usable in device code.
         * @param[in] a First value.
         * @param[in] b Second value.
         * @return `min(a, b)`.
         */
        __device__ __forceinline__ int imin(int a, int b)
        {
            return a < b ? a : b;
        }

        /**
         * @brief Linear index of a coarse block.
         * @param[in] grid Lattice extents.
         * @param[in] ci Block coordinate along x.
         * @param[in] cj Block coordinate along y.
         * @param[in] ck Block coordinate along z.
         * @return Row-major index into the coarse arrays.
         */
        __device__ __forceinline__ int64_t coarse_index(const BandGrid &grid, int ci, int cj, int ck)
        {
            return ((int64_t)ci * grid.cy + cj) * grid.cz + ck;
        }

        /**
         * @brief Linear index of a fine vertex within its block.
         * @param[in] fi Local coordinate along x.
         * @param[in] fj Local coordinate along y.
         * @param[in] fk Local coordinate along z.
         * @return Row-major index in `[0, 512)`.
         */
        __device__ __forceinline__ int local_index(int fi, int fj, int fk)
        {
            return (fi * B + fj) * B + fk;
        }

        /**
         * @brief Tests whether a global vertex coordinate lies on the lattice.
         * @details Padding slots of the partial blocks at the high end fail this test, so they are
         * never treated as neighbours.
         * @param[in] grid Lattice extents.
         * @param[in] x Global coordinate along x.
         * @param[in] y Global coordinate along y.
         * @param[in] z Global coordinate along z.
         * @return True if the vertex exists.
         */
        __device__ __forceinline__ bool in_grid(const BandGrid &grid, int x, int y, int z)
        {
            return x >= 0 && x < grid.rx && y >= 0 && y < grid.ry && z >= 0 && z < grid.rz;
        }

        /**
         * @brief Label of a global lattice vertex.
         * @details A vertex in a band block reads its fine label; any other vertex shares its block's
         * coarse label, because a block outside the band is free space and floods as one piece.
         * @param[in] grid Lattice extents.
         * @param[in] x Global coordinate along x, on the lattice.
         * @param[in] y Global coordinate along y, on the lattice.
         * @param[in] z Global coordinate along z, on the lattice.
         * @param[in] coarse_mask Device array of coarse labels.
         * @param[in] band_lookup Device array mapping coarse index to band slot.
         * @param[in] fine_masks Device array of band-block fine labels.
         * @return The vertex's current label code.
         */
        __device__ __forceinline__ int8_t read_label(const BandGrid &grid, int x, int y, int z,
                                                     const int8_t *__restrict__ coarse_mask,
                                                     const int32_t *__restrict__ band_lookup,
                                                     const int8_t *__restrict__ fine_masks)
        {
            const int64_t c = coarse_index(grid, x / B, y / B, z / B);
            const int8_t coarse = coarse_mask[c];
            if (coarse != BAND_BLOCK)
                return coarse;
            return fine_masks[(int64_t)band_lookup[c] * B3 + local_index(x % B, y % B, z % B)];
        }

        /**
         * @brief Tests two axis-aligned boxes for overlap, touching counting as overlap.
         * @param[in] a_min First box lower corner.
         * @param[in] a_max First box upper corner.
         * @param[in] b_min Second box lower corner.
         * @param[in] b_max Second box upper corner.
         * @return True if the closed boxes intersect.
         */
        __device__ __forceinline__ bool boxes_overlap(const float3 &a_min, const float3 &a_max, const float3 &b_min,
                                                      const float3 &b_max)
        {
            return !(a_max.x < b_min.x || a_min.x > b_max.x || a_max.y < b_min.y || a_min.y > b_max.y ||
                     a_max.z < b_min.z || a_min.z > b_max.z);
        }
    } // namespace band_fill

    using namespace band_fill;

    // -------------------------------------------------------------
    // Band construction
    // -------------------------------------------------------------

    /**
     * @brief Classifies coarse blocks as band, boundary-exterior, or unknown.
     * @details A vertex is in the dilated band `D` iff a triangle meets the cube of half-width
     * `r + 1` around it, and the union of those cubes over a block's vertices is the block's vertex
     * box inflated by `r + 1`. So one box query per block decides exactly whether the block can hold
     * a wall vertex; blocks that cannot are free space and flood as a unit. Free blocks on the grid
     * boundary are the coarse seeds.
     * @param[out] coarse_mask Device array of coarse labels.
     * @param[in] grid Lattice extents.
     * @param[in] block_margin Inflation of the block's vertex box, `r + 1` plus slack, in spacings.
     * @param[in] node_mins Device array of BVH node lower bounds, lattice units.
     * @param[in] node_maxs Device array of BVH node upper bounds, lattice units.
     * @param[in] bvh_children Device array of BVH child index pairs.
     * @param[in] object_ids Device array mapping leaves to triangle indices.
     * @param[in] lattice_vertices Device array of mesh vertices, lattice units.
     * @param[in] triangles Device array of triangle vertex indices.
     * @param[in] num_objects Number of triangles.
     * @param[in,out] overflow_count Device counter of BVH traversals that overflowed their stack.
     * @note Launched with one thread per coarse block, `int64_t` indexed.
     */
    __global__ void
    classify_band_blocks_kernel(int8_t *__restrict__ coarse_mask, const BandGrid grid, const float block_margin,
                                const float3 *__restrict__ node_mins, const float3 *__restrict__ node_maxs,
                                const int2 *__restrict__ bvh_children, const int *__restrict__ object_ids,
                                const float3 *__restrict__ lattice_vertices, const int3 *__restrict__ triangles,
                                const int num_objects, int *__restrict__ overflow_count)
    {
        const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t plane = (int64_t)grid.cy * grid.cz;
        if (idx >= plane * grid.cx)
            return;

        const int ci = (int)(idx / plane);
        const int64_t rem = idx % plane;
        const int cj = (int)(rem / grid.cz);
        const int ck = (int)(rem % grid.cz);

        // The block's vertex extent; the last block along an axis may hold fewer than 8 vertices.
        const float3 box_min =
            make_float3((float)(ci * B) - block_margin, (float)(cj * B) - block_margin, (float)(ck * B) - block_margin);
        const float3 box_max = make_float3((float)imin(ci * B + B - 1, grid.rx - 1) + block_margin,
                                           (float)imin(cj * B + B - 1, grid.ry - 1) + block_margin,
                                           (float)imin(ck * B + B - 1, grid.rz - 1) + block_margin);

        bool overflowed = false;
        const bool hit = test_box_overlap_bvh_cf(box_min, box_max, node_mins, node_maxs, bvh_children, object_ids,
                                                 lattice_vertices, triangles, num_objects, &overflowed);
        if (overflowed)
            atomicAdd(overflow_count, 1);

        if (hit)
            coarse_mask[idx] = BAND_BLOCK;
        else if (ci == 0 || ci == grid.cx - 1 || cj == 0 || cj == grid.cy - 1 || ck == 0 || ck == grid.cz - 1)
            coarse_mask[idx] = EXTERIOR;
        else
            coarse_mask[idx] = UNKNOWN;
    }

    /**
     * @brief Marks each band-block vertex as surface band, dilated band, or free.
     * @details `S0` is the set of corners of cells a triangle overlaps, and a vertex is such a corner
     * iff a triangle meets the cube of half-width 1 around it (the union of its eight cells). The
     * Chebyshev dilation of `S0` by `r` is likewise the set of vertices whose cube of half-width
     * `r + 1` meets a triangle. Each vertex therefore needs only two box queries -- equivalent to
     * voxelizing every triangle and dilating, without allocating the cells or scattering writes. The
     * traversal first looks for any triangle in the large cube and then narrows its node test to the
     * small one. Free vertices on the grid boundary are the fine seeds, and padding slots past the
     * last vertex are set exterior so they never read as interior.
     * @param[in] band_coords Device array of band-block coarse coordinates.
     * @param[out] fine_masks Device array of band-block fine labels.
     * @param[in] num_band_blocks Number of band blocks.
     * @param[in] grid Lattice extents.
     * @param[in] surface_half_width Half-width of the `S0` cube, `1` plus slack, in spacings.
     * @param[in] dilated_half_width Half-width of the `D` cube, `r + 1` plus slack, in spacings.
     * @param[in] node_mins Device array of BVH node lower bounds, lattice units.
     * @param[in] node_maxs Device array of BVH node upper bounds, lattice units.
     * @param[in] bvh_children Device array of BVH child index pairs.
     * @param[in] object_ids Device array mapping leaves to triangle indices.
     * @param[in] lattice_vertices Device array of mesh vertices, lattice units.
     * @param[in] triangles Device array of triangle vertex indices.
     * @param[in] num_objects Number of triangles.
     * @param[in,out] overflow_count Device counter of BVH traversals that overflowed their stack.
     * @note Launched with one CUDA block per band block and one thread per fine vertex.
     */
    __global__ void mark_band_vertices_kernel(const int3 *__restrict__ band_coords, int8_t *__restrict__ fine_masks,
                                              const int num_band_blocks, const BandGrid grid,
                                              const float surface_half_width, const float dilated_half_width,
                                              const float3 *__restrict__ node_mins,
                                              const float3 *__restrict__ node_maxs,
                                              const int2 *__restrict__ bvh_children, const int *__restrict__ object_ids,
                                              const float3 *__restrict__ lattice_vertices,
                                              const int3 *__restrict__ triangles, const int num_objects,
                                              int *__restrict__ overflow_count)
    {
        const int b = blockIdx.x;
        if (b >= num_band_blocks)
            return;

        const int3 c = band_coords[b];
        const int x = c.x * B + threadIdx.x;
        const int y = c.y * B + threadIdx.y;
        const int z = c.z * B + threadIdx.z;
        int8_t &label = fine_masks[(int64_t)b * B3 + local_index(threadIdx.x, threadIdx.y, threadIdx.z)];

        if (!in_grid(grid, x, y, z))
        {
            label = EXTERIOR;
            return;
        }

        // Cells exist only inside the grid, so both cubes are clipped to it.
        const float fx = (float)x, fy = (float)y, fz = (float)z;
        const float3 upper = make_float3((float)(grid.rx - 1), (float)(grid.ry - 1), (float)(grid.rz - 1));
        const float3 d_min = make_float3(fmaxf(fx - dilated_half_width, 0.0f), fmaxf(fy - dilated_half_width, 0.0f),
                                         fmaxf(fz - dilated_half_width, 0.0f));
        const float3 d_max =
            make_float3(fminf(fx + dilated_half_width, upper.x), fminf(fy + dilated_half_width, upper.y),
                        fminf(fz + dilated_half_width, upper.z));
        const float3 s_min = make_float3(fmaxf(fx - surface_half_width, 0.0f), fmaxf(fy - surface_half_width, 0.0f),
                                         fmaxf(fz - surface_half_width, 0.0f));
        const float3 s_max =
            make_float3(fminf(fx + surface_half_width, upper.x), fminf(fy + surface_half_width, upper.y),
                        fminf(fz + surface_half_width, upper.z));

        bool in_dilated = false;
        bool in_surface = false;
        bool overflowed = false;
        bvh::traverse(
            num_objects, bvh_children,
            [&](int node_idx)
            {
                // Once any triangle meets the large cube, only the small cube is still undecided.
                return in_dilated ? boxes_overlap(s_min, s_max, node_mins[node_idx], node_maxs[node_idx])
                                  : boxes_overlap(d_min, d_max, node_mins[node_idx], node_maxs[node_idx]);
            },
            [&](int leaf_idx)
            {
                const int3 tri = triangles[object_ids[leaf_idx]];
                Triangle T(lattice_vertices[tri.x], lattice_vertices[tri.y], lattice_vertices[tri.z]);
                if (!in_dilated)
                {
                    if (!T.is_voxel_intersect(d_min, d_max))
                        return true;
                    in_dilated = true;
                }
                if (T.is_voxel_intersect(s_min, s_max))
                {
                    in_surface = true;
                    return false;
                }
                return true;
            },
            &overflowed);
        if (overflowed)
            atomicAdd(overflow_count, 1);

        if (in_surface)
            label = SURFACE;
        else if (in_dilated)
            label = DILATED;
        else if (x == 0 || x == grid.rx - 1 || y == 0 || y == grid.ry - 1 || z == 0 || z == grid.rz - 1)
            label = EXTERIOR;
        else
            label = UNKNOWN;
    }

    // -------------------------------------------------------------
    // Exterior flood
    // -------------------------------------------------------------

    /**
     * @brief Pulls the exterior label into unknown coarse blocks.
     * @details A free block is one connected piece of free vertices, so it becomes exterior as soon
     * as any of its face neighbours is: an exterior free block, or a band block with an exterior
     * vertex on the adjoining face layer. Only the thread owning a block writes it, and labels only
     * ever change from unknown to exterior, so racing reads can delay a step but never change the
     * fixed point.
     * @param[in,out] coarse_mask Device array of coarse labels.
     * @param[in] band_lookup Device array mapping coarse index to band slot.
     * @param[in] fine_masks Device array of band-block fine labels.
     * @param[in] grid Lattice extents.
     * @param[out] changed_flag Device flag set when any label changes.
     * @note Launched with one thread per coarse block, `int64_t` indexed.
     */
    __global__ void flood_coarse_pull_kernel(int8_t *__restrict__ coarse_mask, const int32_t *__restrict__ band_lookup,
                                             const int8_t *__restrict__ fine_masks, const BandGrid grid,
                                             int *__restrict__ changed_flag)
    {
        const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t plane = (int64_t)grid.cy * grid.cz;
        if (idx >= plane * grid.cx)
            return;
        if (coarse_mask[idx] != UNKNOWN)
            return;

        const int64_t rem = idx % plane;
        const int block[3] = {(int)(idx / plane), (int)(rem / grid.cz), (int)(rem % grid.cz)};
        const int coarse_dims[3] = {grid.cx, grid.cy, grid.cz};
        const int fine_dims[3] = {grid.rx, grid.ry, grid.rz};

        for (int axis = 0; axis < 3; ++axis)
        {
            for (int step = -1; step <= 1; step += 2)
            {
                int n[3] = {block[0], block[1], block[2]};
                n[axis] += step;
                if (n[axis] < 0 || n[axis] >= coarse_dims[axis])
                    continue;

                const int64_t n_idx = coarse_index(grid, n[0], n[1], n[2]);
                const int8_t n_label = coarse_mask[n_idx];
                bool exterior = (n_label == EXTERIOR);

                if (n_label == BAND_BLOCK)
                {
                    // The neighbour's layer touching this block, over the vertices that exist.
                    const int u_axis = (axis + 1) % 3;
                    const int w_axis = (axis + 2) % 3;
                    const int u_count = imin(B, fine_dims[u_axis] - block[u_axis] * B);
                    const int w_count = imin(B, fine_dims[w_axis] - block[w_axis] * B);
                    const int8_t *n_fine = fine_masks + (int64_t)band_lookup[n_idx] * B3;
                    int f[3];
                    f[axis] = (step < 0) ? B - 1 : 0;
                    for (int u = 0; u < u_count && !exterior; ++u)
                    {
                        for (int w = 0; w < w_count; ++w)
                        {
                            f[u_axis] = u;
                            f[w_axis] = w;
                            if (n_fine[local_index(f[0], f[1], f[2])] == EXTERIOR)
                            {
                                exterior = true;
                                break;
                            }
                        }
                    }
                }

                if (exterior)
                {
                    coarse_mask[idx] = EXTERIOR;
                    atomicExch(changed_flag, 1);
                    return;
                }
            }
        }
    }

    /**
     * @brief Pulls the exterior label through the free vertices of each band block.
     * @details Each unknown vertex first checks its neighbours outside the block -- a free block's
     * coarse label or another band block's fine label -- then the block sweeps `3B` rounds in shared
     * memory, where a free vertex turns exterior when an in-block neighbour is. Wall vertices never
     * change, and every lattice step that crosses the surface lies in a cell the surface touches, so
     * both ends of such a step are wall: water cannot pass. Pull updates are monotone, so the racing
     * in-place sweeps reach the same fixed point in any order.
     * @param[in] band_coords Device array of band-block coarse coordinates.
     * @param[in] band_lookup Device array mapping coarse index to band slot.
     * @param[in] coarse_mask Device array of coarse labels.
     * @param[in,out] fine_masks Device array of band-block fine labels.
     * @param[in] grid Lattice extents.
     * @param[out] changed_flag Device flag set when any label changes.
     * @note Launched with exactly one CUDA block per band block, threads cooperating over that
     * block's vertices through shared memory; no thread returns early, so every thread reaches every
     * synchronization point.
     */
    __global__ void flood_fine_pull_kernel(const int3 *__restrict__ band_coords,
                                           const int32_t *__restrict__ band_lookup,
                                           const int8_t *__restrict__ coarse_mask, int8_t *__restrict__ fine_masks,
                                           const BandGrid grid, int *__restrict__ changed_flag)
    {
        const int b = blockIdx.x;
        const int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;
        const int3 c = band_coords[b];
        const int x = c.x * B + fi, y = c.y * B + fj, z = c.z * B + fk;
        int8_t *block = fine_masks + (int64_t)b * B3;
        const int8_t initial = block[local_index(fi, fj, fk)];

        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};

        __shared__ int8_t s_mask[8][8][8];

        int8_t start = initial;
        if (initial == UNKNOWN)
        {
            for (int k = 0; k < 6; ++k)
            {
                const int nfi = fi + di[k], nfj = fj + dj[k], nfk = fk + dk[k];
                if (nfi >= 0 && nfi < B && nfj >= 0 && nfj < B && nfk >= 0 && nfk < B)
                    continue; // In-block neighbours are swept below.
                const int nx = x + di[k], ny = y + dj[k], nz = z + dk[k];
                if (!in_grid(grid, nx, ny, nz))
                    continue;
                if (read_label(grid, nx, ny, nz, coarse_mask, band_lookup, fine_masks) == EXTERIOR)
                {
                    start = EXTERIOR;
                    break;
                }
            }
        }
        s_mask[fi][fj][fk] = start;
        __syncthreads();

        for (int sweep = 0; sweep < 3 * B; ++sweep)
        {
            if (s_mask[fi][fj][fk] == UNKNOWN)
            {
                for (int k = 0; k < 6; ++k)
                {
                    const int nfi = fi + di[k], nfj = fj + dj[k], nfk = fk + dk[k];
                    if (nfi < 0 || nfi >= B || nfj < 0 || nfj >= B || nfk < 0 || nfk >= B)
                        continue;
                    // Padding slots are labelled exterior but are not vertices.
                    if (!in_grid(grid, x + di[k], y + dj[k], z + dk[k]))
                        continue;
                    if (s_mask[nfi][nfj][nfk] == EXTERIOR)
                    {
                        s_mask[fi][fj][fk] = EXTERIOR;
                        break;
                    }
                }
            }
            __syncthreads();
        }

        if (s_mask[fi][fj][fk] != initial)
        {
            block[local_index(fi, fj, fk)] = s_mask[fi][fj][fk];
            atomicExch(changed_flag, 1);
        }
    }

    // -------------------------------------------------------------
    // Cavity release
    // -------------------------------------------------------------

    /**
     * @brief Flags interior vertices that may be the smallest-index vertex of their component.
     * @details Each interior component is decided once, by its vertex of smallest global index. That
     * vertex has no interior neighbour at `x - 1`, `y - 1` or `z - 1`, so only vertices passing this
     * local test need to run the bounded search; the rest are skipped without any per-thread scratch.
     * A neighbour in an interior free block counts as interior.
     * @param[in] band_coords Device array of band-block coarse coordinates.
     * @param[in] band_lookup Device array mapping coarse index to band slot.
     * @param[in] coarse_mask Device array of coarse labels.
     * @param[in] fine_masks Device array of band-block fine labels.
     * @param[out] root_flags Device array, one byte per fine slot, set to 1 for candidates.
     * @param[in] num_band_blocks Number of band blocks.
     * @param[in] grid Lattice extents.
     * @note Launched with one CUDA block per band block and one thread per fine vertex.
     */
    __global__ void find_cavity_roots_kernel(const int3 *__restrict__ band_coords,
                                             const int32_t *__restrict__ band_lookup,
                                             const int8_t *__restrict__ coarse_mask,
                                             const int8_t *__restrict__ fine_masks, int8_t *__restrict__ root_flags,
                                             const int num_band_blocks, const BandGrid grid)
    {
        const int b = blockIdx.x;
        if (b >= num_band_blocks)
            return;

        const int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;
        const int64_t slot = (int64_t)b * B3 + local_index(fi, fj, fk);
        root_flags[slot] = 0;
        if (fine_masks[slot] != INTERIOR)
            return;

        const int3 c = band_coords[b];
        const int x = c.x * B + fi, y = c.y * B + fj, z = c.z * B + fk;
        if ((x > 0 && read_label(grid, x - 1, y, z, coarse_mask, band_lookup, fine_masks) == INTERIOR) ||
            (y > 0 && read_label(grid, x, y - 1, z, coarse_mask, band_lookup, fine_masks) == INTERIOR) ||
            (z > 0 && read_label(grid, x, y, z - 1, coarse_mask, band_lookup, fine_masks) == INTERIOR))
            return;
        root_flags[slot] = 1;
    }

    /**
     * @brief Releases small interior components that do not reach an interior free block.
     * @details Dilation seals gaps, so it can also enclose pockets of free space that are not inside
     * the object. For each candidate root this runs a breadth-first search over its interior
     * component, bounded at @p cavity_max_voxels vertices, and gives up as soon as the component
     * proves large, touches an interior free block, or contains a vertex of smaller index (another
     * vertex is its root). A search that exhausts its component is the root of a small isolated
     * cavity and marks every member released. Each component has exactly one root, so the writes
     * never collide, and released vertices still read as interior to any other search, so the
     * outcome does not depend on thread order.
     * @param[in] roots Device array of candidate fine slots for this batch.
     * @param[in] num_roots Number of candidates in the batch.
     * @param[in,out] queue_scratch Device array of `num_roots * (cavity_max_voxels + 1)` search slots.
     * @param[in] cavity_max_voxels Largest component released.
     * @param[in] band_coords Device array of band-block coarse coordinates.
     * @param[in] band_lookup Device array mapping coarse index to band slot.
     * @param[in] coarse_mask Device array of coarse labels.
     * @param[in,out] fine_masks Device array of band-block fine labels.
     * @param[in] grid Lattice extents.
     * @note Launched with one thread per candidate, `int64_t` indexed. Visited tests scan the search
     * queue, so the cost of one search grows with the square of @p cavity_max_voxels.
     */
    __global__ void release_small_cavities_kernel(const int64_t *__restrict__ roots, const int64_t num_roots,
                                                  int64_t *__restrict__ queue_scratch, const int cavity_max_voxels,
                                                  const int3 *__restrict__ band_coords,
                                                  const int32_t *__restrict__ band_lookup,
                                                  const int8_t *__restrict__ coarse_mask,
                                                  int8_t *__restrict__ fine_masks, const BandGrid grid)
    {
        const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_roots)
            return;

        const int64_t yz = (int64_t)grid.ry * grid.rz;
        const int64_t slot = roots[idx];
        const int b = (int)(slot / B3);
        const int local = (int)(slot % B3);
        const int3 c = band_coords[b];
        const int64_t root = ((int64_t)(c.x * B + local / (B * B)) * grid.ry + (c.y * B + (local / B) % B)) * grid.rz +
                             (c.z * B + local % B);

        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};

        int64_t *queue = queue_scratch + idx * (int64_t)(cavity_max_voxels + 1);
        queue[0] = root;
        int count = 1;

        for (int head = 0; head < count; ++head)
        {
            const int64_t current = queue[head];
            const int x = (int)(current / yz);
            const int y = (int)((current / grid.rz) % grid.ry);
            const int z = (int)(current % grid.rz);

            for (int k = 0; k < 6; ++k)
            {
                const int nx = x + di[k], ny = y + dj[k], nz = z + dk[k];
                if (!in_grid(grid, nx, ny, nz))
                    continue;

                const int64_t n_coarse = coarse_index(grid, nx / B, ny / B, nz / B);
                const int8_t coarse = coarse_mask[n_coarse];
                if (coarse == INTERIOR)
                    return; // Reaches a free interior block: part of the bulk interior.
                if (coarse != BAND_BLOCK)
                    continue;
                const int8_t fine =
                    fine_masks[(int64_t)band_lookup[n_coarse] * B3 + local_index(nx % B, ny % B, nz % B)];
                if (fine != INTERIOR && fine != RELEASED)
                    continue;

                const int64_t neighbour = ((int64_t)nx * grid.ry + ny) * grid.rz + nz;
                if (neighbour < root)
                    return; // Not the root; that vertex decides this component.

                bool seen = false;
                for (int q = 0; q < count; ++q)
                {
                    if (queue[q] == neighbour)
                    {
                        seen = true;
                        break;
                    }
                }
                if (seen)
                    continue;
                if (count == cavity_max_voxels)
                    return; // Larger than a cavity.
                queue[count++] = neighbour;
            }
        }

        for (int q = 0; q < count; ++q)
        {
            const int x = (int)(queue[q] / yz);
            const int y = (int)((queue[q] / grid.rz) % grid.ry);
            const int z = (int)(queue[q] % grid.rz);
            const int64_t n_coarse = coarse_index(grid, x / B, y / B, z / B);
            fine_masks[(int64_t)band_lookup[n_coarse] * B3 + local_index(x % B, y % B, z % B)] = RELEASED;
        }
    }

    // -------------------------------------------------------------
    // Band resolution
    // -------------------------------------------------------------

    /**
     * @brief Casts one synchronous round of neighbour votes for undetermined band vertices.
     * @details An eligible vertex with at least one resolved face neighbour takes the majority label,
     * ties going to exterior. The vote is written as a pending code into the vertex's own slot and
     * pending codes read as undetermined, so every vertex sees the labels from the start of the
     * round and the result does not depend on thread order. Phase 0 resolves `D \ S0` with `S0` as a
     * barrier, so exterior and interior meet in the middle of a sealed hole while crevices sealed
     * only by the dilation keep their exterior label. Phase 1 resolves `S0`, and phase 2 anything
     * left.
     * @param[in] band_coords Device array of band-block coarse coordinates.
     * @param[in] band_lookup Device array mapping coarse index to band slot.
     * @param[in] coarse_mask Device array of coarse labels.
     * @param[in,out] fine_masks Device array of band-block fine labels.
     * @param[in] num_band_blocks Number of band blocks.
     * @param[in] grid Lattice extents.
     * @param[in] phase 0 for `D \ S0`, 1 for `S0`, 2 for both.
     * @note Launched with one CUDA block per band block and one thread per fine vertex; the host
     * commits pending codes between rounds.
     */
    __global__ void resolve_band_kernel(const int3 *__restrict__ band_coords, const int32_t *__restrict__ band_lookup,
                                        const int8_t *__restrict__ coarse_mask, int8_t *__restrict__ fine_masks,
                                        const int num_band_blocks, const BandGrid grid, const int phase)
    {
        const int b = blockIdx.x;
        if (b >= num_band_blocks)
            return;

        const int fi = threadIdx.x, fj = threadIdx.y, fk = threadIdx.z;
        const int64_t slot = (int64_t)b * B3 + local_index(fi, fj, fk);
        const int8_t label = fine_masks[slot];
        const bool eligible = (phase == 0)   ? label == DILATED
                              : (phase == 1) ? label == SURFACE
                                             : (label == DILATED || label == SURFACE);
        if (!eligible)
            return;

        const int3 c = band_coords[b];
        const int x = c.x * B + fi, y = c.y * B + fj, z = c.z * B + fk;
        const int di[6] = {-1, 1, 0, 0, 0, 0};
        const int dj[6] = {0, 0, -1, 1, 0, 0};
        const int dk[6] = {0, 0, 0, 0, -1, 1};

        int votes_exterior = 0;
        int votes_interior = 0;
        for (int k = 0; k < 6; ++k)
        {
            const int nx = x + di[k], ny = y + dj[k], nz = z + dk[k];
            if (!in_grid(grid, nx, ny, nz))
                continue;
            const int8_t n_label = read_label(grid, nx, ny, nz, coarse_mask, band_lookup, fine_masks);
            if (n_label == EXTERIOR)
                ++votes_exterior;
            else if (n_label == INTERIOR)
                ++votes_interior;
        }

        if (votes_exterior + votes_interior > 0)
            fine_masks[slot] = (votes_interior > votes_exterior) ? PENDING_INTERIOR : PENDING_EXTERIOR;
    }

    // -------------------------------------------------------------
    // Host Pipeline
    // -------------------------------------------------------------

    /**
     * @brief Number of CUDA blocks covering a one-dimensional launch.
     * @param[in] total Number of threads needed.
     * @return `ceil(total / NTHREADS)`, checked to fit a launch.
     */
    static int launch_blocks(int64_t total)
    {
        const int64_t blocks = (total + NTHREADS - 1) / NTHREADS;
        TORCH_CHECK(blocks <= INT_MAX, "compute_flood_fill_band: launch of ", total, " threads is too large.");
        return static_cast<int>(blocks);
    }

    BandFloodFillResult compute_flood_fill_band(const torch::Tensor &vertices, const torch::Tensor &triangles,
                                                const torch::Tensor &aabb_mins, const torch::Tensor &aabb_maxs,
                                                const torch::Tensor &bvh_children, const torch::Tensor &object_ids,
                                                std::vector<float> grid_min, std::vector<float> grid_max,
                                                std::vector<int64_t> grid_res, int dilation_radius,
                                                int64_t cavity_max_voxels)
    {
        TORCH_CHECK(grid_min.size() == 3 && grid_max.size() == 3 && grid_res.size() == 3,
                    "compute_flood_fill_band: grid_min, grid_max and grid_res must have 3 elements.");
        TORCH_CHECK(dilation_radius >= 0, "compute_flood_fill_band: dilation_radius must be non-negative, got ",
                    dilation_radius, ".");

        // The band reaches r + 1 spacings from the surface; one more keeps the boundary vertices free.
        const int64_t margin = (int64_t)dilation_radius + 2;
        for (int a = 0; a < 3; ++a)
        {
            TORCH_CHECK(grid_res[a] >= 2 * margin + 2, "compute_flood_fill_band: grid_res must be at least ",
                        2 * margin + 2, " per axis for dilation_radius ", dilation_radius, ", got ", grid_res[a], ".");
            TORCH_CHECK(grid_res[a] <= (int64_t)1 << 24,
                        "compute_flood_fill_band: grid_res above 2^24 cannot place vertices exactly in float.");
            TORCH_CHECK(grid_max[a] > grid_min[a], "compute_flood_fill_band: grid_max must exceed grid_min.");
        }

        const int64_t reach = 2 * (int64_t)dilation_radius + 1;
        const int64_t cavity_limit = (cavity_max_voxels < 0) ? reach * reach * reach : cavity_max_voxels;
        TORCH_CHECK(cavity_limit <= 4096,
                    "compute_flood_fill_band: cavity_max_voxels above 4096 is not supported (search cost grows "
                    "quadratically), got ",
                    cavity_limit, ".");

        c10::cuda::CUDAGuard device_guard(vertices.device());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        const int64_t rx = grid_res[0], ry = grid_res[1], rz = grid_res[2];
        const BandGrid grid{static_cast<int>(rx),           static_cast<int>(ry),
                            static_cast<int>(rz),           static_cast<int>((rx + 7) / 8),
                            static_cast<int>((ry + 7) / 8), static_cast<int>((rz + 7) / 8)};
        const int64_t total_coarse = (int64_t)grid.cx * grid.cy * grid.cz;

        auto dev = vertices.device();
        auto opt_i8 = torch::TensorOptions().device(dev).dtype(torch::kInt8);
        auto opt_i32 = torch::TensorOptions().device(dev).dtype(torch::kInt32);
        auto opt_i64 = torch::TensorOptions().device(dev).dtype(torch::kInt64);
        auto opt_f64 = torch::TensorOptions().device(dev).dtype(torch::kFloat64);

        BandFloodFillResult result;
        result.block_size = {B, B, B};
        result.coarse_res = {grid.cx, grid.cy, grid.cz};
        result.stats = {{"num_band_blocks", 0},
                        {"flood_rounds", 0},
                        {"num_released_cavity_voxels", 0},
                        {"resolution_rounds", 0},
                        {"num_unresolved_defaulted", 0},
                        {"bvh_stack_overflows", 0}};

        const int64_t num_objects = triangles.size(0);
        if (num_objects == 0)
        {
            result.coarse_mask = torch::full({grid.cx, grid.cy, grid.cz}, EXTERIOR, opt_i8);
            result.band_block_coords = torch::empty({0, 3}, opt_i32);
            result.band_block_lookup = torch::full({grid.cx, grid.cy, grid.cz}, -1, opt_i32);
            result.fine_masks = torch::empty({0, B, B, B}, opt_i8);
            return result;
        }

        // ---- Lattice-unit geometry --------------------------------------------------------------
        const double spacing[3] = {((double)grid_max[0] - grid_min[0]) / (double)(rx - 1),
                                   ((double)grid_max[1] - grid_min[1]) / (double)(ry - 1),
                                   ((double)grid_max[2] - grid_min[2]) / (double)(rz - 1)};
        auto origin_t = torch::tensor({(double)grid_min[0], (double)grid_min[1], (double)grid_min[2]}, opt_f64);
        auto spacing_t = torch::tensor({spacing[0], spacing[1], spacing[2]}, opt_f64);

        // Mapped in double so the rounding to float happens once, at the final coordinate.
        auto lattice_vertices =
            ((vertices.to(torch::kFloat64) - origin_t) / spacing_t).to(torch::kFloat32).contiguous();
        // Node boxes come from world-space float bounds; the slack keeps every mapped vertex inside its box.
        const double node_slack = 1e-2;
        auto node_mins =
            ((aabb_mins.to(torch::kFloat64) - origin_t) / spacing_t - node_slack).to(torch::kFloat32).contiguous();
        auto node_maxs =
            ((aabb_maxs.to(torch::kFloat64) - origin_t) / spacing_t + node_slack).to(torch::kFloat32).contiguous();

        // Box slack in spacings: a few ULPs of the largest lattice coordinate, never below 1e-4.
        const float max_res = static_cast<float>(std::max({rx, ry, rz}));
        const float eps = std::max(1e-4f, 4.0f * max_res / 16777216.0f);
        const float surface_half_width = 1.0f + eps;
        const float dilated_half_width = static_cast<float>(dilation_radius + 1) + eps;
        const float block_margin = static_cast<float>(dilation_radius + 1) + std::max(1e-2f, 2.0f * eps);

        // ---- Preconditions ----------------------------------------------------------------------
        {
            auto used = lattice_vertices.index_select(0, triangles.flatten().to(torch::kInt64));
            auto bounds = torch::stack({std::get<0>(used.min(0)), std::get<0>(used.max(0))}).cpu();
            auto acc = bounds.accessor<float, 2>();
            const char axis_names[3] = {'x', 'y', 'z'};
            for (int a = 0; a < 3; ++a)
            {
                const double lo = acc[0][a];
                const double hi = acc[1][a];
                const double res_a = (double)grid_res[a];
                if (!(lo >= (double)margin && hi <= res_a - 1.0 - (double)margin))
                {
                    const double extent = std::max(0.0, (hi - lo) * spacing[a]);
                    const double required = (double)margin * extent / (res_a - 1.0 - 2.0 * (double)margin);
                    std::ostringstream msg;
                    msg << "compute_flood_fill_band: the mesh must lie at least dilation_radius + 2 = " << margin
                        << " lattice spacings inside the grid so the dilated band never reaches the boundary, but "
                        << "along " << axis_names[a] << " it spans lattice coordinates [" << lo << ", " << hi
                        << "] of [0, " << (grid_res[a] - 1) << "]. Pad the grid by at least " << required
                        << " world units beyond the mesh on each side of that axis, or lower dilation_radius.";
                    throw std::runtime_error(msg.str());
                }
            }
        }

        const float3 *p_node_mins = (const float3 *)node_mins.data_ptr<float>();
        const float3 *p_node_maxs = (const float3 *)node_maxs.data_ptr<float>();
        const int2 *p_children = (const int2 *)bvh_children.data_ptr<int>();
        const int *p_object_ids = object_ids.data_ptr<int>();
        const float3 *p_lattice_vertices = (const float3 *)lattice_vertices.data_ptr<float>();
        const int3 *p_triangles = (const int3 *)triangles.data_ptr<int>();
        const int n_objects = static_cast<int>(num_objects);

        auto overflow_count = torch::zeros({1}, opt_i32);
        const int threads = NTHREADS;
        const int coarse_blocks = launch_blocks(total_coarse);

        // ---- Stage 1: coarse classification -----------------------------------------------------
        auto coarse_mask = torch::empty({grid.cx, grid.cy, grid.cz}, opt_i8);
        classify_band_blocks_kernel<<<coarse_blocks, threads, 0, stream>>>(
            coarse_mask.data_ptr<int8_t>(), grid, block_margin, p_node_mins, p_node_maxs, p_children, p_object_ids,
            p_lattice_vertices, p_triangles, n_objects, overflow_count.data_ptr<int>());

        // ---- Stage 2: index band blocks ---------------------------------------------------------
        auto band_1d = torch::nonzero((coarse_mask == BAND_BLOCK).flatten()).squeeze(1);
        const int64_t num_band_blocks = band_1d.size(0);
        TORCH_CHECK(num_band_blocks <= INT_MAX, "compute_flood_fill_band: too many band blocks.");
        const int n_band = static_cast<int>(num_band_blocks);

        auto band_coords = torch::empty({num_band_blocks, 3}, opt_i32);
        auto band_lookup = torch::full({grid.cx, grid.cy, grid.cz}, -1, opt_i32);
        auto fine_masks = torch::empty({num_band_blocks, B, B, B}, opt_i8);
        if (num_band_blocks > 0)
        {
            const int64_t plane = (int64_t)grid.cy * grid.cz;
            auto ci = band_1d.div(plane, "trunc");
            auto rem = band_1d.remainder(plane);
            band_coords =
                torch::stack({ci, rem.div(grid.cz, "trunc"), rem.remainder(grid.cz)}, 1).to(torch::kInt32).contiguous();
            band_lookup.view({-1}).index_put_({band_1d}, torch::arange(num_band_blocks, opt_i32));
        }
        band_1d = torch::Tensor();

        const int3 *p_band_coords = (const int3 *)band_coords.data_ptr<int>();
        const int32_t *p_band_lookup = band_lookup.data_ptr<int32_t>();
        dim3 fine_threads(B, B, B);

        // ---- Stage 3: surface and dilated band --------------------------------------------------
        if (n_band > 0)
        {
            mark_band_vertices_kernel<<<n_band, fine_threads, 0, stream>>>(
                p_band_coords, fine_masks.data_ptr<int8_t>(), n_band, grid, surface_half_width, dilated_half_width,
                p_node_mins, p_node_maxs, p_children, p_object_ids, p_lattice_vertices, p_triangles, n_objects,
                overflow_count.data_ptr<int>());
        }
        const int64_t overflows = overflow_count.item<int>();
        if (overflows > 0)
        {
            std::ostringstream msg;
            msg << "compute_flood_fill_band: " << overflows << " BVH traversals overflowed the " << BVH_STACK_SIZE
                << "-entry stack, so the band may have holes. Rebuild with a larger BVH_STACK_SIZE.";
            throw std::runtime_error(msg.str());
        }

        // ---- Stage 4: exterior flood ------------------------------------------------------------
        // Every round that sets the flag turns at least one unknown label exterior, so the loop is
        // bounded by the label count; reaching the bound means a kernel bug, not a hard input.
        const int64_t max_flood_rounds = total_coarse + num_band_blocks * B3 + 1;
        auto changed_flag = torch::zeros({1}, opt_i32);
        int64_t flood_rounds = 0;
        for (;;)
        {
            TORCH_CHECK(flood_rounds < max_flood_rounds, "compute_flood_fill_band: flood failed to converge.");
            changed_flag.zero_();
            flood_coarse_pull_kernel<<<coarse_blocks, threads, 0, stream>>>(
                coarse_mask.data_ptr<int8_t>(), p_band_lookup, fine_masks.data_ptr<int8_t>(), grid,
                changed_flag.data_ptr<int>());
            if (n_band > 0)
            {
                flood_fine_pull_kernel<<<n_band, fine_threads, 0, stream>>>(
                    p_band_coords, p_band_lookup, coarse_mask.data_ptr<int8_t>(), fine_masks.data_ptr<int8_t>(), grid,
                    changed_flag.data_ptr<int>());
            }
            ++flood_rounds;
            if (changed_flag.item<int>() == 0)
                break;
        }

        // Whatever water never reached is enclosed by the wall.
        coarse_mask.masked_fill_(coarse_mask == UNKNOWN, INTERIOR);
        fine_masks.masked_fill_(fine_masks == UNKNOWN, INTERIOR);

        // ---- Stage 5: cavity release ------------------------------------------------------------
        int64_t num_released = 0;
        if (cavity_limit > 0 && n_band > 0)
        {
            auto root_flags = torch::empty({num_band_blocks, B, B, B}, opt_i8);
            find_cavity_roots_kernel<<<n_band, fine_threads, 0, stream>>>(
                p_band_coords, p_band_lookup, coarse_mask.data_ptr<int8_t>(), fine_masks.data_ptr<int8_t>(),
                root_flags.data_ptr<int8_t>(), n_band, grid);
            auto roots = torch::nonzero(root_flags.view({-1})).squeeze(1).contiguous();
            root_flags = torch::Tensor();

            const int64_t num_roots = roots.size(0);
            if (num_roots > 0)
            {
                // Search queues live in device memory, at most 16 MiB of candidates at a time.
                const int64_t queue_len = cavity_limit + 1;
                const int64_t batch =
                    std::max<int64_t>(1, std::min<int64_t>(num_roots, ((int64_t)16 << 20) / (8 * queue_len)));
                auto queue_scratch = torch::empty({batch * queue_len}, opt_i64);
                for (int64_t start = 0; start < num_roots; start += batch)
                {
                    const int64_t count = std::min(batch, num_roots - start);
                    auto chunk = roots.slice(0, start, start + count).contiguous();
                    release_small_cavities_kernel<<<launch_blocks(count), threads, 0, stream>>>(
                        chunk.data_ptr<int64_t>(), count, queue_scratch.data_ptr<int64_t>(),
                        static_cast<int>(cavity_limit), p_band_coords, p_band_lookup, coarse_mask.data_ptr<int8_t>(),
                        fine_masks.data_ptr<int8_t>(), grid);
                }
            }

            auto released = (fine_masks == RELEASED);
            num_released = released.sum().item<int64_t>();
            fine_masks.masked_fill_(released, DILATED);
        }

        // ---- Stage 6: band resolution -----------------------------------------------------------
        int64_t resolution_rounds = 0;
        int64_t num_unresolved = 0;
        if (n_band > 0)
        {
            const int64_t max_resolution_rounds = num_band_blocks * B3 + 3;
            for (int phase = 0; phase < 3; ++phase)
            {
                for (;;)
                {
                    TORCH_CHECK(resolution_rounds < max_resolution_rounds,
                                "compute_flood_fill_band: band resolution failed to converge.");
                    resolve_band_kernel<<<n_band, fine_threads, 0, stream>>>(
                        p_band_coords, p_band_lookup, coarse_mask.data_ptr<int8_t>(), fine_masks.data_ptr<int8_t>(),
                        n_band, grid, phase);
                    auto to_exterior = (fine_masks == PENDING_EXTERIOR);
                    auto to_interior = (fine_masks == PENDING_INTERIOR);
                    if (!(to_exterior | to_interior).any().item<bool>())
                        break;
                    fine_masks.masked_fill_(to_exterior, EXTERIOR);
                    fine_masks.masked_fill_(to_interior, INTERIOR);
                    ++resolution_rounds;
                }
            }

            // Only a band component with no resolved neighbour anywhere is left; call it exterior.
            auto leftover = (fine_masks == DILATED) | (fine_masks == SURFACE);
            num_unresolved = leftover.sum().item<int64_t>();
            fine_masks.masked_fill_(leftover, EXTERIOR);
        }

        result.coarse_mask = coarse_mask;
        result.band_block_coords = band_coords;
        result.band_block_lookup = band_lookup;
        result.fine_masks = fine_masks;
        result.stats["num_band_blocks"] = num_band_blocks;
        result.stats["flood_rounds"] = flood_rounds;
        result.stats["num_released_cavity_voxels"] = num_released;
        result.stats["resolution_rounds"] = resolution_rounds;
        result.stats["num_unresolved_defaulted"] = num_unresolved;
        result.stats["bvh_stack_overflows"] = overflows;
        return result;
    }

} // namespace ops
