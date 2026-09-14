#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../../ops/flood_fill_band.h"
#include "../../check.h"

namespace py = pybind11;

/**
 * @brief Tensor-level entry point for the leak-resistant band flood fill.
 * @details Sits between pybind11 and the host dispatcher: it applies the `CHECK_INPUT`
 * contract -- CUDA device, contiguous layout, expected dtype -- then calls the kernel
 * launcher. Validating here keeps the launch path free of checks and gives Python callers a
 * clear error instead of a device fault.
 * @return Mode 5's tuple of labels followed by a dictionary of build diagnostics.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int64_t>, std::vector<int64_t>,
           std::map<std::string, int64_t>>
compute_flood_fill_band_wrapper(torch::Tensor vertices, torch::Tensor triangles, torch::Tensor aabb_mins,
                                torch::Tensor aabb_maxs, torch::Tensor bvh_children, torch::Tensor object_ids,
                                std::vector<float> grid_min, std::vector<float> grid_max, std::vector<int64_t> grid_res,
                                int dilation_radius, int64_t cavity_max_voxels)
{
    CHECK_INPUT(vertices);
    CHECK_INPUT(triangles);
    CHECK_INPUT(aabb_mins);
    CHECK_INPUT(aabb_maxs);
    CHECK_INPUT(bvh_children);
    CHECK_INPUT(object_ids);
    TORCH_CHECK(grid_min.size() == 3, "grid_min must have 3 elements.");
    TORCH_CHECK(grid_max.size() == 3, "grid_max must have 3 elements.");
    TORCH_CHECK(grid_res.size() == 3, "grid_res must have 3 elements.");

    auto res = ops::compute_flood_fill_band(vertices, triangles, aabb_mins, aabb_maxs, bvh_children, object_ids,
                                            grid_min, grid_max, grid_res, dilation_radius, cavity_max_voxels);

    return std::make_tuple(res.coarse_mask, res.band_block_coords, res.band_block_lookup, res.fine_masks,
                           res.block_size, res.coarse_res, res.stats);
}

/**
 * @brief Registers the leak-resistant band flood fill operator on the extension module.
 * @details Called once from `pybind.cpp` with the root module, so every symbol
 * defined here lands directly on `conquer3d._C`.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_flood_fill_band(py::module_ &m)
{
    m.def("compute_flood_fill_band", &compute_flood_fill_band_wrapper, py::arg("vertices"), py::arg("triangles"),
          py::arg("aabb_mins"), py::arg("aabb_maxs"), py::arg("bvh_children"), py::arg("object_ids"),
          py::arg("grid_min"), py::arg("grid_max"), py::arg("grid_res"), py::arg("dilation_radius") = 2,
          py::arg("cavity_max_voxels") = -1,
          R"pbdoc(
          Computes the leak-resistant band flood fill used by sign_mode=6.

          Water floods from the grid boundary and stops at a wall of lattice vertices: the corners of
          every cell a triangle touches, dilated by `dilation_radius` spacings, so holes narrower than
          the dilation are sealed. Small interior pockets the dilation creates are released and the wall
          is resolved by neighbour agreement, leaving every vertex exterior or interior. Storage follows
          compute_flood_fill_cf, so memory scales with surface area rather than volume.

          The wall trades thin detail for leak resistance: solid parts thinner than about
          `2 * dilation_radius + 3` spacings vanish, slots narrower than 3 spacings fill in, and a sealed-off
          pocket holding more than `cavity_max_voxels` free vertices stays interior.

          The lattice has `grid_res` vertices per axis spanning [grid_min, grid_max]. The mesh must lie
          at least `dilation_radius + 2` spacings inside it; otherwise a RuntimeError names the padding
          required.

          Args:
              vertices (torch.Tensor): (V, 3) float32 mesh vertices on CUDA.
              triangles (torch.Tensor): (F, 3) int32 triangle indices on CUDA.
              aabb_mins (torch.Tensor): (2F-1, 3) float32 BVH node min corners.
              aabb_maxs (torch.Tensor): (2F-1, 3) float32 BVH node max corners.
              bvh_children (torch.Tensor): (2F-1, 2) int32 BVH child node indices.
              object_ids (torch.Tensor): (F,) int32 leaf-to-triangle map.
              grid_min (List[float]): Lower grid corner [x, y, z].
              grid_max (List[float]): Upper grid corner [x, y, z].
              grid_res (List[int]): Vertices per axis [rx, ry, rz].
              dilation_radius (int, optional): Chebyshev dilation radius in spacings. Defaults to 2.
              cavity_max_voxels (int, optional): Largest interior component, in vertices, released for
                  resolution; -1 selects (2 * dilation_radius + 1)^3 and 0 disables the release.
                  Defaults to -1.

          Returns:
              Tuple[Tensor, Tensor, Tensor, Tensor, List[int], List[int], Dict[str, int]]:
                  - coarse_mask (Tensor): (Cx, Cy, Cz) int8 labels: 2 exterior, -1 interior, 1 band block.
                  - band_block_coords (Tensor): (N, 3) int32 coarse coordinates of the band blocks.
                  - band_block_lookup (Tensor): (Cx, Cy, Cz) int32 band slot, -1 elsewhere.
                  - fine_masks (Tensor): (N, 8, 8, 8) int8 labels: 2 exterior, -1 interior.
                  - block_size (List[int]): [8, 8, 8].
                  - coarse_res (List[int]): [Cx, Cy, Cz] = ceil(grid_res / 8).
                  - stats (Dict[str, int]): num_band_blocks, flood_rounds, num_released_cavity_voxels,
                    resolution_rounds, num_unresolved_defaulted, bvh_stack_overflows.
          )pbdoc");
}
