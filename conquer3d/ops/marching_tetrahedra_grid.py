"""Marching Tetrahedra Grid (MTG) algorithm for 3D isosurface extraction.

This module provides GPU-accelerated Marching Tetrahedra on structured cubic voxel grids,
internally subdividing each cubic voxel cell into 6 tetrahedra to guarantee watertight
topological consistency without ambiguous face saddle points.
"""

from typing import Tuple, Optional
import torch

from .._C import marching_tetrahedra_grid as marching_tetrahedra_grid_func


def marching_tetrahedra_grid(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    voxel_values: torch.Tensor,
    grid_normals: Optional[torch.Tensor] = None,
    grid_colors: Optional[torch.Tensor] = None,
    iso: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extracts a topological 2-manifold isosurface from a structured voxel grid using MTG.

    Decomposes each cubic voxel into 6 tetrahedral sub-elements on GPU, eliminating
    Marching Cubes ambiguous face configurations.

    Args:
        grid_vertices (torch.Tensor): Global vertex coordinates `(V, 3)` with dtype
            `torch.float32` on CUDA.
        voxels (torch.Tensor): Voxel corner indices `(N, 8)` mapping to `grid_vertices`
            with dtype `torch.int32` on CUDA.
        voxel_values (torch.Tensor): Scalar or SDF values `(V,)` at each grid vertex on CUDA.
        grid_normals (torch.Tensor, optional): Surface normals `(V, 3)` on CUDA. Defaults to None.
        grid_colors (torch.Tensor, optional): RGB/feature colors `(V, C)` on CUDA. Defaults to None.
        iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - vertices (torch.Tensor): Extracted mesh surface vertices `(M, 3)`.
            - triangles (torch.Tensor): Extracted triangle face indices `(T, 3)`.
            - normals (torch.Tensor | None): Interpolated vertex normals `(M, 3)`, or None if omitted.
            - colors (torch.Tensor | None): Interpolated vertex colors `(M, C)`, or None if omitted.

    Raises:
        ValueError: If input tensors are not on CUDA.

    Example:
        >>> from conquer3d.ops import marching_tetrahedra_grid
        >>> verts, faces, _, _ = marching_tetrahedra_grid(grid_vertices, voxels, sdfs, iso=0.0)
    """
    if not all(t.is_cuda for t in [grid_vertices, voxels, voxel_values]):
        raise ValueError("All input tensors must be CUDA tensors.")
        
    grid_vertices_c = grid_vertices.contiguous().to(torch.float32)
    voxels_c = voxels.contiguous().to(torch.int32)
    voxel_values_c = voxel_values.contiguous().to(torch.float32)
    
    grid_normals_c = None
    if grid_normals is not None:
        if not grid_normals.is_cuda:
            raise ValueError("grid_normals must be a CUDA tensor.")
        grid_normals_c = grid_normals.contiguous().to(torch.float32)

    grid_colors_c = None
    if grid_colors is not None:
        if not grid_colors.is_cuda:
            raise ValueError("grid_colors must be a CUDA tensor.")
        grid_colors_c = grid_colors.contiguous().to(torch.float32)

    iso_c = float(iso)

    vertices, triangles, out_normals, out_colors, _ = marching_tetrahedra_grid_func(
        grid_vertices_c,
        voxels_c,
        voxel_values_c,
        grid_normals_c,
        grid_colors_c,
        iso_c,
        False # return_unique_edges
    )

    return vertices, triangles, out_normals, out_colors
