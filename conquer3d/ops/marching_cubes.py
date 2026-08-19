"""Standard Marching Cubes algorithm for 3D isosurface extraction.

This module provides GPU-accelerated classical Marching Cubes (Lorensen & Cline 1987)
for converting discrete 3D scalar fields on structured voxel grids into triangle meshes,
with optional normal and RGB color interpolation.
"""

from typing import Tuple, Optional
import torch

from .._C import marching_cubes as marching_cubes_func


def marching_cubes(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    voxel_values: torch.Tensor,
    grid_normals: Optional[torch.Tensor] = None,
    grid_colors: Optional[torch.Tensor] = None,
    iso: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Executes the classical Marching Cubes algorithm to extract an isosurface mesh.

    Args:
        grid_vertices (torch.Tensor): Global vertex coordinates of shape `(V, 3)`
            with dtype `torch.float32` on CUDA.
        voxels (torch.Tensor): Voxel corner indices of shape `(N, 8)` mapping to
            `grid_vertices` with dtype `torch.int32` on CUDA.
        voxel_values (torch.Tensor): Scalar or SDF values of shape `(V,)` at each
            grid vertex with dtype `torch.float32` on CUDA.
        grid_normals (torch.Tensor, optional): Surface normals of shape `(V, 3)`
            at each vertex on CUDA. Defaults to None.
        grid_colors (torch.Tensor, optional): Feature/RGB colors of shape `(V, 3)`
            at each vertex on CUDA. Defaults to None.
        iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - vertices (torch.Tensor): Extracted mesh surface vertices of shape `(M, 3)`.
            - triangles (torch.Tensor): Extracted triangle face indices of shape `(T, 3)`.
            - normals (torch.Tensor | None): Interpolated vertex normals of shape `(M, 3)`,
              or None if `grid_normals` was not provided.
            - colors (torch.Tensor | None): Interpolated vertex colors of shape `(M, 3)`,
              or None if `grid_colors` was not provided.

    Raises:
        ValueError: If any input tensor is not on a CUDA device.

    Example:
        >>> import torch
        >>> from conquer3d.ops import marching_cubes
        >>> verts, tris, _, _ = marching_cubes(grid_vertices, voxels, voxel_values, iso=0.0)
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

    vertices, triangles, out_normals, out_colors, _ = marching_cubes_func(
        grid_vertices_c,
        voxels_c,
        voxel_values_c,
        grid_normals_c,
        grid_colors_c,
        iso_c,
        False # return_unique_edges
    )

    return vertices, triangles, out_normals, out_colors
