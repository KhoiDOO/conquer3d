"""Differentiable Marching Tetrahedra Grid (DiffMTG) autograd operator.

This module provides end-to-end differentiable Marching Tetrahedra on structured
cubic grids with analytical gradient backpropagation through extracted edge zero-crossings.
"""

from typing import Tuple, Optional
import torch

from .._C import marching_tetrahedra_grid as marching_tetrahedra_grid_func
from .._C import marching_tetrahedra_grid_backward as marching_tetrahedra_grid_backward_func


class DiffMarchingTetrahedraGrid(torch.autograd.Function):
    """Differentiable Marching Tetrahedra Grid autograd Function."""

    @staticmethod
    def forward(
        ctx,
        grid_vertices: torch.Tensor,
        voxels: torch.Tensor,
        voxel_values: torch.Tensor,
        grid_normals: Optional[torch.Tensor] = None,
        grid_colors: Optional[torch.Tensor] = None,
        iso: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass of Differentiable Marching Tetrahedra Grid.

        Args:
            ctx: Autograd context.
            grid_vertices (torch.Tensor): Global vertex coordinates `(V, 3)` on CUDA.
            voxels (torch.Tensor): Voxel corner indices `(N, 8)` on CUDA.
            voxel_values (torch.Tensor): Scalar values `(V,)` at each grid vertex on CUDA.
            grid_normals (torch.Tensor, optional): Vertex normals `(V, 3)`. Defaults to None.
            grid_colors (torch.Tensor, optional): Feature colors `(V, C)`. Defaults to None.
            iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                - out_vertices: (M, 3) float32 surface vertex positions.
                - out_triangles: (T, 3) int32 triangle indices.
                - out_normals: (M, 3) interpolated normals, or None.
                - out_colors: (M, C) interpolated colors, or None.
        """
        out_vertices, out_triangles, out_normals, out_colors, unique_edges = marching_tetrahedra_grid_func(
            grid_vertices,
            voxels,
            voxel_values,
            grid_normals,
            grid_colors,
            iso,
            True # return_unique_edges
        )
        
        ctx.iso = iso
        ctx.has_colors = grid_colors is not None
        
        if ctx.has_colors:
            ctx.save_for_backward(unique_edges, grid_vertices, voxel_values, grid_colors)
        else:
            ctx.save_for_backward(unique_edges, grid_vertices, voxel_values)
        
        return out_vertices, out_triangles, out_normals, out_colors

    @staticmethod
    def backward(ctx, grad_out_vertices, grad_out_triangles, grad_out_normals, grad_out_colors):
        """Backward pass evaluating gradients w.r.t. scalar field values and colors."""
        if ctx.has_colors:
            unique_edges, grid_vertices, voxel_values, grid_colors = ctx.saved_tensors
        else:
            unique_edges, grid_vertices, voxel_values = ctx.saved_tensors
            grid_colors = None
            
        iso = ctx.iso
        grad_voxel_values = torch.zeros_like(voxel_values)
        grad_grid_colors = None
        if ctx.has_colors:
            grad_grid_colors = torch.zeros_like(grid_colors)
            
        if unique_edges is not None and unique_edges.shape[0] > 0 and grad_out_vertices is not None:
            grad_out_vertices = grad_out_vertices.contiguous()
            if grad_out_colors is not None:
                grad_out_colors = grad_out_colors.contiguous()
                
            marching_tetrahedra_grid_backward_func(
                unique_edges,
                grid_vertices,
                grid_colors,
                voxel_values,
                grad_out_vertices,
                grad_out_colors,
                grad_voxel_values,
                grad_grid_colors,
                iso
            )
            
        return None, None, grad_voxel_values, None, grad_grid_colors, None


def diff_marching_tetrahedra_grid(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    voxel_values: torch.Tensor,
    grid_normals: Optional[torch.Tensor] = None,
    grid_colors: Optional[torch.Tensor] = None,
    iso: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Differentiable Marching Tetrahedra Grid surface extraction with PyTorch autograd integration.

    Args:
        grid_vertices (torch.Tensor): Global vertex positions `(V, 3)` on CUDA.
        voxels (torch.Tensor): Voxel corner indices `(N, 8)` mapping to `grid_vertices`.
        voxel_values (torch.Tensor): Scalar values `(V,)` requiring gradient.
        grid_normals (torch.Tensor, optional): Optional normals `(V, 3)`. Defaults to None.
        grid_colors (torch.Tensor, optional): Optional color features `(V, C)` requiring gradient.
        iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - vertices (torch.Tensor): Extracted surface vertices `(M, 3)`.
            - triangles (torch.Tensor): Extracted triangle indices `(T, 3)`.
            - normals (torch.Tensor | None): Interpolated normals `(M, 3)`, or None.
            - colors (torch.Tensor | None): Interpolated colors `(M, C)`, or None.
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

    return DiffMarchingTetrahedraGrid.apply(
        grid_vertices_c,
        voxels_c,
        voxel_values_c,
        grid_normals_c,
        grid_colors_c,
        iso_c
    )
