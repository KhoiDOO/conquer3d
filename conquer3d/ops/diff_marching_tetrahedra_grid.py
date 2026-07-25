import torch
from typing import Tuple, Optional

# Explicitly import the compiled CMake target
from .._C import marching_tetrahedra_grid as marching_tetrahedra_grid_func
from .._C import marching_tetrahedra_grid_backward as marching_tetrahedra_grid_backward_func

class DiffMarchingTetrahedraGrid(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grid_vertices: torch.Tensor,
        voxels: torch.Tensor,
        voxel_values: torch.Tensor,
        grid_normals: Optional[torch.Tensor] = None,
        grid_colors: Optional[torch.Tensor] = None,
        iso: float = 0.0
    ):
        # We need return_unique_edges = True for the backward pass
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
        
        # Save tensors needed for the backward pass
        if ctx.has_colors:
            ctx.save_for_backward(unique_edges, grid_vertices, voxel_values, grid_colors)
        else:
            ctx.save_for_backward(unique_edges, grid_vertices, voxel_values)
        
        return out_vertices, out_triangles, out_normals, out_colors

    @staticmethod
    def backward(ctx, grad_out_vertices, grad_out_triangles, grad_out_normals, grad_out_colors):
        if ctx.has_colors:
            unique_edges, grid_vertices, voxel_values, grid_colors = ctx.saved_tensors
        else:
            unique_edges, grid_vertices, voxel_values = ctx.saved_tensors
            grid_colors = None
            
        iso = ctx.iso
        
        # Initialize gradients for inputs
        grad_voxel_values = torch.zeros_like(voxel_values)
        
        grad_grid_colors = None
        if ctx.has_colors:
            grad_grid_colors = torch.zeros_like(grid_colors)
            
        # Call the backward C++ binding if we have valid output vertices
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
            
        # Return gradients for all inputs in the same order as forward
        # grid_vertices, voxels, voxel_values, grid_normals, grid_colors, iso
        return None, None, grad_voxel_values, None, grad_grid_colors, None


def diff_marching_tetrahedra_grid(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    voxel_values: torch.Tensor,
    grid_normals: Optional[torch.Tensor] = None,
    grid_colors: Optional[torch.Tensor] = None,
    iso: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Executes the Differentiable Marching Tetrahedra Grid algorithm to extract an isosurface from a voxel grid.

    Args:
        grid_vertices (torch.Tensor): (V, 3) tensor of global vertex positions.
        voxels (torch.Tensor): (N, 8) tensor of voxel corner indices mapping to `grid_vertices`.
        voxel_values (torch.Tensor): (V,) tensor of SDF/scalar values at each vertex. Requires grad.
        grid_normals (torch.Tensor, optional): (V, 3) optional tensor of SDF normals at each vertex. Defaults to None.
        grid_colors (torch.Tensor, optional): (V, 3) optional tensor of RGB colors at each vertex. Defaults to None. Requires grad.
        iso (float, optional): The isosurface extraction threshold. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - vertices (torch.Tensor): (M, 3) tensor of extracted mesh vertices. Supports gradients.
            - triangles (torch.Tensor): (T, 3) tensor of extracted mesh triangle indices.
            - normals (torch.Tensor, optional): (M, 3) tensor of extracted mesh vertex normals, if `grid_normals` was provided.
            - colors (torch.Tensor, optional): (M, 3) tensor of extracted mesh vertex colors, if `grid_colors` was provided. Supports gradients.
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
