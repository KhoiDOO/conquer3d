import torch
from typing import Tuple

# Explicitly import the compiled CMake target
from .._C import marching_cubes as marching_cubes_func

def marching_cubes(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    voxel_values: torch.Tensor,
    iso: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Executes the Marching Cubes algorithm to extract an isosurface from a voxel grid.

    Args:
        grid_vertices (torch.Tensor): (V, 3) tensor of global vertex positions.
        voxels (torch.Tensor): (N, 8) tensor of voxel corner indices mapping to `grid_vertices`.
        voxel_values (torch.Tensor): (V,) tensor of SDF/scalar values at each vertex.
        iso (float, optional): The isosurface extraction threshold. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - vertices (torch.Tensor): (M, 3) tensor of extracted mesh vertices.
            - triangles (torch.Tensor): (T, 3) tensor of extracted mesh triangle indices.
    """
    if not all(t.is_cuda for t in [grid_vertices, voxels, voxel_values]):
        raise ValueError("All input tensors must be CUDA tensors.")
        
    grid_vertices_c = grid_vertices.contiguous().to(torch.float32)
    voxels_c = voxels.contiguous().to(torch.int32)
    voxel_values_c = voxel_values.contiguous().to(torch.float32)
    iso_c = float(iso)

    vertices, triangles = marching_cubes_func(
        grid_vertices_c,
        voxels_c,
        voxel_values_c,
        iso_c
    )

    return vertices, triangles
