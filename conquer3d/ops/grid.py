import torch
from typing import Tuple, List, Union
from .. import _C

def create_voxel_grid(
    grid_min: Union[List[float], Tuple[float, float, float]],
    grid_max: Union[List[float], Tuple[float, float, float]],
    res: Union[List[int], Tuple[int, int, int]],
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates a structured 3D voxel grid efficiently.
    
    Args:
        grid_min (List[float] | Tuple[float, float, float]): The minimum (x, y, z) bounding box coordinates.
        grid_max (List[float] | Tuple[float, float, float]): The maximum (x, y, z) bounding box coordinates.
        res (List[int] | Tuple[int, int, int]): The number of vertices along each axis (rx, ry, rz).
        device (str, optional): Target device for the tensors (e.g., "cuda" or "cpu"). Defaults to "cuda".
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - grid_vertices (torch.Tensor): A float32 tensor of shape (N, 3) containing all grid coordinates.
            - voxels (torch.Tensor): An int32 tensor of shape (V, 8) containing corner indices 
                                     for each voxel, mapped identically to the Marching Cubes convention.
            - idx_grids (torch.Tensor): An int64 tensor of shape (N, 3) containing the (i, j, k) 3D coordinate indices for each vertex.
    """
    return _C.create_voxel_grid(list(grid_min), list(grid_max), list(res), device)

def compute_grid_normal(
    sdf: torch.Tensor,
    grid_vertices: torch.Tensor,
    idx_grids: torch.Tensor,
    res: Union[List[int], Tuple[int, int, int]]
) -> torch.Tensor:
    """
    Computes smooth normals (gradients) for a voxel grid using central differences.
    
    Args:
        sdf (torch.Tensor): A float32 tensor of shape (N,) containing the SDF values.
        grid_vertices (torch.Tensor): A float32 tensor of shape (N, 3) containing the grid coordinates.
        idx_grids (torch.Tensor): An int64 tensor of shape (N, 3) containing the (i, j, k) indices, returned by create_voxel_grid.
        res (List[int] | Tuple[int, int, int]): The number of vertices along each axis (rx, ry, rz).
        
    Returns:
        torch.Tensor: A float32 tensor of shape (N, 3) containing the normalized gradient vectors.
    """
    return _C.compute_grid_normal(sdf.contiguous(), grid_vertices.contiguous(), idx_grids.contiguous(), list(res))
