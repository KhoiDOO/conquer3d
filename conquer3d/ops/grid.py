import torch
from typing import Tuple, List, Union
from .. import _C

def create_voxel_grid(
    grid_min: Union[List[float], Tuple[float, float, float]],
    grid_max: Union[List[float], Tuple[float, float, float]],
    res: Union[List[int], Tuple[int, int, int]],
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a structured 3D voxel grid efficiently.
    
    Args:
        grid_min (List[float] | Tuple[float, float, float]): The minimum (x, y, z) bounding box coordinates.
        grid_max (List[float] | Tuple[float, float, float]): The maximum (x, y, z) bounding box coordinates.
        res (List[int] | Tuple[int, int, int]): The number of vertices along each axis (rx, ry, rz).
        device (str, optional): Target device for the tensors (e.g., "cuda" or "cpu"). Defaults to "cuda".
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - grid_vertices (torch.Tensor): A float32 tensor of shape (N, 3) containing all grid coordinates.
            - voxels (torch.Tensor): An int32 tensor of shape (V, 8) containing corner indices 
                                     for each voxel, mapped identically to the Marching Cubes convention.
    """
    return _C.create_voxel_grid(list(grid_min), list(grid_max), list(res), device)
