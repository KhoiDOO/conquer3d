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

def compute_active_voxels(
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    iso: float = 0.0
) -> torch.Tensor:
    """
    Computes the indices of active voxels that intersect the isosurface.
    
    Args:
        voxels (torch.Tensor): An int32 tensor of shape (V, 8) containing corner indices for each voxel.
        sdf (torch.Tensor): A float32 tensor of shape (N,) containing the SDF values.
        iso (float): The isosurface value. Defaults to 0.0.
        
    Returns:
        torch.Tensor: An int64 tensor containing the indices of the active voxels.
    """
    return _C.compute_active_voxels(voxels.contiguous(), sdf.contiguous(), iso)

def voxel2sparse(
    active_indices: torch.Tensor,
    voxels: torch.Tensor,
    idx_grids: torch.Tensor,
    batch_idx: int = 0
) -> torch.Tensor:
    """
    Converts 1D active voxel indices into the [N, 4] coordinates tensor 
    expected by sparse neural networks (like spconv/torchsparse).
    
    Args:
        active_indices (torch.Tensor): 1D tensor of active voxel indices.
        voxels (torch.Tensor): An int32 tensor of shape (V, 8) containing corner indices for each voxel.
        idx_grids (torch.Tensor): An int64 tensor of shape (N, 3) containing the (i, j, k) 3D coordinate indices for each vertex.
        batch_idx (int, optional): The batch index to prepend. Defaults to 0.
        
    Returns:
        torch.Tensor: An int32 tensor of shape (N, 4) in the format [batch_idx, x, y, z].
    """
    active_corners = voxels[active_indices]
    base_vertex_indices = active_corners[:, 0].to(torch.int64)
    voxel_coords = idx_grids[base_vertex_indices]
    
    batch_col = torch.full((voxel_coords.shape[0], 1), batch_idx, device=voxel_coords.device, dtype=torch.int64)
    sparse_coords = torch.cat([batch_col, voxel_coords], dim=1).to(torch.int32)
    return sparse_coords

def sparse2voxel(
    sparse_coords: torch.Tensor,
    res: Union[List[int], Tuple[int, int, int]]
) -> torch.Tensor:
    """
    Converts a [N, 4] sparse coordinates tensor back to 1D active voxel indices.
    
    Args:
        sparse_coords (torch.Tensor): An int32 tensor of shape (N, 4) in the format [batch_idx, x, y, z].
        res (List[int] | Tuple[int, int, int]): The number of vertices along each axis (rx, ry, rz).
        
    Returns:
        torch.Tensor: An int64 tensor containing the indices of the active voxels.
    """
    x = sparse_coords[:, 1].to(torch.int64)
    y = sparse_coords[:, 2].to(torch.int64)
    z = sparse_coords[:, 3].to(torch.int64)
    
    rx, ry, rz = res
    active_indices = x * (ry - 1) * (rz - 1) + y * (rz - 1) + z
    return active_indices
