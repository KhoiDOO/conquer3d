import torch
from typing import Tuple, List, Union

def voxel2sparse(
    active_indices: torch.Tensor,
    voxels: torch.Tensor,
    idx_grids: torch.Tensor,
    sdf: torch.Tensor = None,
    batch_idx: int = 0
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Converts 1D active voxel indices into the [N, 4] coordinates tensor 
    expected by sparse neural networks (like spconv/torchsparse).
    
    Args:
        active_indices (torch.Tensor): 1D tensor of active voxel indices.
        voxels (torch.Tensor): An int32 tensor of shape (V, 8) containing corner indices for each voxel.
        idx_grids (torch.Tensor): An int64 tensor of shape (N, 3) containing the (i, j, k) 3D coordinate indices for each vertex.
        sdf (torch.Tensor, optional): A float32 tensor of shape (N,) containing the SDF values. If provided, the function will also extract the 8 corner SDFs for each active voxel. Defaults to None.
        batch_idx (int, optional): The batch index to prepend. Defaults to 0.
        
    Returns:
        If sdf is None:
            torch.Tensor: An int32 tensor of shape (N, 4) in the format [batch_idx, x, y, z].
        If sdf is provided:
            Tuple[torch.Tensor, torch.Tensor]: 
                - sparse_coords (torch.Tensor): [N, 4] tensor
                - sparse_sdfs (torch.Tensor): [N, 8] tensor containing the 8 corner SDF values for each voxel.
    """
    active_corners = voxels[active_indices]
    base_vertex_indices = active_corners[:, 0].to(torch.int64)
    voxel_coords = idx_grids[base_vertex_indices]
    
    batch_col = torch.full((voxel_coords.shape[0], 1), batch_idx, device=voxel_coords.device, dtype=torch.int64)
    sparse_coords = torch.cat([batch_col, voxel_coords], dim=1).to(torch.int32)
    
    if sdf is not None:
        sparse_sdfs = sdf[active_corners.to(torch.int64)]
        return sparse_coords, sparse_sdfs
    
    return sparse_coords

def sparse2voxel(
    sparse_coords: torch.Tensor,
    sparse_sdfs: torch.Tensor,
    grid_min: Union[List[float], Tuple[float, float, float]],
    grid_max: Union[List[float], Tuple[float, float, float]],
    res: Union[List[int], Tuple[int, int, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts pure voxel-centric sparse representations back into a localized, 
    dense-free mesh topology ready for marching cubes.
    
    Args:
        sparse_coords (torch.Tensor): [N, 4] tensor of voxel coords [batch_idx, x, y, z].
        sparse_sdfs (torch.Tensor): [N, 8] tensor of corner SDFs.
        grid_min (List[float]): Global minimum (x, y, z) bounds.
        grid_max (List[float]): Global maximum (x, y, z) bounds.
        res (List[int]): Global resolution (rx, ry, rz).
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - unique_vertices (torch.Tensor): [M, 3] float coordinates of all unique mesh vertices.
            - local_voxels (torch.Tensor): [N, 8] int tensor of corner indices mapping to unique_vertices.
            - merged_sdfs (torch.Tensor): [M,] float tensor of averaged SDF values at unique_vertices.
    """
    device = sparse_coords.device
    
    # 1. Base integer coordinates of the voxels
    base_coords = sparse_coords[:, 1:] # [N, 3]
    
    # 2. Local offsets for the 8 corners of a cube
    cube_corners = torch.tensor([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], 
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=torch.int32, device=device)
    
    # 3. Compute all corner integer coordinates [N, 8, 3]
    all_corners = base_coords.unsqueeze(1) + cube_corners.unsqueeze(0)
    all_corners_flat = all_corners.reshape(-1, 3) # [N*8, 3]
    
    # 4. Find the unique vertices to form our local mesh graph
    unique_corners_int, inverse_indices = torch.unique(all_corners_flat, dim=0, return_inverse=True)
    local_voxels = inverse_indices.reshape(-1, 8).to(torch.int32) # [N, 8]
    
    # 5. Average the SDFs at the overlapping corners
    num_unique = unique_corners_int.shape[0]
    merged_sdfs = torch.zeros(num_unique, dtype=sparse_sdfs.dtype, device=device)
    counts = torch.zeros(num_unique, dtype=sparse_sdfs.dtype, device=device)
    
    flat_inverse = inverse_indices # [N*8]
    flat_sdfs = sparse_sdfs.flatten() # [N*8]
    
    merged_sdfs.scatter_add_(0, flat_inverse, flat_sdfs)
    counts.scatter_add_(0, flat_inverse, torch.ones_like(flat_sdfs))
    merged_sdfs = merged_sdfs / counts
    
    # 6. Convert integer coordinates back to global float positions
    rx, ry, rz = res
    x_coords = torch.linspace(grid_min[0], grid_max[0], rx + 1, device=device, dtype=torch.float32)
    y_coords = torch.linspace(grid_min[1], grid_max[1], ry + 1, device=device, dtype=torch.float32)
    z_coords = torch.linspace(grid_min[2], grid_max[2], rz + 1, device=device, dtype=torch.float32)
    
    ux = x_coords[unique_corners_int[:, 0]]
    uy = y_coords[unique_corners_int[:, 1]]
    uz = z_coords[unique_corners_int[:, 2]]
    
    unique_vertices = torch.stack([ux, uy, uz], dim=1)
    
    return unique_vertices, local_voxels, merged_sdfs

def sparse_coo2dense_occ(
    sparse_coords: torch.Tensor,
    batch_size: int,
    res: Union[List[int], Tuple[int, int, int]]
) -> torch.Tensor:
    """
    Converts a [N, 4] sparse coordinates tensor into a [B, 1, rx, ry, rz] dense occupancy grid.
    
    Args:
        sparse_coords (torch.Tensor): An int32 tensor of shape (N, 4) in the format [batch_idx, x, y, z].
        batch_size (int): The batch size B.
        res (List[int] | Tuple[int, int, int]): The grid resolution (rx, ry, rz).
        
    Returns:
        torch.Tensor: A float32 tensor of shape [B, 1, rx, ry, rz] containing 1.0 at active voxels and 0.0 otherwise.
    """
    rx, ry, rz = res
    dense_occ = torch.zeros((batch_size, 1, rx, ry, rz), dtype=torch.float32, device=sparse_coords.device)
    
    b = sparse_coords[:, 0].to(torch.int64)
    x = sparse_coords[:, 1].to(torch.int64)
    y = sparse_coords[:, 2].to(torch.int64)
    z = sparse_coords[:, 3].to(torch.int64)
    
    dense_occ[b, 0, x, y, z] = 1.0
    return dense_occ

def dense_occ2sparse_coo(
    dense_occ: torch.Tensor,
    threshold: float = 0.0
) -> torch.Tensor:
    """
    Converts a [B, C, rx, ry, rz] dense occupancy grid back to a [N, 4] sparse coordinates tensor.
    
    Args:
        dense_occ (torch.Tensor): A tensor of shape [B, C, rx, ry, rz].
        
    Returns:
        torch.Tensor: An int32 tensor of shape (N, 4) in the format [batch_idx, x, y, z].
    """
    active_mask = dense_occ > threshold
    indices = torch.argwhere(active_mask)
    # indices has shape [N, 5] for [b, c, x, y, z]. We want [batch_idx, x, y, z]
    sparse_coords = indices[:, [0, 2, 3, 4]].to(torch.int32)
    return sparse_coords