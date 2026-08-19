import torch
from typing import Tuple, List, Union
from .. import _C

def create_voxel_grid(
    grid_min: Union[List[float], Tuple[float, float, float]],
    grid_max: Union[List[float], Tuple[float, float, float]],
    res: Union[List[int], Tuple[int, int, int]],
    device: str = "cuda",
    return_idx_grids: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
    """
    Creates a structured 3D voxel grid efficiently.
    
    Args:
        grid_min (List[float] | Tuple[float, float, float]): The minimum (x, y, z) bounding box coordinates.
        grid_max (List[float] | Tuple[float, float, float]): The maximum (x, y, z) bounding box coordinates.
        res (List[int] | Tuple[int, int, int]): The number of vertices along each axis (rx, ry, rz).
        device (str, optional): Target device for the tensors (e.g., "cuda" or "cpu"). Defaults to "cuda".
        return_idx_grids (bool, optional): If True, returns the 3D coordinate indices. Defaults to True.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
            - grid_vertices (torch.Tensor): A float32 tensor of shape (N, 3) containing all grid coordinates.
            - voxels (torch.Tensor): An int32 tensor of shape (V, 8) containing corner indices 
                                     for each voxel, mapped identically to the Marching Cubes convention.
            - idx_grids (torch.Tensor | None): An int64 tensor of shape (N, 3) containing the (i, j, k) 3D coordinate indices for each vertex (or None).
    """
    return _C.create_voxel_grid(list(grid_min), list(grid_max), list(res), device, return_idx_grids)

def create_voxel_grid_from_tmesh(
    grid_min: Union[List[float], Tuple[float, float, float]],
    grid_max: Union[List[float], Tuple[float, float, float]],
    res: Union[List[int], Tuple[int, int, int]],
    tmesh: '_C.TriangleMesh',
    return_unique_vert_ids: bool = True,
    pad: int = 0,
    return_normals: bool = False,
    normal_mode: int = 0
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]],
    Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None], torch.Tensor]
]:
    """
    Creates a sparse 3D voxel grid strictly around the surface with optional surface normals.
    
    Args:
        grid_min (List[float] | Tuple[float, float, float]): Minimum (x, y, z) bounding box coordinates.
        grid_max (List[float] | Tuple[float, float, float]): Maximum (x, y, z) bounding box coordinates.
        res (List[int] | Tuple[int, int, int]): Number of vertices along each axis (rx, ry, rz).
        tmesh (TriangleMesh): Input TriangleMesh data structure.
        return_unique_vert_ids (bool, optional): If True, returns global 1D indices of active vertices (default: True).
        pad (int, optional): Number of voxel layers to dilate the sparse grid (default: 0).
            Set pad=1 when performing Dual Contouring or DMC to guarantee all 4 incident voxels
            are present for every sign-crossing edge.
        return_normals (bool, optional): If True, queries BVH to compute and return (N, 3) surface normals (default: False).
        normal_mode (int, optional): Normal computation mode (default: 0).
            - 0: Closest triangle face normal (preserves sharp CAD creases in Dual Contouring).
            - 1: Barycentric interpolated vertex normal (smooth shading).
            - 2: Normalized displacement vector (spatial SDF gradient).
            
    Returns:
        If return_normals=False:
            Tuple[grid_vertices, voxels, unique_vert_ids]
        If return_normals=True:
            Tuple[grid_vertices, voxels, unique_vert_ids, grid_normals]
    """
    res_tuple = _C.create_voxel_grid_from_tmesh(
        list(grid_min), list(grid_max), list(res), tmesh, return_unique_vert_ids, pad, return_normals, normal_mode
    )
    if return_normals:
        return res_tuple[0], res_tuple[1], res_tuple[2], res_tuple[3]
    return res_tuple[0], res_tuple[1], res_tuple[2]

def get_active_voxel_ids_from_depth(
    depth_image: torch.Tensor,
    c2w: torch.Tensor,
    intrinsics_inv: torch.Tensor,
    grid_min: Union[List[float], Tuple[float, float, float]],
    grid_max: Union[List[float], Tuple[float, float, float]],
    res: Union[List[int], Tuple[int, int, int]],
    activate_neighbor: bool = False,
    trunc_margin: float = 0.0
) -> torch.Tensor:
    """
    Extracts active voxel IDs from a single depth map.
    """
    return _C.get_active_voxel_ids_from_depth(
        depth_image.contiguous(), c2w.contiguous(), intrinsics_inv.contiguous(), 
        list(grid_min), list(grid_max), list(res), activate_neighbor, trunc_margin
    )

def build_sparse_grid_from_active_voxels(
    active_voxel_ids: torch.Tensor,
    grid_min: Union[List[float], Tuple[float, float, float]],
    grid_max: Union[List[float], Tuple[float, float, float]],
    res: Union[List[int], Tuple[int, int, int]],
    return_unique_vert_ids: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
    """
    Builds a sparse 3D voxel grid from unique active voxel IDs.
    """
    return _C.build_sparse_grid_from_active_voxels(
        active_voxel_ids.contiguous(), list(grid_min), list(grid_max), list(res), return_unique_vert_ids
    )

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

def create_random_points_ball(n_points: int, radius: float = 1.0, device: str = 'cuda') -> torch.Tensor:
    """
    Creates a set of random points uniformly distributed inside a sphere.

    Args:
        n_points (int): Number of points to sample.
        radius (float): Radius of the sphere. Defaults to 1.0.
        device (str): Device to place the tensor on. Defaults to 'cuda'.

    Returns:
        torch.Tensor: Random points of shape (n_points, 3).
    """
    points = torch.randn(n_points, 3, device=device)
    radii = torch.rand(n_points, 1, device=device) ** (1/3) * radius
    points = (points / points.norm(dim=-1, keepdim=True)) * radii
    return points