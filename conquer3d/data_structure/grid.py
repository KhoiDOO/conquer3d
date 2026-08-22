"""3D Voxel Grid creation and spatial partitioning data structures.

This module provides high-speed structured and sparse voxel grid generators,
sparse surface voxel extraction from TriangleMesh and depth images, central-difference
grid normal operators, and active voxel filtering for Marching Cubes and Dual Contouring.
"""

from typing import Tuple, List, Union, Optional
import torch
from .. import _C


def create_voxel_grid(
    grid_min: Union[List[float], Tuple[float, float, float]],
    grid_max: Union[List[float], Tuple[float, float, float]],
    res: Union[List[int], Tuple[int, int, int]],
    device: str = "cuda",
    return_idx_grids: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Creates a structured 3D dense voxel grid efficiently on device.
    
    Args:
        grid_min (Union[List[float], Tuple[float, float, float]]): The minimum
            `(x, y, z)` bounding box coordinates.
        grid_max (Union[List[float], Tuple[float, float, float]]): The maximum
            `(x, y, z)` bounding box coordinates.
        res (Union[List[int], Tuple[int, int, int]]): The number of grid vertices
            along each axis `(rx, ry, rz)`.
        device (str, optional): Target compute device (e.g., `"cuda"` or `"cpu"`).
            Defaults to `"cuda"`.
        return_idx_grids (bool, optional): If True, returns the 3D integer coordinate
            indices. Defaults to True.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - grid_vertices (torch.Tensor): A float32 tensor of shape `(N, 3)`
              containing all flattened 3D grid coordinates ($N = rx \\times ry \\times rz$).
            - voxels (torch.Tensor): An int32 tensor of shape `(V, 8)` containing corner
              indices for each voxel cell, mapped identically to the Marching Cubes convention.
            - idx_grids (torch.Tensor | None): An int64 tensor of shape `(N, 3)` containing
              the `(i, j, k)` 3D discrete grid indices, or None if `return_idx_grids=False`.

    Example:
        >>> from conquer3d.data_structure import create_voxel_grid
        >>> verts, voxels, idxs = create_voxel_grid([-1, -1, -1], [1, 1, 1], [64, 64, 64])
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
    normal_mode: int = 0,
    drop_empty_vertex_voxels: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
]:
    """Creates a sparse 3D voxel grid strictly around the surface of a TriangleMesh with optional normals.
    
    Bypasses dense 3D volume allocation by querying the mesh BVH for intersecting voxel cells,
    optionally dilating the active voxel neighborhood, and re-indexing corner vertices.

    Args:
        grid_min (Union[List[float], Tuple[float, float, float]]): Minimum `(x, y, z)` bounding coordinates.
        grid_max (Union[List[float], Tuple[float, float, float]]): Maximum `(x, y, z)` bounding coordinates.
        res (Union[List[int], Tuple[int, int, int]]): Grid resolution `(rx, ry, rz)`.
        tmesh (TriangleMesh): Input TriangleMesh GPU data structure.
        return_unique_vert_ids (bool, optional): If True, returns global 1D indices
            of unique active vertices. Defaults to True.
        pad (int, optional): Number of voxel layers to dilate the sparse grid. Defaults to 0.
            Set `pad=1` when performing Dual Contouring or DMC to guarantee all 4 incident voxels
            are present for every sign-crossing edge.
        return_normals (bool, optional): If True, queries BVH to compute and return `(N, 3)`
            surface normals at sparse vertices. Defaults to False.
        normal_mode (int, optional): Normal computation mode. Defaults to 0.
            - 0: Closest triangle face normal (preserves sharp CAD creases in Dual Contouring).
            - 1: Barycentric interpolated vertex normal (smooth shading).
            - 2: Normalized displacement vector (spatial SDF gradient).
        drop_empty_vertex_voxels (bool, optional): If True, filters out voxels containing no mesh
            vertices inside their 3D bounding box cell. Defaults to False.
            
    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
            - If `return_normals=False`:
                `(sparse_grid_vertices, local_voxels, unique_vert_ids)`
            - If `return_normals=True`:
                `(sparse_grid_vertices, local_voxels, unique_vert_ids, grid_normals)`
    """
    res_tuple = _C.create_voxel_grid_from_tmesh(
        list(grid_min), list(grid_max), list(res), tmesh, return_unique_vert_ids, pad, return_normals, normal_mode, drop_empty_vertex_voxels
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
    """Extracts 1D active voxel linear IDs from an unprojected single-view depth map.

    Args:
        depth_image (torch.Tensor): Depth map tensor of shape `(H, W)` on CUDA.
        c2w (torch.Tensor): Camera-to-World 4x4 extrinsic transformation matrix.
        intrinsics_inv (torch.Tensor): Inverse 3x3 camera intrinsic matrix.
        grid_min (Union[List[float], Tuple[float, float, float]]): Minimum `(x, y, z)` grid bounds.
        grid_max (Union[List[float], Tuple[float, float, float]]): Maximum `(x, y, z)` grid bounds.
        res (Union[List[int], Tuple[int, int, int]]): Resolution `(rx, ry, rz)`.
        activate_neighbor (bool, optional): If True, activates adjacent voxel cells.
            Defaults to False.
        trunc_margin (float, optional): Truncation distance in world units. Defaults to 0.0.

    Returns:
        torch.Tensor: Int64 tensor of unique active 1D voxel indices.
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
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Builds a sparse 3D voxel grid and local corner indexing from unique active voxel IDs.

    Args:
        active_voxel_ids (torch.Tensor): 1D int64 tensor of active voxel linear IDs.
        grid_min (Union[List[float], Tuple[float, float, float]]): Global minimum `(x, y, z)` bounds.
        grid_max (Union[List[float], Tuple[float, float, float]]): Global maximum `(x, y, z)` bounds.
        res (Union[List[int], Tuple[int, int, int]]): Global grid resolution `(rx, ry, rz)`.
        return_unique_vert_ids (bool, optional): If True, returns global 1D vertex indices.
            Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - sparse_grid_vertices (torch.Tensor): Float32 tensor of shape `(N, 3)` with coordinates.
            - local_voxels (torch.Tensor): Int32 tensor of shape `(M, 8)` mapping to sparse vertices.
            - unique_vert_ids (torch.Tensor | None): Global linear indices of vertices.
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
    """Computes smooth normal vectors (gradients) for a voxel grid using central differences.
    
    Args:
        sdf (torch.Tensor): A float32 tensor of shape `(N,)` containing scalar SDF values.
        grid_vertices (torch.Tensor): A float32 tensor of shape `(N, 3)` containing grid coordinates.
        idx_grids (torch.Tensor): An int64 tensor of shape `(N, 3)` containing the `(i, j, k)` indices.
        res (Union[List[int], Tuple[int, int, int]]): Number of vertices along each axis `(rx, ry, rz)`.
        
    Returns:
        torch.Tensor: A float32 tensor of shape `(N, 3)` containing normalized unit gradient vectors.
    """
    return _C.compute_grid_normal(sdf.contiguous(), grid_vertices.contiguous(), idx_grids.contiguous(), list(res))


def compute_active_voxels(
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    iso: float = 0.0
) -> torch.Tensor:
    """Computes indices of active voxels whose corners cross the isosurface threshold.
    
    A voxel cell is active if at least one corner value is $< \\text{iso}$ and at least
    one corner value is $\\ge \\text{iso}$.

    Args:
        voxels (torch.Tensor): An int32 tensor of shape `(V, 8)` containing corner indices.
        sdf (torch.Tensor): A float32 tensor of shape `(N,)` containing corner SDF values.
        iso (float, optional): The isosurface threshold value. Defaults to 0.0.
        
    Returns:
        torch.Tensor: An int64 1D tensor containing the indices of active intersecting voxels.
    """
    return _C.compute_active_voxels(voxels.contiguous(), sdf.contiguous(), iso)


def create_random_points_ball(
    n_points: int,
    radius: float = 1.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """Samples random 3D points uniformly distributed inside a solid sphere.

    Args:
        n_points (int): Number of points to sample.
        radius (float, optional): Radius of the sphere. Defaults to 1.0.
        device (str, optional): Target device for generated tensor. Defaults to `'cuda'`.

    Returns:
        torch.Tensor: Float32 tensor of shape `(n_points, 3)` containing uniform ball coordinates.
    """
    points = torch.randn(n_points, 3, device=device)
    radii = torch.rand(n_points, 1, device=device) ** (1/3) * radius
    points = (points / points.norm(dim=-1, keepdim=True)) * radii
    return points