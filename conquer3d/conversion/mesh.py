"""Mesh batching and sparse volumetric conversion utilities.

This module provides conversion functions from raw mesh vertices and `BTriangleMesh`
batched structures to dense or sparse coordinate tensors and corner SDF representations.
"""

from typing import Union, List, Tuple, Optional
import torch
from conquer3d.data_structure.grid import create_voxel_grid, compute_active_voxels
from conquer3d.conversion.grid import voxel2sparse
from conquer3d.data_structure import TriangleMesh
from conquer3d.data_structure.bmesh import BTriangleMesh


def mesh2voxel(
    vertices: torch.Tensor,
    res: Union[List[int], Tuple[int, int, int]],
    grid_bound: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float], List[float]]:
    """Computes bounding extents of a mesh and generates a structured voxel grid.

    Args:
        vertices (torch.Tensor): Vertex coordinate tensor of shape `(V, 3)`.
        res (Union[List[int], Tuple[int, int, int]]): Resolution along each axis `(rx, ry, rz)`.
        grid_bound (float, optional): If specified, uses symmetric bounds `[-grid_bound, grid_bound]^3`.
            If None, tight bounding box of `vertices` is computed. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float], List[float]]:
            - grid_vertices (torch.Tensor): Float32 tensor of shape `(N, 3)` with grid coordinates.
            - voxels (torch.Tensor): Int32 tensor of shape `(M, 8)` with voxel corner indices.
            - idx_grids (torch.Tensor): Int64 tensor of shape `(N, 3)` with discrete indices.
            - grid_min (List[float]): Lower bounding coordinates `[x_min, y_min, z_min]`.
            - grid_max (List[float]): Upper bounding coordinates `[x_max, y_max, z_max]`.
    """
    if grid_bound is None:
        grid_min = vertices.min(dim=0)[0].tolist()
        grid_max = vertices.max(dim=0)[0].tolist()
    else:
        grid_min = [-grid_bound, -grid_bound, -grid_bound]
        grid_max = [grid_bound, grid_bound, grid_bound]
        
    device_str = str(vertices.device)
    if device_str.startswith('cuda'):
        # Fix format if there's an index like cuda:0
        device_str = "cuda"
        
    grid_vertices, voxels, idx_grids = create_voxel_grid(grid_min, grid_max, res, device=device_str)
    
    # Ensure they are on the exact same device as vertices (especially if multiple GPUs)
    grid_vertices = grid_vertices.to(vertices.device)
    voxels = voxels.to(vertices.device)
    idx_grids = idx_grids.to(vertices.device)
    
    return grid_vertices, voxels, idx_grids, grid_min, grid_max


def mesh2voxel_sdf(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    grid_vertices: torch.Tensor,
    sign_mode: int = 2
) -> torch.Tensor:
    """Computes Signed Distance Field (SDF) values at grid vertices against a mesh.

    Args:
        vertices (torch.Tensor): Vertex coordinate tensor of shape `(V, 3)`.
        faces (torch.Tensor): Face indices tensor of shape `(F, 3)`.
        grid_vertices (torch.Tensor): Query grid vertex coordinates of shape `(N, 3)`.
        sign_mode (int, optional): Sign evaluation mode:
            - 0: Ray parity casting.
            - 1: Fast Winding Number (FWN).
            - 2: Angle-weighted pseudonormals (default).
            - 3: Volumetric flood fill mask.

    Returns:
        torch.Tensor: Float32 tensor of shape `(N,)` containing signed distances.
    """
    mesh = TriangleMesh(vertices, faces.to(torch.int32))
    
    # query_points returns: q_ids, tri_ids, prj_pts, sdf
    _, _, _, sdf = mesh.query_points(
        grid_vertices,
        return_sdf=True,
        return_prj_pts=False,
        sign_mode=sign_mode,
        distance_mode=0
    )
    return sdf


def mesh2sparse(
    bmesh: BTriangleMesh,
    res: Union[List[int], Tuple[int, int, int]],
    grid_bound: float = 1.2,
    iso: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts a batched mesh container `BTriangleMesh` into sparse coordinates and corner SDFs.

    Args:
        bmesh (BTriangleMesh): Batched mesh container holding concatenated geometry.
        res (Union[List[int], Tuple[int, int, int]]): Grid resolution `(rx, ry, rz)`.
        grid_bound (float, optional): Symmetric boundary extent `[-grid_bound, grid_bound]^3`.
            Defaults to 1.2.
        iso (float, optional): Isosurface boundary threshold. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - sparse_coords (torch.Tensor): Int32 tensor of shape `(Total_Active, 4)` in `(batch_idx, x, y, z)`.
            - sparse_sdfs (torch.Tensor): Float32 tensor of shape `(Total_Active, 8)` with 8 corner SDFs per voxel.
    """
    all_coords = []
    all_sdfs = []
    
    for b in range(bmesh.batch_size):
        # Extract individual mesh
        v_mask = bmesh.vertbids == b
        f_mask = bmesh.facebids == b
        
        v = bmesh.vertices[v_mask]
        f = bmesh.faces[f_mask]
        
        if len(v) == 0:
            continue
            
        # Voxelize and Compute SDF
        grid_vertices, voxels, idx_grids, _, _ = mesh2voxel(v, res, grid_bound)
        sdf = mesh2voxel_sdf(v, f, grid_vertices)
        
        # Sparse conversion
        active_voxels = compute_active_voxels(voxels, sdf, iso)
        sparse_coords, sparse_sdfs = voxel2sparse(active_voxels, voxels, idx_grids, sdf=sdf, batch_idx=b)
        
        all_coords.append(sparse_coords)
        all_sdfs.append(sparse_sdfs)
        
    if len(all_coords) > 0:
        return torch.cat(all_coords, dim=0), torch.cat(all_sdfs, dim=0)
    else:
        # Fallback for empty batch
        device = bmesh.vertices.device
        return torch.empty((0, 4), dtype=torch.int32, device=device), torch.empty((0, 8), dtype=torch.float32, device=device)


def mesh2sparse_with_dense(
    bmesh: BTriangleMesh,
    res: Union[List[int], Tuple[int, int, int]],
    grid_bound: float = 1.2,
    iso: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched conversion of `BTriangleMesh` returning both sparse surface voxels and full dense grids.

    Args:
        bmesh (BTriangleMesh): Batched mesh container.
        res (Union[List[int], Tuple[int, int, int]]): Resolution `(rx, ry, rz)`.
        grid_bound (float, optional): Symmetric boundary limit. Defaults to 1.2.
        iso (float, optional): Isosurface threshold. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - sparse_coords (torch.Tensor): `(Total_Active, 4)` active sparse voxel coordinates.
            - sparse_sdfs (torch.Tensor): `(Total_Active, 8)` corner SDFs of active voxels.
            - dense_coords (torch.Tensor): `(Total_Dense, 4)` all grid voxel coordinates.
            - dense_sdfs (torch.Tensor): `(Total_Dense, 8)` corner SDFs of all grid voxels.
    """
    all_coords = []
    all_sdfs = []
    all_dense_coords = []
    all_dense_sdfs = []
    
    for b in range(bmesh.batch_size):
        v_mask = bmesh.vertbids == b
        f_mask = bmesh.facebids == b
        
        v = bmesh.vertices[v_mask]
        f = bmesh.faces[f_mask]
        
        if len(v) == 0:
            continue
            
        grid_vertices, voxels, idx_grids, _, _ = mesh2voxel(v, res, grid_bound)
        sdf = mesh2voxel_sdf(v, f, grid_vertices)
        
        active_voxels = compute_active_voxels(voxels, sdf, iso)
        sparse_coords, sparse_sdfs = voxel2sparse(active_voxels, voxels, idx_grids, sdf=sdf, batch_idx=b)
        dense_coords, dense_sdfs = voxel2sparse(torch.ones_like(voxels[:, 0], dtype=torch.bool), voxels, idx_grids, sdf=sdf, batch_idx=b)
        
        all_coords.append(sparse_coords)
        all_sdfs.append(sparse_sdfs)
        all_dense_coords.append(dense_coords)
        all_dense_sdfs.append(dense_sdfs)
        
    if len(all_coords) > 0:
        return torch.cat(all_coords, dim=0), torch.cat(all_sdfs, dim=0), torch.cat(all_dense_coords, dim=0), torch.cat(all_dense_sdfs, dim=0)
    else:
        device = bmesh.vertices.device
        return torch.empty((0, 4), dtype=torch.int32, device=device), torch.empty((0, 8), dtype=torch.float32, device=device), torch.empty((0, 4), dtype=torch.int32, device=device), torch.empty((0, 8), dtype=torch.float32, device=device)
