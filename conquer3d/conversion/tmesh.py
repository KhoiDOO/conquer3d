"""Conversion routines between TriangleMesh representations and volumetric grids.

This module provides high-throughput pipelines converting 3D TriangleMesh
objects into dense or sparse voxel grids and evaluating Signed Distance Fields (SDF)
via GPU-accelerated BVH queries, pseudonormals, or volumetric flood filling.
"""

from typing import Tuple, List, Union, Optional, Any
import torch
import tqdm
from conquer3d.data_structure.grid import create_voxel_grid
from conquer3d.data_structure import create_voxel_grid_from_tmesh


def tmesh2voxel(
    tm: Any,
    res: Union[int, List[int], Tuple[int, int, int]],
    grid_min: Optional[List[float]] = None,
    grid_max: Optional[List[float]] = None,
    chunk_size: int = 5000000,
    device: str = 'cuda',
    show_progress: bool = True,
    sign_mode: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constructs a dense voxel grid from a TriangleMesh and evaluates its Signed Distance Field.

    Args:
        tm (TriangleMesh): Input TriangleMesh GPU data structure.
        res (Union[int, List[int], Tuple[int, int, int]]): Resolution along each axis.
        grid_min (List[float], optional): Minimum `(x, y, z)` bounding coordinates.
            Defaults to `[-1.0, -1.0, -1.0]`.
        grid_max (List[float], optional): Maximum `(x, y, z)` bounding coordinates.
            Defaults to `[1.0, 1.0, 1.0]`.
        chunk_size (int, optional): Batch size for parallel SDF point queries. Defaults to 5,000,000.
        device (str, optional): Target compute device. Defaults to `'cuda'`.
        show_progress (bool, optional): If True, renders a tqdm progress bar. Defaults to True.
        sign_mode (int, optional): SDF sign evaluation strategy:
            - 0: Ray parity casting.
            - 1: Fast Winding Number (FWN).
            - 2: Angle-weighted pseudonormals (default).
            - 3: Volumetric 3D flood fill mask (dense).
            - 4: Hybrid WN + pseudonormals.
            - 5: Coarse-to-Fine (CF) Hierarchical Volumetric Flood Fill (< 10 MB VRAM).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - grid_vertices (torch.Tensor): Dense float32 coordinates of shape `(N, 3)`.
            - voxels (torch.Tensor): Int32 tensor of shape `(V, 8)` containing voxel corner indices.
            - idx_grids (torch.Tensor): Int64 tensor of shape `(N, 3)` containing discrete 3D indices.
            - sdfs (torch.Tensor): Float32 tensor of shape `(N,)` with evaluated SDF values.
    """
    if grid_min is None:
        grid_min = [-1.0, -1.0, -1.0]
    if grid_max is None:
        grid_max = [1.0, 1.0, 1.0]
        
    if isinstance(res, int):
        res_list = [res, res, res]
    else:
        res_list = list(res)
    grid_vertices, voxels, idx_grids = create_voxel_grid(grid_min, grid_max, res_list, device=device)
    
    if sign_mode == 3:
        tm.build_flood_fill_data(grid_min, grid_max, res_list)
    elif sign_mode == 5:
        tm.build_flood_fill_cf_data(grid_min, grid_max, res_list)
    if sign_mode in [2, 4]:
        tm.compute_triangle_normals()
        tm.compute_vertex_normals(1)
        tm.compute_edge_normals()
    
    num_points = grid_vertices.shape[0]
    sdfs = torch.empty(num_points, dtype=torch.float32, device=device)
    
    iterator = range(0, num_points, chunk_size)
    if show_progress:
        iterator = tqdm.tqdm(iterator, desc="Computing SDF")
        
    for i in iterator:
        end = min(i + chunk_size, num_points)
        chunk_points = grid_vertices[i:end]
        _, _, _, chunk_sdf = tm.query_points(chunk_points, return_sdf=True, return_prj_pts=False, sign_mode=sign_mode)
        sdfs[i:end] = chunk_sdf
        
    return grid_vertices, voxels, idx_grids, sdfs


def tmesh2sparse(
    tm: Any,
    res: Union[int, List[int], Tuple[int, int, int]],
    grid_min: Optional[List[float]] = None,
    grid_max: Optional[List[float]] = None,
    chunk_size: int = 5000000,
    iso: float = 0.0,
    device: str = 'cuda',
    show_progress: bool = True,
    sign_mode: int = 2,
    pad: int = 0,
    return_normals: bool = False,
    normal_mode: int = 0,
    drop_empty_vertex_voxels: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """Computes Signed Distance Fields on sparse voxel grids strictly near the surface.

    Completely bypasses dense $O(R^3)$ volume allocation by querying the mesh BVH,
    re-indexing sparse active vertices, and evaluating SDF solely on active points.

    Args:
        tm (TriangleMesh): Input TriangleMesh GPU object.
        res (Union[int, List[int], Tuple[int, int, int]]): Resolution along each axis.
        grid_min (List[float], optional): Minimum `(x, y, z)` bounds. Defaults to `[-1.0, -1.0, -1.0]`.
        grid_max (List[float], optional): Maximum `(x, y, z)` bounds. Defaults to `[1.0, 1.0, 1.0]`.
        chunk_size (int, optional): Chunk size for batch SDF querying. Defaults to 5,000,000.
        iso (float, optional): Isolevel threshold. Defaults to 0.0.
        device (str, optional): Computation device. Defaults to `'cuda'`.
        show_progress (bool, optional): Whether to display a progress bar. Defaults to True.
        sign_mode (int, optional): Sign evaluation mode:
            - 0: Ray casting.
            - 1: Fast Winding Number.
            - 2: Pseudonormals (default).
            - 3: Volumetric flood fill (dense).
            - 4: Hybrid WN + pseudonormals.
            - 5: Coarse-to-Fine (CF) Hierarchical Volumetric Flood Fill (< 10 MB VRAM).
        pad (int, optional): Voxel layer dilation radius. Defaults to 0 (set `pad=1` for DMC / DC).
        return_normals (bool, optional): If True, returns surface normal vectors. Defaults to False.
        normal_mode (int, optional): Normal mode (0: face normals, 1: vertex normals, 2: displacement vector).
        drop_empty_vertex_voxels (bool, optional): If True, drops voxels containing no mesh vertices inside their bounding box. Defaults to False.

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
            - If `return_normals=False`: `(sparse_grid_vertices, active_voxels, sdfs)`
            - If `return_normals=True`: `(sparse_grid_vertices, active_voxels, sdfs, grid_normals)`
    """
    if grid_min is None:
        grid_min = [-1.0, -1.0, -1.0]
    if grid_max is None:
        grid_max = [1.0, 1.0, 1.0]
        
    if isinstance(res, int):
        res_list = [res, res, res]
    else:
        res_list = list(res)
    
    if tm.bvh is None:
        tm.build_bvh()
        
    if return_normals:
        grid_vertices, active_voxels, unique_vert_ids, grid_normals = create_voxel_grid_from_tmesh(
            grid_min, grid_max, res_list, tm, pad=pad, return_normals=True, normal_mode=normal_mode, drop_empty_vertex_voxels=drop_empty_vertex_voxels
        )
    else:
        grid_vertices, active_voxels, unique_vert_ids = create_voxel_grid_from_tmesh(
            grid_min, grid_max, res_list, tm, pad=pad, return_normals=False, drop_empty_vertex_voxels=drop_empty_vertex_voxels
        )
    
    num_points = grid_vertices.shape[0]
    if num_points == 0:
        sdfs = torch.empty(0, dtype=torch.float32, device=device)
        if return_normals:
            return grid_vertices, active_voxels.to(torch.int64), sdfs, grid_normals
        return grid_vertices, active_voxels.to(torch.int64), sdfs

    if sign_mode == 3:
        tm.build_flood_fill_data(grid_min, grid_max, res_list)
    elif sign_mode == 5:
        tm.build_flood_fill_cf_data(grid_min, grid_max, res_list)
    if sign_mode in [2, 4]:
        tm.compute_triangle_normals()
        tm.compute_vertex_normals(1)
        tm.compute_edge_normals()
    
    sdfs = torch.empty(num_points, dtype=torch.float32, device=device)
    
    iterator = range(0, num_points, chunk_size)
    if show_progress:
        iterator = tqdm.tqdm(iterator, desc="Computing Sparse SDF")
        
    for i in iterator:
        end = min(i + chunk_size, num_points)
        chunk_points = grid_vertices[i:end]
        _, _, _, chunk_sdf = tm.query_points(chunk_points, return_sdf=True, return_prj_pts=False, sign_mode=sign_mode)
        sdfs[i:end] = chunk_sdf
        
    if return_normals:
        return grid_vertices, active_voxels.to(torch.int64), sdfs, grid_normals
    return grid_vertices, active_voxels.to(torch.int64), sdfs
