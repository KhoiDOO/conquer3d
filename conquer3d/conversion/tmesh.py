import torch
import tqdm
from conquer3d.data_structure.grid import create_voxel_grid, compute_active_voxels

def tmesh2voxel(tm, res, grid_min=None, grid_max=None, chunk_size=5000000, device='cuda', show_progress=True):
    """
    Constructs a dense voxel grid from a TriangleMesh and evaluates its SDF.
    
    Returns:
        grid_vertices, voxels, idx_grids, sdfs
    """
    if grid_min is None:
        grid_min = [-1.0, -1.0, -1.0]
    if grid_max is None:
        grid_max = [1.0, 1.0, 1.0]
        
    res_list = [res, res, res]
    grid_vertices, voxels, idx_grids = create_voxel_grid(grid_min, grid_max, res_list, device=device)
    
    num_points = grid_vertices.shape[0]
    sdfs = torch.empty(num_points, dtype=torch.float32, device=device)
    
    iterator = range(0, num_points, chunk_size)
    if show_progress:
        iterator = tqdm.tqdm(iterator, desc="Computing SDF")
        
    for i in iterator:
        end = min(i + chunk_size, num_points)
        chunk_points = grid_vertices[i:end]
        _, _, _, chunk_sdf = tm.query_points(chunk_points, return_sdf=True, return_prj_pts=False, sign_mode=1)
        sdfs[i:end] = chunk_sdf
        
    return grid_vertices, voxels, idx_grids, sdfs

def tmesh2sparse(tm, res, grid_min=None, grid_max=None, chunk_size=5000000, iso=0.0, device='cuda', show_progress=True):
    """
    Computes the SDF and returns only the active voxels (sparse grid) near the surface.
    
    Returns:
        grid_vertices, active_voxels, idx_grids, sdfs
    """
    grid_vertices, voxels, idx_grids, sdfs = tmesh2voxel(
        tm, res, grid_min=grid_min, grid_max=grid_max, chunk_size=chunk_size, device=device, show_progress=show_progress
    )
    
    active_indices = compute_active_voxels(voxels, sdfs, iso=iso)
    active_voxels = voxels[active_indices]
    
    return grid_vertices, active_voxels, idx_grids, sdfs
