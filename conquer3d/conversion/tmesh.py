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
        
    if isinstance(res, int):
        res_list = [res, res, res]
    else:
        res_list = list(res)
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
    This now completely bypasses dense memory allocation!
    
    Returns:
        grid_vertices, active_voxels, idx_grids, sdfs
    """
    if grid_min is None:
        grid_min = [-1.0, -1.0, -1.0]
    if grid_max is None:
        grid_max = [1.0, 1.0, 1.0]
        
    if isinstance(res, int):
        res_list = [res, res, res]
    else:
        res_list = list(res)

    from conquer3d.data_structure import create_voxel_grid_from_tmesh
    
    if tm.bvh is None:
        tm.build_bvh()
        
    grid_vertices, active_voxels, unique_vert_ids = create_voxel_grid_from_tmesh(
        grid_min, grid_max, res_list, tm
    )
    
    num_points = grid_vertices.shape[0]
    sdfs = torch.empty(num_points, dtype=torch.float32, device=device)
    
    iterator = range(0, num_points, chunk_size)
    if show_progress:
        iterator = tqdm.tqdm(iterator, desc="Computing Sparse SDF")
        
    for i in iterator:
        end = min(i + chunk_size, num_points)
        chunk_points = grid_vertices[i:end]
        _, _, _, chunk_sdf = tm.query_points(chunk_points, return_sdf=True, return_prj_pts=False, sign_mode=1)
        sdfs[i:end] = chunk_sdf
        
    # Since we strictly extracted surface-intersecting voxels in C++, 
    # we don't strictly need to re-filter them, but we can do a quick check to refine the band.
    # However, marching_cubes only uses active_voxels.
    # To keep identical API, we return None for idx_grids since it's dense and unnecessary.
    return grid_vertices, active_voxels.to(torch.int64), None, sdfs

