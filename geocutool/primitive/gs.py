import torch
from typing import Tuple

# Explicitly import the compiled CMake target
from .._C import compute_aabb_wrapper, query_gs_voxel_intersection_brute_force_wrapper

def compute_gaussian_aabb(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    level: int,
    iso: float = 11.345,
    tol: float = 1. / 8.,
    rotnorm: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Axis-Aligned Bounding Box (AABB) for 3D Gaussians in world space.

    Args:
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        rotations (torch.Tensor): (N, 4) tensor of quaternions [w, x, y, z].
        scales (torch.Tensor): (N, 3) tensor of scale factors (pre-activation).
        iso (float, optional): The opacity cutoff threshold. Defaults to 11.345.
        tol (float, optional): Safety tolerance multiplier. Defaults to 0.125.
        level (int, optional): Octree subdivision level. Resolution will be at :math:`2^level`
        rotnorm (bool, optional): Set to True to force normalizing quaternions.

    Returns:
        aabb_min: torch.Tensor: (N, 3) tensor of AABB minimum corners.
        aabb_max: torch.Tensor: (N, 3) tensor of AABB maximum corners.
        contact_points: torch.Tensor: (N, 3) tensor of contact points on the AABB surface.
        covi: torch.Tensor: (N, 6) tensor of covariance inverses (flattened upper-triangular) for each Gaussian. 
            The 6 values correspond to [Cxx, Cxy, Cxz, Cyy, Cyz, Czz] where C is the covariance matrix of the Gaussian.
    """
    
    # Safety Boundary: Ensure memory is perfectly aligned before hitting C++
    if not means.is_cuda:
        raise ValueError("Inputs must be CUDA tensors.")
        
    means_c = means.contiguous().to(torch.float32)
    rotations_c = rotations.contiguous().to(torch.float32)
    scales_c = scales.contiguous().to(torch.float32)

    # Call the explicit C++ binding
    aabb_min, aabb_max, contact_points, covi = compute_aabb_wrapper(
        means_c,
        rotations_c,
        scales_c,
        iso,
        tol,
        level,
        rotnorm
    )

    return aabb_min, aabb_max, contact_points, covi

def query_gs_voxel_intersection(
    vx_aabb_mins: torch.Tensor,
    vx_aabb_maxs: torch.Tensor,
    means: torch.Tensor,
    covis: torch.Tensor,
    gs_aabb_mins: torch.Tensor,
    gs_aabb_maxs: torch.Tensor,
    contact_points: torch.Tensor,
    iso: float = 11.345,
    max_capacity: int = 10000000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Queries intersections between Gaussians and voxels using a brute-force approach.

    Args:
        vx_aabb_mins (torch.Tensor): (M, 3) tensor of voxel AABB minimum corners.
        vx_aabb_maxs (torch.Tensor): (M, 3) tensor of voxel AABB maximum corners.
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        covis (torch.Tensor): (N, 6) tensor of covariance inverses.
        gs_aabb_mins (torch.Tensor): (N, 3) tensor of Gaussian AABB minimum corners.
        gs_aabb_maxs (torch.Tensor): (N, 3) tensor of Gaussian AABB maximum corners.
        contact_points (torch.Tensor): (N, 9) tensor of contact points on the Gaussian surfaces.
        iso (float, optional): The opacity cutoff threshold. Defaults to 11.345.
        max_capacity (int, optional): Maximum number of intersections to store. Defaults to 10000000.

    Returns:
        out_voxel_ids: torch.Tensor: (K,) tensor of voxel IDs with intersections.
        out_gaus_ids: torch.Tensor: (K,) tensor of Gaussian IDs with intersections.
    """
    
    # Safety Boundary: Ensure memory is perfectly aligned before hitting C++
    if not vx_aabb_mins.is_cuda or not vx_aabb_maxs.is_cuda or not means.is_cuda or not covis.is_cuda or not gs_aabb_mins.is_cuda or not gs_aabb_maxs.is_cuda or not contact_points.is_cuda:
        raise ValueError("Inputs must be CUDA tensors.")

    vx_aabb_mins_c = vx_aabb_mins.contiguous().to(torch.float32)
    vx_aabb_maxs_c = vx_aabb_maxs.contiguous().to(torch.float32)
    means_c = means.contiguous().to(torch.float32)
    covis_c = covis.contiguous().to(torch.float32)
    gs_aabb_mins_c = gs_aabb_mins.contiguous().to(torch.float32)
    gs_aabb_maxs_c = gs_aabb_maxs.contiguous().to(torch.float32)
    contact_points_c = contact_points.contiguous().to(torch.float32)

    # Call the explicit C++ binding
    hit_mask, out_voxel_ids, out_gaus_ids = query_gs_voxel_intersection_brute_force_wrapper(
        vx_aabb_mins_c,
        vx_aabb_maxs_c,
        means_c,
        covis_c,
        gs_aabb_mins_c,
        gs_aabb_maxs_c,
        contact_points_c,
        iso,
        max_capacity
    )

    return hit_mask, out_voxel_ids, out_gaus_ids