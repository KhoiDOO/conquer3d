import torch
from typing import Tuple, Optional, Union

# Explicitly import the compiled CMake target
from .._C import (
    compute_gs_covi_func,
    solve_gs_neighbor_mahalanobis_radius_func,
    compute_gs_aabb_func
)

def compute_gs_covi(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    level: int,
    tol: float = 1. / 8.,
    rotnorm: bool = False
) -> torch.Tensor:
    """
    Computes the covariance matrices for 3D Gaussians in world space.

    Args:
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        rotations (torch.Tensor): (N, 4) tensor of quaternions [w, x, y, z].
        scales (torch.Tensor): (N, 3) tensor of scale factors (pre-activation).
        level (int, optional): Octree subdivision level. Resolution will be at :math:`2^level`
        tol (float, optional): Safety tolerance multiplier. Defaults to 0.125.
        rotnorm (bool, optional): Set to True to force normalizing quaternions.

    Returns:
        covi: torch.Tensor: (N, 6) tensor of covariance inverses (flattened upper-triangular) for each Gaussian. 
            The 6 values correspond to [Cxx, Cxy, Cxz, Cyy, Cyz, Czz] where C is the covariance matrix of the Gaussian.
    """
    
    if not all(t.is_cuda for t in [means, rotations, scales]):
        raise ValueError("All input tensors must be CUDA tensors.")
        
    means_c = means.contiguous().to(torch.float32)
    rotations_c = rotations.contiguous().to(torch.float32)
    scales_c = scales.contiguous().to(torch.float32)

    covi = compute_gs_covi_func(
        means_c,
        rotations_c,
        scales_c,
        rotnorm,
        tol,
        level
    )

    return covi

def solve_gs_neighbor_mahalanobis_radius(
    means: torch.Tensor,
    covis: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Computes the Mahalanobis radius for each Gaussian based on its k-nearest neighbors.

    Args:
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        covis (torch.Tensor): (N, 6) tensor of covariance inverses (flattened upper-triangular).
        k (int): Number of nearest neighbors to consider.

    Returns:
        isos: torch.Tensor: (N,) tensor of Mahalanobis radii for each Gaussian.
    """
    
    if not all(t.is_cuda for t in [means, covis]):
        raise ValueError("All input tensors must be CUDA tensors.")
        
    means_c = means.contiguous().to(torch.float32)
    covis_c = covis.contiguous().to(torch.float32)

    isos = solve_gs_neighbor_mahalanobis_radius_func(
        means_c,
        covis_c,
        k
    )

    return isos

def compute_gs_aabb(
    means: torch.Tensor,
    scales: torch.Tensor,
    covis: torch.Tensor,
    level: int,
    iso: Union[float, torch.Tensor] = 11.345,
    tol: float = 1. / 8.
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Axis-Aligned Bounding Box (AABB) for 3D Gaussians in world space.

    Args:
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        scales (torch.Tensor): (N, 3) tensor of scale factors.
        covis (torch.Tensor): (N, 6) tensor of covariance inverses (flattened upper-triangular).
        level (int, optional): Octree subdivision level. Resolution will be at :math:`2^level`
        iso (float, optional): The opacity cutoff threshold. Defaults to 11.345.
        tol (float, optional): Safety tolerance multiplier. Defaults to 0.125.

    Returns:
        aabb_min: torch.Tensor: (N, 3) tensor of AABB minimum corners.
        aabb_max: torch.Tensor: (N, 3) tensor of AABB maximum corners.
        contact_points: torch.Tensor: (N, 9) tensor of contact points on the AABB surface.
    """
    
    if not all(t.is_cuda for t in [means, scales, covis]):
        raise ValueError("All input tensors must be CUDA tensors.")
        
    means_c = means.contiguous().to(torch.float32)
    scales_c = scales.contiguous().to(torch.float32)
    covis_c = covis.contiguous().to(torch.float32)
    
    if isinstance(iso, torch.Tensor):
        if not iso.is_cuda: raise ValueError("iso tensor must be on CUDA")
        isos = iso.contiguous().to(torch.float32)
        iso = 0.0
    else:
        isos = None
        iso = float(iso)

    aabb_min, aabb_max, contact_points = compute_gs_aabb_func(
        means_c,
        scales_c,
        covis_c,
        isos,
        iso,
        tol,
        level
    )

    return aabb_min, aabb_max, contact_points