"""3D Gaussian Splatting (3DGS) primitive mathematical operators.

This module provides GPU-accelerated covariance matrix inverse evaluation,
k-nearest neighbor Mahalanobis radiometry solving, and oriented bounding
ellipsoid AABB generation for 3D Gaussian radiance fields.
"""

from typing import Tuple, Optional, Union
import torch

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
    """Computes inverse covariance matrices $\\Sigma^{-1}$ for 3D Gaussians on CUDA.

    Evaluates $\\Sigma^{-1} = (S^{-1} R^T)^T (S^{-1} R^T)$ with lower-bound voxel clamping.

    Args:
        means (torch.Tensor): Float32 tensor of shape `(N, 3)` with Gaussian center positions.
        rotations (torch.Tensor): Float32 tensor of shape `(N, 4)` with unit quaternions `[w, x, y, z]`.
        scales (torch.Tensor): Float32 tensor of shape `(N, 3)` with scale parameters.
        level (int): Octree subdivision level determining minimal voxel scale ($2^{-\\text{level}}$).
        tol (float, optional): Minimal scale multiplier threshold. Defaults to 0.125.
        rotnorm (bool, optional): If True, normalizes quaternions in CUDA kernel. Defaults to False.

    Returns:
        torch.Tensor: Float32 tensor of shape `(N, 6)` storing flattened upper-triangular
        inverse covariance entries $[C_{xx}, C_{xy}, C_{xz}, C_{yy}, Cyz, C_{zz}]$.

    Raises:
        ValueError: If input tensors are not on CUDA.
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
    """Solves the adaptive Mahalanobis radius for each Gaussian from its k-NN neighbors.

    Args:
        means (torch.Tensor): Float32 tensor of shape `(N, 3)` with Gaussian center coordinates.
        covis (torch.Tensor): Float32 tensor of shape `(N, 6)` with inverse covariance entries.
        k (int): Number of nearest neighbors to query via KD-Tree.

    Returns:
        torch.Tensor: Float32 tensor of shape `(N,)` with optimal Mahalanobis isosurface radii.

    Raises:
        ValueError: If input tensors are not on CUDA.
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
    """Computes tight 3D Axis-Aligned Bounding Boxes (AABBs) for 3D Gaussians.

    Args:
        means (torch.Tensor): Float32 tensor of shape `(N, 3)` with Gaussian centers.
        scales (torch.Tensor): Float32 tensor of shape `(N, 3)` with scales.
        covis (torch.Tensor): Float32 tensor of shape `(N, 6)` with inverse covariance values.
        level (int): Octree resolution level.
        iso (Union[float, torch.Tensor], optional): Mahalanobis cutoff threshold $\\chi^2$.
            Defaults to 11.345 ($3\\sigma$ confidence ellipsoid).
        tol (float, optional): Safety tolerance multiplier. Defaults to 0.125.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - aabb_min (torch.Tensor): Float32 tensor of shape `(N, 3)` with lower box corners.
            - aabb_max (torch.Tensor): Float32 tensor of shape `(N, 3)` with upper box corners.
            - contact_points (torch.Tensor): Float32 tensor of shape `(N, 9)` with ellipsoid contact points.

    Raises:
        ValueError: If inputs are not on CUDA.
    """
    if not all(t.is_cuda for t in [means, scales, covis]):
        raise ValueError("All input tensors must be CUDA tensors.")
        
    means_c = means.contiguous().to(torch.float32)
    scales_c = scales.contiguous().to(torch.float32)
    covis_c = covis.contiguous().to(torch.float32)
    
    if isinstance(iso, torch.Tensor):
        if not iso.is_cuda:
            raise ValueError("iso tensor must be on CUDA")
        isos = iso.contiguous().to(torch.float32)
        iso_val = 0.0
    else:
        isos = None
        iso_val = float(iso)

    aabb_min, aabb_max, contact_points = compute_gs_aabb_func(
        means_c,
        scales_c,
        covis_c,
        isos,
        iso_val,
        tol,
        level
    )

    return aabb_min, aabb_max, contact_points