"""Periodic Gaussian Splatting (PGS) primitive mathematical operators.

This module provides pairwise tangency radius solving for Periodic Gaussian
Splatting representations and directional radiance structures.
"""

from typing import Tuple
import torch

from .._C import (
    solve_pgs_cluster_tangency_radius_func
)


def solve_pgs_cluster_tangency_radius(
    means: torch.Tensor,
    normals: torch.Tensor,
    covis: torch.Tensor,
    k: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes optimal tangency contact radii for Periodic Gaussians from k-NN clusters.

    Args:
        means (torch.Tensor): Float32 tensor of shape `(N, 3)` with Gaussian centers on CUDA.
        normals (torch.Tensor): Float32 tensor of shape `(N, 3)` with surface normals / principal axes.
        covis (torch.Tensor): Float32 tensor of shape `(N, 6)` with inverse covariance upper-triangles.
        k (int, optional): Number of nearest neighbors to query. Defaults to 16.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - isos (torch.Tensor): Float32 tensor of shape `(N,)` with computed tangency radii.
            - invalid_mask (torch.Tensor): Bool tensor of shape `(N,)` marking points where
              tangency could not be analytically solved.

    Raises:
        ValueError: If input tensors are not on CUDA.
    """
    if not all(t.is_cuda for t in [means, normals, covis]):
        raise ValueError("All input tensors must be CUDA tensors.")
    
    isos, invalid_mask = solve_pgs_cluster_tangency_radius_func(
        means.contiguous().to(torch.float32),
        normals.contiguous().to(torch.float32),
        covis.contiguous().to(torch.float32),
        k
    )
    
    return isos, invalid_mask