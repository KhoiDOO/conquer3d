import torch
from typing import Tuple

from .._C import (
    solve_pgs_cluster_tangency_radius_func
)

def solve_pgs_cluster_tangency_radius(
    means: torch.Tensor,
    normals: torch.Tensor,
    covis: torch.Tensor,
    k: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the tangency radius for each Gaussian cluster based on its k-nearest neighbors.

    Args:
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        normals (torch.Tensor): (N, 3) tensor of Gaussian principal axes.
        covis (torch.Tensor): (N, 6) tensor of covariance inverses.
        k (int, optional): Number of nearest neighbors to consider. Defaults to 16.

    Returns:
        isos: (N,) tensor of computed tangency radii for each Gaussian.
        invalid_mask: (N,) boolean tensor indicating which Gaussians had invalid computations.
    """
    if not all(t.is_cuda for t in [means, normals, covis]):
        raise ValueError("All input tensors must be CUDA tensors.")
    
    isos, invalid_mask = solve_pgs_cluster_tangency_radius_func(
        means.contiguous().to(torch.float32),
        normals.contiguous().to(torch.float32),
        covis.contiguous().to(torch.float32),
        k)
    
    return isos, invalid_mask