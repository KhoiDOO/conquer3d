import torch
from typing import Tuple

# Explicitly import the compiled CMake target
from .._C import get_aabb_wrapper

def compute_gaussian_aabb(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    level: int,
    iso: float = 11.345,
    tol: float = 1. / 8.,
    rotnorm: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]: (aabb_min, aabb_max), both of shape (N, 3).
    """
    
    # Safety Boundary: Ensure memory is perfectly aligned before hitting C++
    if not means.is_cuda:
        raise ValueError("Inputs must be CUDA tensors.")
        
    means_c = means.contiguous().to(torch.float32)
    rotations_c = rotations.contiguous().to(torch.float32)
    scales_c = scales.contiguous().to(torch.float32)

    # Call the explicit C++ binding
    aabb_min, aabb_max = get_aabb_wrapper(
        means_c,
        rotations_c,
        scales_c,
        iso,
        tol,
        level,
        rotnorm
    )

    return aabb_min, aabb_max