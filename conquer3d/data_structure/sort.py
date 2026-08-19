"""Spatial Z-curve (Morton code) sorting routines for 3D point sets.

This module provides GPU-accelerated spatial sorting based on 30-bit 3D Morton
codes (Lebesgue space-filling curve), enabling cache-coherent spatial hashing
and fast geometric neighborhood traversal.
"""

from typing import Tuple
import torch
import conquer3d._C as _C


def z_curve_sort(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sorts 3D point sets along the space-filling Morton Z-curve.

    Assumes points are normalized in the $[0, 1]^3$ unit cube. Coordinates outside
    this range are automatically clamped by the underlying CUDA kernel.

    Args:
        points (torch.Tensor): Coordinates tensor of shape `(..., N, 3)` with dtype
            `torch.float32` on CUDA device.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - sorted_points (torch.Tensor): Points sorted along the Z-curve of shape `(..., N, 3)`.
            - sorted_indices (torch.Tensor): Forward permutation indices of shape `(..., N)`.
            - inverse_indices (torch.Tensor): Inverse scatter indices of shape `(..., N)`
              satisfying `torch.gather(sorted_points, -2, inverse_indices) == points`.

    Raises:
        AssertionError: If `points` is not on CUDA, not float32, or does not have trailing size 3.

    Example:
        >>> import torch
        >>> from conquer3d.data_structure import z_curve_sort
        >>> pts = torch.rand(1000, 3, device='cuda')
        >>> sorted_pts, sort_idx, inv_idx = z_curve_sort(pts)
    """
    assert points.is_cuda, "Points must be on CUDA"
    assert points.dtype == torch.float32, "Points must be float32"
    assert points.shape[-1] == 3, "Points must be 3D"
    
    # Compute the Morton codes for each point
    codes = _C.compute_zcurve(points.contiguous())
    
    # codes has shape (..., N). We always sort along the last dimension (dim=-1)
    sorted_codes, sorted_indices = torch.sort(codes, dim=-1)
    
    # Gather the points using the sorted indices
    dim = -2 if points.dim() > 1 else -1
    expanded_indices = sorted_indices.unsqueeze(-1).expand_as(points)
    sorted_points = torch.gather(points, dim=dim, index=expanded_indices)
    
    # Compute the inverse indices (scatter)
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    
    return sorted_points, sorted_indices, inverse_indices
