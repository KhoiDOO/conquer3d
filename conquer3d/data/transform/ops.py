"""Functional geometric transformation operations on 3D vertex tensors.

This module provides functional primitives for rotating and scaling 3D point
coordinates with PyTorch tensor arithmetic.
"""

from typing import Union, Tuple, List
import math
import torch


def rotation(vertices: torch.Tensor, rotation_axis: str, rotation_degree: float) -> torch.Tensor:
    """Rotates 3D vertex coordinates around a canonical Cartesian axis by a specified angle in degrees.
    
    Args:
        vertices (torch.Tensor): Float32 tensor of shape `(N, 3)` representing 3D coordinates.
        rotation_axis (str): Canonical axis to rotate around (`'x'`, `'y'`, or `'z'`).
        rotation_degree (float): Rotation angle in degrees.
        
    Returns:
        torch.Tensor: Rotated vertex tensor of shape `(N, 3)`.

    Raises:
        ValueError: If `rotation_axis` is not `'x'`, `'y'`, or `'z'`.

    Example:
        >>> from conquer3d.data.transform.ops import rotation
        >>> rotated_verts = rotation(verts, rotation_axis='y', rotation_degree=45.0)
    """
    theta = math.radians(rotation_degree)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    device = vertices.device
    dtype = vertices.dtype
    
    if rotation_axis == 'x':
        rot_mat = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, cos_t, -sin_t],
            [0.0, sin_t, cos_t]
        ], dtype=dtype, device=device)
    elif rotation_axis == 'y':
        rot_mat = torch.tensor([
            [cos_t, 0.0, sin_t],
            [0.0, 1.0, 0.0],
            [-sin_t, 0.0, cos_t]
        ], dtype=dtype, device=device)
    elif rotation_axis == 'z':
        rot_mat = torch.tensor([
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=dtype, device=device)
    else:
        raise ValueError("rotation_axis must be 'x', 'y', or 'z'")
        
    return torch.matmul(vertices, rot_mat.T)


def scale(
    vertices: torch.Tensor,
    scale_factor: Union[float, Tuple[float, float, float], List[float]]
) -> torch.Tensor:
    """Scales 3D vertex coordinates by uniform or anisotropic scale factors.
    
    Args:
        vertices (torch.Tensor): Tensor of shape `(N, 3)` representing 3D coordinates.
        scale_factor (Union[float, Tuple[float, float, float], List[float]]): Scalar or per-axis scale factor.
        
    Returns:
        torch.Tensor: Scaled vertex tensor of shape `(N, 3)`.

    Example:
        >>> from conquer3d.data.transform.ops import scale
        >>> scaled_verts = scale(verts, scale_factor=0.5)
    """
    if isinstance(scale_factor, (list, tuple)):
        scale_factor = torch.tensor(scale_factor, dtype=vertices.dtype, device=vertices.device)
    return vertices * scale_factor
