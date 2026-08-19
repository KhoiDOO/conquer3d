"""Procedural 3D canonical primitive generation routines.

This module provides parametric generators for standard canonical primitives
such as UV spheres and regular tetrahedra.
"""

from typing import Tuple
import torch

from .._C import create_sphere as create_sphere_c
from .._C import create_tetrahedra as create_tetrahedra_c


def create_sphere(
    sectors: int = 32,
    stacks: int = 16,
    radius: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a parametric UV sphere mesh with vertex coordinates and triangle faces.

    Args:
        sectors (int, optional): Number of longitudinal angular sectors. Defaults to 32.
        stacks (int, optional): Number of latitudinal horizontal stacks. Defaults to 16.
        radius (float, optional): Radius of the generated sphere. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - vertices (torch.Tensor): Float32 tensor of shape `(N, 3)` with 3D vertex positions.
            - triangles (torch.Tensor): Int32 tensor of shape `(M, 3)` with triangle corner indices.

    Example:
        >>> from conquer3d.creation import create_sphere
        >>> verts, faces = create_sphere(sectors=64, stacks=32, radius=1.0)
    """
    return create_sphere_c(sectors, stacks, radius)


def create_tetrahedra(
    radius: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a regular canonical tetrahedron mesh.

    Args:
        radius (float, optional): Circumscribed radius of the tetrahedron. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - vertices (torch.Tensor): Float32 tensor of shape `(4, 3)` with 4 vertex coordinates.
            - triangles (torch.Tensor): Int32 tensor of shape `(4, 3)` with 4 outward triangle faces.

    Example:
        >>> from conquer3d.creation import create_tetrahedra
        >>> verts, faces = create_tetrahedra(radius=1.0)
    """
    return create_tetrahedra_c(radius)
