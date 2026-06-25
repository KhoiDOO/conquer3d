import torch
from typing import Tuple

from .._C import create_sphere as create_sphere_c

def create_sphere(
    sectors: int = 32,
    stacks: int = 16,
    radius: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a UV sphere returning (vertices, triangles) tensors on the CPU.

    Args:
        sectors (int, optional): Number of longitudinal sectors. Defaults to 32.
        stacks (int, optional): Number of latitudinal stacks. Defaults to 16.
        radius (float, optional): Radius of the sphere. Defaults to 1.0.

    Returns:
        vertices: (N, 3) float32 tensor of vertex positions.
        triangles: (M, 3) int32 tensor of triangle indices.
    """
    return create_sphere_c(sectors, stacks, radius)
