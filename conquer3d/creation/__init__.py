"""Procedural canonical 3D primitive generators.

This package provides procedural creation functions for 3D meshes such as
UV spheres (`create_sphere`) and regular tetrahedra (`create_tetrahedra`).
"""

from .triangle_creation import create_sphere, create_tetrahedra

__all__ = [
    "create_sphere",
    "create_tetrahedra"
]
