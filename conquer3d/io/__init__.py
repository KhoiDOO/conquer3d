"""Input/Output routines for 3D geometry and mesh file formats.

This package provides high-speed I/O loaders and writers for:
- Wavefront OBJ files (`read_obj`, `write_obj`, `write_quad_obj`, `write_voxel_obj`).
- Stanford PLY files (`read_ply`, `write_ply`, `write_voxel_ply`).
- Geomview OFF files (`read_off`).
"""

from .obj import read_obj, write_obj, write_quad_obj, write_voxel_obj
from .ply import read_ply, write_ply, write_voxel_ply
from .off import read_off

__all__ = [
    "read_obj",
    "write_obj",
    "write_quad_obj",
    "write_voxel_obj",
    "read_ply",
    "write_ply",
    "write_voxel_ply",
    "read_off"
]