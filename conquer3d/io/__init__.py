"""Input/Output routines for 3D geometry and mesh file formats.

This package provides high-speed I/O loaders and writers for:
- Wavefront OBJ files (`read_obj`, `write_obj`).
- Geomview OFF files (`read_off`).
"""

from .obj import read_obj, write_obj
from .off import read_off

__all__ = [
    "read_obj",
    "write_obj",
    "read_off"
]