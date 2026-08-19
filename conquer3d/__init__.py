"""Conquer3D: High-Performance GPU Differentiable 3D Geometry & Vision Library.

Conquer3D is a unified PyTorch/CUDA library providing GPU-accelerated spatial
data structures (Linear BVH, KD-Tree, Z-Curves), discrete differential geometry
operators (Laplace-Beltrami, Curvatures), differentiable isosurface extraction
(Dual Marching Cubes, Dual Contouring, Marching Tetrahedra), and 3D Gaussian
Splatting geometric utilities.
"""

import importlib.metadata
import torch

try:
    __version__ = importlib.metadata.version('conquer3d')
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from . import _C
from . import creation
from . import data_structure
from . import primitive
from . import ops
from . import conversion
from . import data
from . import io

from .primitive import Triangle, Ray

__all__ = [
    '_C',
    'creation',
    'data_structure',
    'primitive',
    'ops',
    'conversion',
    'data',
    'io',
    'Triangle',
    'Ray',
    '__version__'
]
