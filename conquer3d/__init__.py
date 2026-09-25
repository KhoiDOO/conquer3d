"""Conquer3D: High-Performance GPU Differentiable 3D Geometry & Vision Library.

Conquer3D is a unified PyTorch/CUDA library providing GPU-accelerated spatial
data structures (Linear BVH, KD-Tree, Z-Curves), discrete differential geometry
operators (Laplace-Beltrami, Curvatures), differentiable isosurface extraction
(Dual Marching Cubes, Dual Contouring, Marching Tetrahedra), and 3D Gaussian
Splatting geometric utilities.
"""

import os
import sys
import importlib.metadata
import torch

# Ensure PyTorch C++ symbols (libc10.so, libtorch.so) are loaded into global symbol table on Linux
if hasattr(os, 'RTLD_GLOBAL') and hasattr(sys, 'getdlopenflags'):
    try:
        sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_GLOBAL)
    except Exception:
        pass

try:
    __version__ = importlib.metadata.version('conquer3d')
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

# Imported before any subpackage: it decides what `conquer3d._C` resolves to, and the
# subpackages bind names out of it at their own import time.
from ._extension import _C, HAS_EXTENSION, ExtensionUnavailableError, extension_error

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
    'HAS_EXTENSION',
    'ExtensionUnavailableError',
    'extension_error',
    '__version__'
]
