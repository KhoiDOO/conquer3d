import torch
import importlib.metadata

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

__all__ = ['_C', 'creation', 'data_structure', 'primitive', 'ops', 'conversion', 'data', 'io', 'Triangle', 'Ray']
