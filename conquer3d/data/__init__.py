"""Data loading, dataset abstractions, geometric transformations, and batch collation.

Benchmark mesh datasets, collate functions for variable-sized meshes and sparse tensors,
composable geometric augmentations, and the downloadable standard assets in
:mod:`conquer3d.data.assets`.
"""

from . import assets
from . import dataset
from . import transform
from . import collate

from .dataset import BaseMeshDataset, MeshDataset, MeshFolderDataset, ToyMeshDataset, Digit3D, PointDigit3D, Digit3DMV

__all__ = [
    'assets', 
    'dataset', 
    'transform', 
    'collate',
    'BaseMeshDataset',
    'MeshDataset',
    'MeshFolderDataset',
    'ToyMeshDataset',
    'Digit3D',
    'PointDigit3D',
    'Digit3DMV',
]