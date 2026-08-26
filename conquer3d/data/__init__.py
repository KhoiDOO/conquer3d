"""Data loading, dataset abstractions, geometric transformations, and batch collation.

This package provides:
- Benchmark mesh datasets (`MeshDataset`, `Digit3D`, `RedWood`).
- Custom PyTorch DataLoader collate functions (`bmesh_collate_fn`, `sparse_collate_fn`).
- Geometric data augmentations (`Rotation`, `Scale`, `MeshSequence`).
- Canonical 3D asset downloads (`Bunny`, `Dragon`, `Armadillo`, `Iphiagenia`).
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