"""Geometric format and volumetric conversion pipelines.

This module provides bidirectional conversions between:
- Dense voxel grids and sparse COO coordinates (`voxel2sparse`, `sparse2voxel`).
- Dense occupancy grids and sparse COO indices (`dense_occ2sparse_coo`, `sparse_coo2dense_occ`).
- Triangle meshes and dense/sparse voxel signed distance fields (`tmesh2voxel`, `tmesh2sparse`, `tmesh2voxelcloud`).
"""

from .grid import (
    voxel2sparse,
    sparse2voxel,
    sparse_coo2dense_occ,
    dense_occ2sparse_coo
)
from .tmesh import (
    tmesh2voxel,
    tmesh2sparse,
    tmesh2voxelcloud
)

__all__ = [
    "voxel2sparse",
    "sparse2voxel",
    "sparse_coo2dense_occ",
    "dense_occ2sparse_coo",
    "tmesh2voxel",
    "tmesh2sparse",
    "tmesh2voxelcloud"
]