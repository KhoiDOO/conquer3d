"""Geometric format and volumetric conversion pipelines.

Bidirectional conversions between dense grids, sparse COO coordinates, occupancy volumes,
and the signed distance fields built from a triangle mesh.
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