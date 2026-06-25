from .._C import (
    KDTree,
    BVH,
    GSBVH,
    PGSBVH
)

from .._C import (
    TriangleMesh,
)

spatial_data_structures = ['KDTree', 'BVH', 'GSBVH', 'PGSBVH']
mesh_data_structures = ['TriangleMesh']

__all__ = spatial_data_structures + mesh_data_structures