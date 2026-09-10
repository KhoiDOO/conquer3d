"""3D geometric primitives and radiance field representations.

Ray and triangle primitives, 3D and periodic Gaussian splatting operators, and learnable
superquadric sets with their inside-outside function and the softmin that unions a set
into one solid.
"""

from conquer3d._C import Triangle, Ray
from .gs import (
    compute_gs_covi,
    solve_gs_neighbor_mahalanobis_radius,
    compute_gs_aabb
)
from .pgs import (
    solve_pgs_cluster_tangency_radius
)
from .sq import (
    SuperQuadrics,
    compute_sq_sdf,
    compute_sq_union
)

__all__ = [
    'Triangle',
    'Ray',
    'compute_gs_covi',
    'solve_gs_neighbor_mahalanobis_radius',
    'compute_gs_aabb',
    'solve_pgs_cluster_tangency_radius',
    'SuperQuadrics',
    'compute_sq_sdf',
    'compute_sq_union'
]
