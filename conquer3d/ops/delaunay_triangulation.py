"""Delaunay triangulation and tetrahedralization routines for 3D point sets.

This module provides utilities to tetrahedralize 3D point clouds via
scipy.spatial.Delaunay, with jittering for near-duplicates, outwards
normal verification, and tetrahedron edge extraction.
"""

from typing import Tuple
import torch
import numpy as np
import scipy.spatial


def find_near_duplicates_sort(points: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Detects indices of near-duplicate points in a point cloud using 1D sorting.

    Args:
        points (torch.Tensor): Tensor of shape `(N, 3)` containing 3D coordinates.
        epsilon (float, optional): Euclidean distance threshold for declaring duplicate
            points. Defaults to 1e-7.

    Returns:
        torch.Tensor: 1D int64 tensor containing indices of points that have near-duplicates.
    """
    sorted_indices = torch.argsort(points[:, 0]) 
    points_sorted = points[sorted_indices]
    
    diffs = points_sorted[1:] - points_sorted[:-1] 
    distances = torch.norm(diffs, dim=-1)

    close_mask = distances < epsilon
    duplicate_indices_1, duplicate_indices_2 = sorted_indices[:-1][close_mask], sorted_indices[1:][close_mask]
    duplicate_indices = torch.cat((duplicate_indices_1, duplicate_indices_2))
    return duplicate_indices


def tet_is_inside_oriented(vertices: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    """Checks whether tetrahedron face triangles are oriented outwards.

    Evaluates the sign of the dot product between the outward face normal and
    the vector from the tetrahedron centroid to the face barycenter.

    Args:
        vertices (torch.Tensor): Vertex coordinate tensor of shape `(N, 3)`.
        tets (torch.Tensor): Tetrahedron corner index tensor of shape `(M, 4)`.

    Returns:
        torch.Tensor: Bool tensor of shape `(M, 4)` indicating whether each face
        is inward-oriented.
    """
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.long)
    tets_combinations = tets[:, faces]
    normals_per_tets = torch.cross(
        vertices[tets_combinations[:, :, 1]] - vertices[tets_combinations[:, :, 0]],
        vertices[tets_combinations[:, :, 2]] - vertices[tets_combinations[:, :, 0]],
        dim=2
    )
    tet_barycenter = vertices[tets].mean(dim=1)
    faces_barycenter = vertices[tets_combinations].mean(dim=2)
    faces_b_to_tet_b = faces_barycenter - tet_barycenter.unsqueeze(1)
    dot = torch.sum(normals_per_tets * faces_b_to_tet_b, dim=2)
    return dot < 0


def reorient_tetrahedra(vertices: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    """Reorients tetrahedron corner indices to guarantee consistent positive volume orientation.

    Args:
        vertices (torch.Tensor): Vertex coordinate tensor of shape `(N, 3)`.
        tets (torch.Tensor): Tetrahedron index tensor of shape `(M, 4)`.

    Returns:
        torch.Tensor: Consistently oriented tetrahedron tensor of shape `(M, 4)`.
    """
    inside_oriented_triangles = tet_is_inside_oriented(vertices, tets)
    to_flip = inside_oriented_triangles.all(dim=1)
    tets[to_flip] = tets[to_flip][:, [0, 2, 1, 3]]
    return tets


@torch.no_grad()
def delaunay_simplices_tetgen(points: torch.Tensor) -> torch.Tensor:
    """Computes 3D Delaunay triangulation simplices using SciPy Delaunay.

    Args:
        points (torch.Tensor): Point coordinates of shape `(N, 3)`.

    Returns:
        torch.Tensor: Int64 tensor of shape `(M, 4)` containing 4-vertex simplices (tetrahedra).
    """
    delaunay = scipy.spatial.Delaunay(points.detach().cpu().numpy())
    return torch.tensor(delaunay.simplices.astype(np.int64), device=points.device)


def tetrahedralize(points: torch.Tensor) -> torch.Tensor:
    """Tetrahedralizes an arbitrary 3D point cloud via Delaunay triangulation.

    Applies small random jittering to near-duplicate points to prevent numerical
    singularities, constructs 3D Delaunay simplices, and reorients all tetrahedra.

    Args:
        points (torch.Tensor): Tensor of shape `(N, 3)` containing 3D coordinates.

    Returns:
        torch.Tensor: Int64 tensor of shape `(M, 4)` containing oriented tetrahedron vertex indices.

    Example:
        >>> import torch
        >>> from conquer3d.ops import tetrahedralize
        >>> points = torch.randn(100, 3, device='cuda')
        >>> tets = tetrahedralize(points)
        >>> print(tets.shape)
        torch.Size([..., 4])
    """
    duplicates = find_near_duplicates_sort(points)
    if duplicates.shape[0] > 0:
        with torch.no_grad():
            points[duplicates] += torch.randn_like(points[duplicates]) * 1e-6
    tetrahedra = delaunay_simplices_tetgen(points)
    tetrahedra = reorient_tetrahedra(points, tetrahedra)
    return tetrahedra


def get_edges(tetrahedra: torch.Tensor) -> torch.Tensor:
    """Extracts the 6 unique edges of each tetrahedron.

    Args:
        tetrahedra (torch.Tensor): Tensor of shape `(M, 4)` containing tetrahedron indices.

    Returns:
        torch.Tensor: Tensor of shape `(M, 6, 2)` containing corner index pairs for each edge.
    """
    tet_edges = torch.tensor([
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3], [2, 3]
    ], device=tetrahedra.device)
    edges = tetrahedra[:, tet_edges]
    return edges
