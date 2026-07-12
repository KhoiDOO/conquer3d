import torch
import numpy as np

def find_near_duplicates_sort(points, epsilon=1e-7):
    """
    Approximate detection of near-duplicate points in a point cloud using sorting.
    
    Args:
    - points: Tensor of shape (N, 3), where N is the number of points.
    - epsilon: Distance threshold for considering points as duplicates.
    
    Returns:
    - duplicate_indices: A tensor of shape (M,) where M is the number of points who have a near-duplicate.
    """
    sorted_indices = torch.argsort(points[:, 0]) 
    points_sorted = points[sorted_indices]
    
    diffs = points_sorted[1:] - points_sorted[:-1] 
    distances = torch.norm(diffs, dim=-1)

    close_mask = distances < epsilon
    duplicate_indices_1, duplicate_indices_2 = sorted_indices[:-1][close_mask], sorted_indices[1:][close_mask]
    duplicate_indices = torch.cat((duplicate_indices_1, duplicate_indices_2))
    return duplicate_indices

def tet_is_inside_oriented(vertices, tets):
    """
    Check if the tetrahedra are oriented correctly, i.e. if the normals of the faces point outwards. 
    This is tested by checking if the dot product of the normals and the vector from the barycenter of the tetrahedron to the barycenter of the face is negative.
    """
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.long)
    tets_combinations = tets[:, faces]
    normals_per_tets = torch.cross(vertices[tets_combinations[:, :, 1]] - vertices[tets_combinations[:, :, 0]], vertices[tets_combinations[:, :, 2]] - vertices[tets_combinations[:, :, 0]], dim=2)
    tet_barycenter = vertices[tets].mean(dim=1)
    faces_barycenter = vertices[tets_combinations].mean(dim=2)
    faces_b_to_tet_b = faces_barycenter - tet_barycenter.unsqueeze(1)
    dot = torch.sum(normals_per_tets * faces_b_to_tet_b, dim=2)
    return dot < 0

def reorient_tetrahedra(vertices, tets):
    inside_oriented_triangles = tet_is_inside_oriented(vertices, tets)
    to_flip = inside_oriented_triangles.all(dim=1)
    tets[to_flip] = tets[to_flip][:, [0, 2, 1, 3]]
    return tets

import scipy.spatial

@torch.no_grad()
def delaunay_simplices_tetgen(points):
    # PyVista's tetgen wrapper raises RuntimeError on point clouds without faces.
    # We natively use scipy's Delaunay which perfectly tetrahedralizes 3D point clouds.
    delaunay = scipy.spatial.Delaunay(points.detach().cpu().numpy())
    return torch.tensor(delaunay.simplices.astype(np.int64), device=points.device)

def tetrahedralize(points):
    duplicates = find_near_duplicates_sort(points)
    if duplicates.shape[0] > 0:
        with torch.no_grad():
            points[duplicates] += torch.randn_like(points[duplicates]) * 1e-6
    tetrahedra = delaunay_simplices_tetgen(points)
    tetrahedra = reorient_tetrahedra(points, tetrahedra)
    return tetrahedra

def get_edges(tetrahedra):
    tet_edges = torch.tensor([
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3], [2, 3]
    ], device=tetrahedra.device)
    edges = tetrahedra[:, tet_edges]
    return edges
