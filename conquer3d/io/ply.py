"""Stanford PLY 3D mesh format read and write I/O routines.

This module provides high-speed I/O functions to read and write Stanford PLY
geometry files with vertex coordinates, face topologies (triangles & quads), and per-vertex colors.
"""

from typing import Tuple, Optional, Union, BinaryIO
import torch
import trimesh
import numpy as np


def read_ply(file_obj: Union[str, BinaryIO]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Reads a Stanford PLY mesh file and returns vertices, faces, and optional vertex colors.

    Args:
        file_obj (Union[str, BinaryIO]): File path string or binary file-like object.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - vertices (torch.Tensor): Float32 tensor of shape `(V, 3)` containing 3D vertex coordinates.
            - faces (torch.Tensor): Int64 tensor of shape `(F, 3)` or `(F, 4)` containing face indices.
            - colors (torch.Tensor | None): Float32 tensor of shape `(V, 3)` in normalized $[0, 1]$ range,
              or None if the file does not define vertex colors.

    Example:
        >>> from conquer3d.io import read_ply
        >>> verts, faces, colors = read_ply("model.ply")
    """
    mesh = trimesh.load(file_obj, process=False, skip_materials=True)
    
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    
    colors = None
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
        colors = torch.tensor(mesh.visual.vertex_colors[:, :3], dtype=torch.float32) / 255.0
        
    return vertices, faces, colors


def write_ply(
    filepath: str,
    vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> None:
    """Exports 3D mesh vertices, triangle or quad faces, and optional colors to a Stanford PLY file.

    Args:
        filepath (str): Destination file path string ending with `.ply`.
        vertices (Union[torch.Tensor, np.ndarray]): Vertex coordinates of shape `(V, 3)`.
        faces (Union[torch.Tensor, np.ndarray]): Face corner indices of shape `(F, 3)` or `(F, 4)`.
        colors (Union[torch.Tensor, np.ndarray], optional): Per-vertex RGB colors in range $[0, 1]$.
            Defaults to None.

    Raises:
        TypeError: If `vertices`, `faces`, or `colors` are not torch.Tensor or numpy.ndarray.
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    
    vc = None
    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        vc = (colors * 255.0).clip(0, 255).astype(np.uint8)
        
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vc, process=False)
    mesh.export(filepath)


def write_voxel_ply(
    filepath: str,
    vertices: Union[torch.Tensor, np.ndarray],
    voxels: Union[torch.Tensor, np.ndarray]
) -> None:
    """Exports 3D voxel cells (each with 8 corner indices) as a 6-quad per voxel mesh to Stanford PLY.

    Args:
        filepath (str): Target file path string ending with `.ply`.
        vertices (Union[torch.Tensor, np.ndarray]): Sparse grid/cloud corner coordinates `(V, 3)`.
        voxels (Union[torch.Tensor, np.ndarray]): Voxel corner indices `(K, 8)`.
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.detach().cpu().numpy()

    # 6 quad faces for each voxel cell [v0, v1, v2, v3, v4, v5, v6, v7]
    quad_offsets = np.array([
        [0, 3, 2, 1],  # Bottom
        [4, 5, 6, 7],  # Top
        [0, 1, 5, 4],  # Front
        [3, 7, 6, 2],  # Back
        [0, 4, 7, 3],  # Left
        [1, 2, 6, 5],  # Right
    ], dtype=np.int64)

    quad_faces = voxels[:, quad_offsets].reshape(-1, 4)
    write_ply(filepath, vertices, quad_faces)
