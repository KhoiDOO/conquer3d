"""Wavefront OBJ 3D mesh format read and write I/O routines.

This module provides high-speed I/O functions to read and write Wavefront OBJ
geometry files with vertex coordinates, triangle face topologies, and per-vertex colors.
"""

from typing import Tuple, Optional, Union, BinaryIO
import torch
import trimesh
import numpy as np


def read_obj(file_obj: Union[str, BinaryIO]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Reads a Wavefront OBJ mesh file and returns vertices, faces, and vertex colors.
    
    Args:
        file_obj (Union[str, BinaryIO]): File path string or binary file-like object.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - vertices (torch.Tensor): Float32 tensor of shape `(V, 3)` containing 3D vertex coordinates.
            - faces (torch.Tensor): Int64 tensor of shape `(F, 3)` containing triangle face indices.
            - colors (torch.Tensor | None): Float32 tensor of shape `(V, 3)` in normalized $[0, 1]$ range,
              or None if the file does not define vertex colors.

    Example:
        >>> from conquer3d.io import read_obj
        >>> verts, faces, colors = read_obj("model.obj")
    """
    mesh = trimesh.load(file_obj, process=False, force='mesh', skip_materials=True)
    
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    
    colors = None
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
        colors = torch.tensor(mesh.visual.vertex_colors[:, :3], dtype=torch.float32) / 255.0
        
    return vertices, faces, colors


def write_obj(
    filepath: str,
    vertices: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> None:
    """Exports 3D mesh vertices, faces, and optional colors to a Wavefront OBJ file.

    Args:
        filepath (str): Destination file path string ending with `.obj`.
        vertices (Union[torch.Tensor, np.ndarray]): Vertex coordinates of shape `(V, 3)`.
        faces (Union[torch.Tensor, np.ndarray]): Face corner indices of shape `(F, 3)`.
        colors (Union[torch.Tensor, np.ndarray], optional): Per-vertex RGB colors in range $[0, 1]$.
            Defaults to None.

    Raises:
        TypeError: If `vertices`, `faces`, or `colors` are not torch.Tensor or numpy.ndarray.
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    elif isinstance(vertices, np.ndarray):
        vertices = vertices
    else:
        raise TypeError("vertices must be a torch.Tensor or numpy.ndarray")
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    elif isinstance(faces, np.ndarray):
        faces = faces
    else:
        raise TypeError("faces must be a torch.Tensor or numpy.ndarray")
    
    vc = None
    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        elif isinstance(colors, np.ndarray):
            colors = colors
        else:
            raise TypeError("colors must be a torch.Tensor or numpy.ndarray")
        vc = (colors * 255.0).clip(0, 255).astype(np.uint8)
        
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vc, process=False)
    mesh.export(filepath)