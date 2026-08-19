"""Object File Format (OFF) 3D mesh format read I/O routines.

This module provides loaders for Geomview OFF (Object File Format) files,
returning PyTorch vertex coordinate and face index tensors.
"""

from typing import Tuple, Union, BinaryIO
import torch
import trimesh


def read_off(filepath_or_filelike: Union[str, BinaryIO]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reads a Geomview OFF 3D mesh file and returns vertex coordinates and triangle faces.
    
    Args:
        filepath_or_filelike (Union[str, BinaryIO]): File path string or binary stream.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - vertices (torch.Tensor): Float32 tensor of shape `(V, 3)` containing vertex coordinates.
            - faces (torch.Tensor): Int64 tensor of shape `(F, 3)` containing triangle face indices.

    Example:
        >>> from conquer3d.io import read_off
        >>> verts, faces = read_off("model.off")
    """
    # trimesh handles both string paths and binary file streams natively
    mesh = trimesh.load(filepath_or_filelike, file_type='off', process=False, force='mesh')
    
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
        
    return vertices, faces
