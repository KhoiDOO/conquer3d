"""Batched 3D triangle mesh data container.

This module provides `BTriangleMesh`, a packed container for variable-sized
batches of 3D triangle meshes with per-element batch assignment maps.
"""

from typing import Union
import torch


class BTriangleMesh:
    """Container for concatenated batches of variable-sized 3D triangle meshes.

    Attributes:
        vertices (torch.Tensor): Concatenated float32 vertex coordinates of shape `(Total_V, 3)`.
        faces (torch.Tensor): Concatenated triangle face indices of shape `(Total_F, 3)`.
        vertbids (torch.Tensor): Int32 tensor of shape `(Total_V,)` mapping each vertex to its batch ID.
        facebids (torch.Tensor): Int32 tensor of shape `(Total_F,)` mapping each face to its batch ID.
        batch_size (int): Total number of meshes in the batch.
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertbids: torch.Tensor,
        facebids: torch.Tensor,
        batch_size: int
    ) -> None:
        """Initializes the BTriangleMesh instance.

        Args:
            vertices (torch.Tensor): Concatenated vertex coordinates `(Total_V, 3)`.
            faces (torch.Tensor): Concatenated face indices `(Total_F, 3)`.
            vertbids (torch.Tensor): Batch ID index map for vertices `(Total_V,)`.
            facebids (torch.Tensor): Batch ID index map for faces `(Total_F,)`.
            batch_size (int): Number of meshes packed in this batch.
        """
        self.vertices = vertices
        self.faces = faces
        self.vertbids = vertbids
        self.facebids = facebids
        self.batch_size = batch_size

    def to(self, device: Union[str, torch.device]) -> 'BTriangleMesh':
        """Moves all internal tensors to the target device in-place.

        Args:
            device (Union[str, torch.device]): Target compute device.

        Returns:
            BTriangleMesh: Self instance after moving tensors.
        """
        self.vertices = self.vertices.to(device)
        self.faces = self.faces.to(device)
        self.vertbids = self.vertbids.to(device)
        self.facebids = self.facebids.to(device)
        return self
        
    def cuda(self, non_blocking: bool = False) -> 'BTriangleMesh':
        """Moves all internal tensors to CUDA memory in-place.

        Args:
            non_blocking (bool, optional): If True, performs asynchronous copy. Defaults to False.

        Returns:
            BTriangleMesh: Self instance after moving tensors to CUDA.
        """
        self.vertices = self.vertices.cuda(non_blocking=non_blocking)
        self.faces = self.faces.cuda(non_blocking=non_blocking)
        self.vertbids = self.vertbids.cuda(non_blocking=non_blocking)
        self.facebids = self.facebids.cuda(non_blocking=non_blocking)
        return self
