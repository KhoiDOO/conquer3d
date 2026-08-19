"""Base class for 3D mesh PyTorch datasets.

This module defines the abstract interface `BaseMeshDataset` for datasets
providing 3D meshes with vertex coordinates, triangle topologies, and data transforms.
"""

from typing import Callable, Optional, Tuple
import torch
from torch.utils.data import Dataset


class BaseMeshDataset(Dataset):
    """Abstract base PyTorch Dataset class for loading 3D meshes.

    Attributes:
        root (str): Root filesystem path containing mesh data files.
        transform (Callable, optional): Optional data transformation function
            applied to `(vertices, faces)` tuples.
    """

    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        """Initializes the BaseMeshDataset instance.

        Args:
            root (str): Root directory path containing mesh files.
            transform (Callable, optional): Optional callable transformation applied
                to vertices and faces upon retrieval. Defaults to None.
        """
        super().__init__()
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        """int: Number of mesh samples in the dataset."""
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the mesh at the given index.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - vertices (torch.Tensor): (V, 3) float32 coordinates.
                - faces (torch.Tensor): (F, 3) int32 or int64 face indices.
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
