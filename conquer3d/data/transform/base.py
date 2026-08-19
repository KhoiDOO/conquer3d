"""Base classes and compositional containers for 3D geometric data transforms.

This module provides probabilistic transformation interfaces (`BaseTransform`),
sequential composition pipelines (`Sequence`), and mesh-specific pipelines (`MeshSequence`).
"""

from typing import List, Callable, Dict, Any, Tuple
import random
import torch


class BaseTransform(object):
    """Abstract base class for probabilistic geometric data transforms.

    Attributes:
        p (float): Execution probability in $[0.0, 1.0]$.
    """

    def __init__(self, p: float = 1.0) -> None:
        """Initializes the BaseTransform instance.

        Args:
            p (float, optional): Probability of applying this transform. Defaults to 1.0.
        """
        self.p = p

    def __call__(self, **data: Any) -> Dict[str, Any]:
        """Conditionally applies the transformation with probability `p`.

        Args:
            **data: Key-value geometric data pairs (e.g. `vertices=...`).

        Returns:
            Dict[str, Any]: Transformed or unmodified data dictionary.
        """
        if random.random() <= self.p:
            return self.apply(**data)
        return data
        
    def apply(self, **data: Any) -> Dict[str, Any]:
        """Applies the underlying geometric transformation.

        Args:
            **data: Geometric data dictionary.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Transform must implement apply")


class Sequence(BaseTransform):
    """Composes multiple transforms sequentially.

    Attributes:
        transforms (List[BaseTransform]): Ordered sequence of transforms.
        shuffle (bool): If True, randomly permutes transform order per invocation.
    """

    def __init__(self, transforms: List[BaseTransform], shuffle: bool = False, p: float = 1.0) -> None:
        """Initializes the Sequence container.

        Args:
            transforms (List[BaseTransform]): List of transform objects to apply in order.
            shuffle (bool, optional): If True, shuffles execution order. Defaults to False.
            p (float, optional): Execution probability. Defaults to 1.0.
        """
        super().__init__(p=p)
        self.transforms = transforms
        self.shuffle = shuffle

    def apply(self, **data: Any) -> Dict[str, Any]:
        """Applies each child transform sequentially.

        Args:
            **data: Geometric data dictionary.

        Returns:
            Dict[str, Any]: Processed data dictionary.
        """
        transforms_to_apply = self.transforms.copy()
        if self.shuffle:
            random.shuffle(transforms_to_apply)
            
        for transform in transforms_to_apply:
            data = transform(**data)
            
        return data


class MeshSequence(Sequence):
    """Mesh-specific transform sequence operating directly on `(vertices, faces)` tuples."""

    def __call__(self, vertices: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies sequence with probability `p`.

        Args:
            vertices (torch.Tensor): Vertex coordinate tensor `(V, 3)`.
            faces (torch.Tensor): Face indices tensor `(F, 3)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed `(vertices, faces)`.
        """
        if random.random() <= self.p:
            return self.apply(vertices, faces)
        return vertices, faces

    def apply(self, vertices: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes the sequence of mesh transformations.

        Args:
            vertices (torch.Tensor): Vertex coordinate tensor `(V, 3)`.
            faces (torch.Tensor): Face indices tensor `(F, 3)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed `(vertices, faces)`.
        """
        data = super().apply(vertices=vertices, faces=faces)
        return data.get('vertices', vertices), data.get('faces', faces)
