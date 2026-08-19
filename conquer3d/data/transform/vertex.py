"""Object-oriented vertex transformation classes.

This module defines `Rotation` and `Scale` transformation modules operating
on dictionary data containing 3D vertex coordinates.
"""

from typing import Union, List, Tuple, Dict, Any
import random
import torch
from .base import BaseTransform
from .ops import rotation, scale


class Rotation(BaseTransform):
    """Random 3D spatial rotation transform.

    Attributes:
        axis (Union[str, List[str]]): Target axis or sequence of axes (`'x'`, `'y'`, `'z'`).
        degree_range (Tuple[float, float]): Range of angles `(min_deg, max_deg)` to sample from.
    """

    def __init__(
        self,
        axis: Union[str, List[str], Tuple[str, ...]] = 'z',
        degree_range: Tuple[float, float] = (-180.0, 180.0),
        p: float = 1.0
    ) -> None:
        """Initializes the Rotation transform.

        Args:
            axis (Union[str, List[str], Tuple[str, ...]], optional): Axis or list of axes to rotate around.
                Defaults to `'z'`.
            degree_range (Tuple[float, float], optional): Range of degrees `(min, max)`. Defaults to `(-180, 180)`.
            p (float, optional): Application probability. Defaults to 1.0.
        """
        super().__init__(p=p)
        self.axis = axis
        self.degree_range = degree_range
        
    def apply(self, **data: Any) -> Dict[str, Any]:
        """Applies random rotation to `data['vertices']`.

        Args:
            **data: Dictionary containing key `'vertices'`.

        Returns:
            Dict[str, Any]: Updated data dictionary with rotated vertices.

        Raises:
            KeyError: If `'vertices'` is not in `data`.
        """
        if 'vertices' not in data:
            raise KeyError("Rotation transform requires 'vertices' in data")
            
        vertices = data['vertices']
        axes = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
        
        for ax in axes:
            degree = random.uniform(self.degree_range[0], self.degree_range[1])
            vertices = rotation(vertices, ax, degree)
            
        data['vertices'] = vertices
        return data


class Scale(BaseTransform):
    """Random uniform scaling transform.

    Attributes:
        scale_range (Tuple[float, float]): Scaling range `(min_scale, max_scale)`.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), p: float = 1.0) -> None:
        """Initializes the Scale transform.

        Args:
            scale_range (Tuple[float, float], optional): Range of scale factors `(min, max)`.
                Defaults to `(0.8, 1.2)`.
            p (float, optional): Application probability. Defaults to 1.0.
        """
        super().__init__(p=p)
        self.scale_range = scale_range
        
    def apply(self, **data: Any) -> Dict[str, Any]:
        """Applies random uniform scaling to `data['vertices']`.

        Args:
            **data: Dictionary containing key `'vertices'`.

        Returns:
            Dict[str, Any]: Updated data dictionary with scaled vertices.

        Raises:
            KeyError: If `'vertices'` is not in `data`.
        """
        if 'vertices' not in data:
            raise KeyError("Scale transform requires 'vertices' in data")
            
        vertices = data['vertices']
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        data['vertices'] = scale(vertices, scale_factor)
        return data
