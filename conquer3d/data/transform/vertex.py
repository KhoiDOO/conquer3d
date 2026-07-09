import torch
import random
from .base import BaseTransform
from .ops import rotation, scale

class Rotation(BaseTransform):
    def __init__(self, axis='z', degree_range=(-180, 180), p=1.0):
        """
        Args:
            axis (str or list): The axis or list of axes to rotate around ('x', 'y', 'z').
            degree_range (tuple): Range of degrees to randomly sample from (min, max).
            p (float): Probability of applying the transform.
        """
        super().__init__(p=p)
        self.axis = axis
        self.degree_range = degree_range
        
    def apply(self, **data):
        if 'vertices' not in data:
            raise KeyError("Rotation transform requires 'vertices' in data")
            
        vertices = data['vertices']
        
        # Support multiple axes in sequence
        axes = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
        
        for ax in axes:
            # Sample a random degree for each axis
            degree = random.uniform(self.degree_range[0], self.degree_range[1])
            vertices = rotation(vertices, ax, degree)
            
        data['vertices'] = vertices
        return data

class Scale(BaseTransform):
    def __init__(self, scale_range=(0.8, 1.2), p=1.0):
        """
        Args:
            scale_range (tuple): Range of scale factors to randomly sample from (min, max).
            p (float): Probability of applying the transform.
        """
        super().__init__(p=p)
        self.scale_range = scale_range
        
    def apply(self, **data):
        if 'vertices' not in data:
            raise KeyError("Scale transform requires 'vertices' in data")
            
        vertices = data['vertices']
        
        # Sample a uniform random scale factor
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        
        data['vertices'] = scale(vertices, scale_factor)
        return data
