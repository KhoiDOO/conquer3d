import torch
import math

def rotation(vertices, rotation_axis, rotation_degree):
    """
    Rotates vertices around a given axis by a specific degree.
    
    Args:
        vertices (torch.Tensor): Shape (N, 3) representing 3D coordinates.
        rotation_axis (str): 'x', 'y', or 'z'.
        rotation_degree (float): Degree to rotate by (in degrees).
        
    Returns:
        torch.Tensor: The rotated vertices of the same type.
    """
    theta = math.radians(rotation_degree)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    device = vertices.device
    dtype = vertices.dtype
    
    # Define rotation matrices based on the axis directly as PyTorch tensors
    if rotation_axis == 'x':
        rot_mat = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, cos_t, -sin_t],
            [0.0, sin_t, cos_t]
        ], dtype=dtype, device=device)
    elif rotation_axis == 'y':
        rot_mat = torch.tensor([
            [cos_t, 0.0, sin_t],
            [0.0, 1.0, 0.0],
            [-sin_t, 0.0, cos_t]
        ], dtype=dtype, device=device)
    elif rotation_axis == 'z':
        rot_mat = torch.tensor([
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=dtype, device=device)
    else:
        raise ValueError("rotation_axis must be 'x', 'y', or 'z'")
        
    # Compute dot product
    return torch.matmul(vertices, rot_mat.T)

def scale(vertices, scale_factor):
    """
    Scales vertices by a given factor.
    
    Args:
        vertices (torch.Tensor): Shape (N, 3) representing 3D coordinates.
        scale_factor (float, tuple, or list): The factor(s) to scale by.
        
    Returns:
        torch.Tensor: The scaled vertices.
    """
    if isinstance(scale_factor, (list, tuple)):
        scale_factor = torch.tensor(scale_factor, dtype=vertices.dtype, device=vertices.device)
        
    return vertices * scale_factor
