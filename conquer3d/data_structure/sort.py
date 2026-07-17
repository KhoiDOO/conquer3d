import torch
import conquer3d._C as _C

def z_curve_sort(points: torch.Tensor):
    """
    Computes the Z-curve (Morton code) for a batch of points and sorts them.
    Assumes points are normalized between [0, 1]. The function will clamp points outside this range.
    
    Args:
        points (torch.Tensor): Tensor of shape (..., 3) containing the point coordinates.
        
    Returns:
        sorted_points (torch.Tensor): Points sorted by their Z-curve value.
        sorted_indices (torch.Tensor): The indices used to sort the points.
        inverse_indices (torch.Tensor): The indices to reverse the sorting.
    """
    assert points.is_cuda, "Points must be on CUDA"
    assert points.dtype == torch.float32, "Points must be float32"
    assert points.shape[-1] == 3, "Points must be 3D"
    
    # Compute the Morton codes for each point
    codes = _C.compute_zcurve(points.contiguous())
    
    # codes has shape (..., N). We always sort along the last dimension (dim=-1)
    
    sorted_codes, sorted_indices = torch.sort(codes, dim=-1)
    
    # Gather the points using the sorted indices
    # We need to expand sorted_indices to match points: (..., N, 3)
    dim = -2 if points.dim() > 1 else -1
    expanded_indices = sorted_indices.unsqueeze(-1).expand_as(points)
    sorted_points = torch.gather(points, dim=dim, index=expanded_indices)
    
    # Compute the inverse indices (scatter)
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    
    return sorted_points, sorted_indices, inverse_indices
