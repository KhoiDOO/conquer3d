import torch
import conquer3d._C as _C

def one_sided_chamfer_distance(query_points: torch.Tensor, reference_points: torch.Tensor, squared: bool = True):
    assert query_points.is_cuda and reference_points.is_cuda, "Points must be on CUDA"
    assert query_points.dtype == torch.float32 and reference_points.dtype == torch.float32, "Points must be float32"
    assert query_points.shape[1] == 3 and reference_points.shape[1] == 3, "Points must be 3D"
    
    query_points_c = query_points.contiguous()
    reference_points_c = reference_points.contiguous()
    
    distances, indices = _C.one_sided_chamfer_distance(query_points_c, reference_points_c)
    
    if not squared:
        # clamp to avoid NaN from floating point inaccuracies
        distances = torch.sqrt(torch.clamp(distances, min=1e-12))
        
    return distances, indices

def chamfer_distance(x: torch.Tensor, y: torch.Tensor, squared: bool = True, return_indices: bool = False):
    dist_x_to_y, idx_x_to_y = one_sided_chamfer_distance(x, y, squared=squared)
    dist_y_to_x, idx_y_to_x = one_sided_chamfer_distance(y, x, squared=squared)
    
    loss = dist_x_to_y.mean() + dist_y_to_x.mean()
    
    if return_indices:
        return loss, idx_x_to_y, idx_y_to_x
    return loss
