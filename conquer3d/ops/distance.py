"""Point cloud geometric distance metrics and loss functions.

This module provides GPU-accelerated nearest-neighbor geometric distance
functions between 3D point sets, including one-sided Chamfer distance,
symmetric bidirectional Chamfer distance, one-sided Hausdorff distance,
and symmetric bidirectional Hausdorff distance.

Example:
    >>> import torch
    >>> from conquer3d.ops import chamfer_distance, hausdorff_distance
    >>> x = torch.randn(1000, 3, device='cuda')
    >>> y = torch.randn(1500, 3, device='cuda')
    >>> cd_loss = chamfer_distance(x, y)
    >>> hd_dist = hausdorff_distance(x, y)
"""

from typing import Tuple, Union
import torch
import conquer3d._C as _C


def one_sided_chamfer_distance(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    squared: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the one-sided nearest-neighbor distance from query points to reference points.

    For each point in `query_points`, finds the closest point in `reference_points`
    using GPU KD-Tree spatial acceleration.

    Args:
        query_points (torch.Tensor): Point coordinates of shape `(N, 3)` with dtype
            `torch.float32` on CUDA device.
        reference_points (torch.Tensor): Reference point coordinates of shape `(M, 3)`
            with dtype `torch.float32` on CUDA device.
        squared (bool, optional): If True, returns squared Euclidean distances.
            If False, returns true Euclidean distances (clamped for stability).
            Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - distances (torch.Tensor): Float32 tensor of shape `(N,)` containing the
              minimum distance from each query point to the reference point set.
            - indices (torch.Tensor): Int64 tensor of shape `(N,)` containing the index
              of the closest reference point for each query point.

    Raises:
        AssertionError: If tensors are not on CUDA, not float32, or not 3D.
    """
    assert query_points.is_cuda and reference_points.is_cuda, "Points must be on CUDA"
    assert query_points.dtype == torch.float32 and reference_points.dtype == torch.float32, "Points must be float32"
    assert query_points.shape[1] == 3 and reference_points.shape[1] == 3, "Points must be 3D"
    
    if query_points.shape[0] == 0:
        return (
            torch.empty((0,), device=query_points.device, dtype=torch.float32),
            torch.empty((0,), device=query_points.device, dtype=torch.int64)
        )
    if reference_points.shape[0] == 0:
        return (
            torch.full((query_points.shape[0],), float('inf'), device=query_points.device, dtype=torch.float32),
            torch.full((query_points.shape[0],), -1, device=query_points.device, dtype=torch.int64)
        )

    query_points_c = query_points.contiguous()
    reference_points_c = reference_points.contiguous()
    
    distances, indices = _C.one_sided_chamfer_distance(query_points_c, reference_points_c)
    
    if not squared:
        # Critically small epsilon clamp to prevent NaN from floating-point underflow
        distances = torch.sqrt(torch.clamp(distances, min=1e-26))
        
    return distances, indices


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    squared: bool = True,
    return_indices: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Computes the symmetric bidirectional Chamfer distance between two 3D point clouds.

    Calculates the mean squared or Euclidean nearest-neighbor distance in both directions:
    $$\\mathcal{L}_{CD}(X, Y) = \\frac{1}{|X|} \\sum_{x \\in X} \\min_{y \\in Y} \\|x - y\\|^2 + \\frac{1}{|Y|} \\sum_{y \\in Y} \\min_{x \\in X} \\|y - x\\|^2$$

    Args:
        x (torch.Tensor): First point cloud of shape `(N, 3)` on CUDA device.
        y (torch.Tensor): Second point cloud of shape `(M, 3)` on CUDA device.
        squared (bool, optional): If True, computes squared distances. Defaults to True.
        return_indices (bool, optional): If True, also returns the nearest neighbor
            indices for both directions. Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            - If `return_indices=False`: Scalar tensor containing the symmetric Chamfer loss.
            - If `return_indices=True`: Tuple of `(loss, idx_x_to_y, idx_y_to_x)` where
              `idx_x_to_y` has shape `(N,)` and `idx_y_to_x` has shape `(M,)`.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        zero_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if return_indices:
            return (
                zero_loss,
                torch.empty((x.shape[0],), dtype=torch.int64, device=x.device),
                torch.empty((y.shape[0],), dtype=torch.int64, device=y.device)
            )
        return zero_loss

    dist_x_to_y, idx_x_to_y = one_sided_chamfer_distance(x, y, squared=squared)
    dist_y_to_x, idx_y_to_x = one_sided_chamfer_distance(y, x, squared=squared)
    
    loss = dist_x_to_y.mean() + dist_y_to_x.mean()
    
    if return_indices:
        return loss, idx_x_to_y, idx_y_to_x
    return loss


def one_sided_hausdorff_distance(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    squared: bool = True,
    return_indices: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Computes the one-sided directed Hausdorff distance from query points to reference points.

    Finds the maximum of the minimum distances from each point in `query_points` to `reference_points`:
    $$h(X, Y) = \\max_{x \\in X} \\min_{y \\in Y} \\|x - y\\|$$

    Args:
        query_points (torch.Tensor): Point coordinates of shape `(N, 3)` on CUDA.
        reference_points (torch.Tensor): Reference point coordinates of shape `(M, 3)` on CUDA.
        squared (bool, optional): If True, computes squared distance. Defaults to True.
        return_indices (bool, optional): If True, returns the indices of the worst-case pair.
            Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            - If `return_indices=False`: Scalar float tensor of maximum distance.
            - If `return_indices=True`: Tuple `(max_dist, query_idx, ref_idx)` identifying
              the query point index and closest reference point index yielding the maximum distance.
    """
    if query_points.shape[0] == 0 or reference_points.shape[0] == 0:
        zero_dist = torch.tensor(0.0, device=query_points.device, dtype=query_points.dtype)
        if return_indices:
            return (
                zero_dist,
                torch.tensor(-1, device=query_points.device, dtype=torch.int64),
                torch.tensor(-1, device=query_points.device, dtype=torch.int64)
            )
        return zero_dist

    distances, indices = one_sided_chamfer_distance(query_points, reference_points, squared=squared)
    
    max_dist, max_idx = torch.max(distances, dim=0)
    
    if return_indices:
        return max_dist, max_idx, indices[max_idx]
    return max_dist


def hausdorff_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    squared: bool = True,
    return_indices: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Computes the symmetric bidirectional Hausdorff distance between two 3D point clouds.

    Calculates the maximum directed Hausdorff distance between set $X$ and set $Y$:
    $$H(X, Y) = \\max(h(X, Y), h(Y, X))$$

    Args:
        x (torch.Tensor): First point set of shape `(N, 3)` on CUDA device.
        y (torch.Tensor): Second point set of shape `(M, 3)` on CUDA device.
        squared (bool, optional): If True, computes squared distance. Defaults to True.
        return_indices (bool, optional): If True, returns indices of the point pair
            responsible for the maximum Hausdorff distance. Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            - If `return_indices=False`: Scalar tensor with symmetric Hausdorff distance.
            - If `return_indices=True`: Tuple `(max_dist, x_idx, y_idx)` containing the
              maximal distance and corresponding point indices in `x` and `y`.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        zero_dist = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if return_indices:
            return (
                zero_dist,
                torch.tensor(-1, device=x.device, dtype=torch.int64),
                torch.tensor(-1, device=x.device, dtype=torch.int64)
            )
        return zero_dist

    if return_indices:
        dist_x_to_y, idx_x1, idx_y1 = one_sided_hausdorff_distance(x, y, squared=squared, return_indices=True)
        dist_y_to_x, idx_y2, idx_x2 = one_sided_hausdorff_distance(y, x, squared=squared, return_indices=True)
        
        if dist_x_to_y > dist_y_to_x:
            return dist_x_to_y, idx_x1, idx_y1
        else:
            return dist_y_to_x, idx_x2, idx_y2
    else:
        dist_x_to_y = one_sided_hausdorff_distance(x, y, squared=squared, return_indices=False)
        dist_y_to_x = one_sided_hausdorff_distance(y, x, squared=squared, return_indices=False)
        
        if dist_x_to_y > dist_y_to_x:
            return dist_x_to_y
        else:
            return dist_y_to_x
