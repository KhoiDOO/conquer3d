"""Point cloud geometric distance metrics and loss functions.

GPU nearest-neighbour distances between 3D point sets: Chamfer and Hausdorff, each
one-sided and symmetric.

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


class _OneSidedChamferFunction(torch.autograd.Function):
    """PyTorch autograd Function for differentiable one-sided Chamfer distance."""

    @staticmethod
    def forward(
        ctx,
        query_points: torch.Tensor,
        reference_points: torch.Tensor,
        squared: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_points_c = query_points.contiguous()
        reference_points_c = reference_points.contiguous()

        distances, indices = _C.one_sided_chamfer_distance(query_points_c, reference_points_c)

        if not squared:
            distances = torch.sqrt(torch.clamp(distances, min=1e-26))

        ctx.save_for_backward(query_points_c, reference_points_c, indices)
        ctx.squared = squared
        ctx.mark_non_differentiable(indices)

        return distances, indices

    @staticmethod
    def backward(
        ctx,
        grad_distances: torch.Tensor,
        grad_indices: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], None]:
        query_points, reference_points, indices = ctx.saved_tensors
        squared = ctx.squared

        compute_grad_query = ctx.needs_input_grad[0]
        compute_grad_reference = ctx.needs_input_grad[1]

        if not compute_grad_query and not compute_grad_reference:
            return None, None, None

        grad_distances_c = grad_distances.contiguous()

        grad_query, grad_reference = _C.one_sided_chamfer_distance_backward(
            query_points,
            reference_points,
            indices,
            grad_distances_c,
            squared,
            compute_grad_query,
            compute_grad_reference
        )

        return grad_query, grad_reference, None


def one_sided_chamfer_distance(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    squared: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the one-sided nearest-neighbor distance from query points to reference points.

    Nearest neighbours come from a GPU KD-Tree. Differentiable with respect to both point
    sets through an analytical CUDA backward kernel.

    Args:
        query_points (torch.Tensor): Point coordinates of shape `(N, 3)` with dtype
            `torch.float32` on CUDA device.
        reference_points (torch.Tensor): Reference point coordinates of shape `(M, 3)`
            with dtype `torch.float32` on CUDA device. Must be non-empty.
        squared (bool, optional): If True, returns squared Euclidean distances $\\|x - y\\|^2$.
            If False, returns true Euclidean distances $\\|x - y\\|$ (clamped for stability).
            Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - distances (torch.Tensor): Float32 tensor of shape `(N,)` containing the minimum
              distance from each query point to the reference point set, squared when
              `squared=True`.
            - indices (torch.Tensor): Int64 tensor of shape `(N,)` containing the index
              of the closest reference point for each query point.

    Raises:
        AssertionError: If tensors are not on CUDA, not float32, or not 3D.
        ValueError: If `reference_points` is empty, since a nearest neighbour in an empty
            set does not exist.

    Note:
        An empty `query_points` yields empty outputs, so chunking a large query set needs no
        special case for a trailing empty chunk. An empty `reference_points` is rejected
        because its $+\\infty$ distance would propagate into `NaN` far from its cause.
    """
    assert query_points.is_cuda and reference_points.is_cuda, "Points must be on CUDA"
    assert query_points.dtype == torch.float32 and reference_points.dtype == torch.float32, "Points must be float32"
    assert query_points.shape[1] == 3 and reference_points.shape[1] == 3, "Points must be 3D"

    if reference_points.shape[0] == 0:
        raise ValueError(
            "reference_points is empty; the nearest-neighbour distance is undefined. "
            "Guard the call, or supply at least one reference point."
        )

    if query_points.shape[0] == 0:
        return (
            torch.empty((0,), device=query_points.device, dtype=torch.float32),
            torch.empty((0,), device=query_points.device, dtype=torch.int64)
        )

    return _OneSidedChamferFunction.apply(query_points, reference_points, squared)


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    squared: bool = True,
    return_indices: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Computes the symmetric bidirectional Chamfer distance between two 3D point clouds.

    Calculates the mean nearest-neighbor distance in both directions:
    $$\\mathcal{L}_{CD}(X, Y) = \\frac{1}{|X|} \\sum_{x \\in X} \\min_{y \\in Y} \\|x - y\\|^p + \\frac{1}{|Y|} \\sum_{y \\in Y} \\min_{x \\in X} \\|y - x\\|^p$$

    where $p = 2$ when `squared=True` (the default) and $p = 1$ otherwise.

    Fully differentiable with respect to both `x` and `y` via PyTorch autograd.

    Args:
        x (torch.Tensor): First point cloud of shape `(N, 3)` on CUDA device. Must be non-empty.
        y (torch.Tensor): Second point cloud of shape `(M, 3)` on CUDA device. Must be non-empty.
        squared (bool, optional): If True, computes squared distances ($p = 2$); if False,
            Euclidean distances ($p = 1$). Defaults to True.
        return_indices (bool, optional): If True, also returns the nearest neighbor
            indices for both directions. Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            - If `return_indices=False`: Scalar tensor containing the symmetric Chamfer loss.
            - If `return_indices=True`: Tuple of `(loss, idx_x_to_y, idx_y_to_x)` where
              `idx_x_to_y` has shape `(N,)` and `idx_y_to_x` has shape `(M,)`.

    Raises:
        ValueError: If either point cloud is empty. The symmetric distance averages over both
            sets, so an empty one makes it undefined rather than zero.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError(
            f"chamfer_distance needs two non-empty point clouds, got |x|={x.shape[0]} "
            f"and |y|={y.shape[0]}."
        )

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
    $$h(X, Y) = \\max_{x \\in X} \\min_{y \\in Y} \\|x - y\\|^p$$

    where $p = 2$ when `squared=True` (the default) and $p = 1$ otherwise. Note the default
    therefore returns a *squared* distance, not a metric one.

    Args:
        query_points (torch.Tensor): Point coordinates of shape `(N, 3)` on CUDA. Must be non-empty.
        reference_points (torch.Tensor): Reference point coordinates of shape `(M, 3)` on CUDA.
            Must be non-empty.
        squared (bool, optional): If True, computes squared distance ($p = 2$); if False,
            Euclidean distance ($p = 1$). Defaults to True.
        return_indices (bool, optional): If True, returns the indices of the worst-case pair.
            Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            - If `return_indices=False`: Scalar float tensor of maximum distance.
            - If `return_indices=True`: Tuple `(max_dist, query_idx, ref_idx)` identifying
              the query point index and closest reference point index yielding the maximum distance.

    Raises:
        ValueError: If either point set is empty, since the inner minimum and the outer
            maximum are both undefined over an empty set.
    """
    if query_points.shape[0] == 0 or reference_points.shape[0] == 0:
        raise ValueError(
            f"one_sided_hausdorff_distance needs two non-empty point sets, got "
            f"|query|={query_points.shape[0]} and |reference|={reference_points.shape[0]}."
        )

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
    $$H(X, Y) = \\max\\big(h(X, Y),\\; h(Y, X)\\big), \\qquad
    h(X, Y) = \\max_{x \\in X} \\min_{y \\in Y} \\|x - y\\|^p$$

    where $p = 2$ when `squared=True` (the default) and $p = 1$ otherwise. Squaring is
    monotone, so the selected pair is the same either way; only the returned magnitude differs.

    Args:
        x (torch.Tensor): First point set of shape `(N, 3)` on CUDA device. Must be non-empty.
        y (torch.Tensor): Second point set of shape `(M, 3)` on CUDA device. Must be non-empty.
        squared (bool, optional): If True, computes squared distance ($p = 2$); if False,
            Euclidean distance ($p = 1$). Defaults to True.
        return_indices (bool, optional): If True, returns indices of the point pair
            responsible for the maximum Hausdorff distance. Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            - If `return_indices=False`: Scalar tensor with symmetric Hausdorff distance.
            - If `return_indices=True`: Tuple `(max_dist, x_idx, y_idx)` containing the
              maximal distance and corresponding point indices in `x` and `y`.

    Raises:
        ValueError: If either point set is empty, since each directed term is undefined there.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError(
            f"hausdorff_distance needs two non-empty point sets, got |x|={x.shape[0]} "
            f"and |y|={y.shape[0]}."
        )

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
