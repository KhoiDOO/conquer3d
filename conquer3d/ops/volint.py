"""Single-view and stream-based TSDF volumetric integration operations.

This module provides GPU-accelerated volumetric Truncated Signed Distance Function (TSDF)
and running RGB feature integration for 3D camera reconstruction pipelines.
"""

from typing import Optional
import torch

from .._C import single_view_volume_integral as _single_view_volume_integral


def single_view_volume_integral(
    grid_vertices: torch.Tensor,
    sdf: torch.Tensor,
    weight: torch.Tensor,
    depth_image: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    color: Optional[torch.Tensor] = None,
    color_image: Optional[torch.Tensor] = None,
    trunc_margin: float = 0.04,
    mode: int = 1
) -> None:
    """Executes volumetric TSDF and color integration for a single RGB-D view in-place on CUDA.

    Projects every 3D voxel coordinate into the camera coordinate frame using `extrinsics`
    and `intrinsics`, evaluates the signed distance along the camera ray against `depth_image`,
    and updates the running weighted average `sdf`, `weight`, and `color` in-place.

    Args:
        grid_vertices (torch.Tensor): Float32 tensor of shape `(N, 3)` with 3D grid vertex positions on CUDA.
        sdf (torch.Tensor): Float32 tensor of shape `(N,)` with running TSDF values updated in-place on CUDA.
        weight (torch.Tensor): Float32 tensor of shape `(N,)` with running average weights updated in-place on CUDA.
        depth_image (torch.Tensor): Float32 tensor of shape `(H, W)` with depth values in meters on CUDA.
        extrinsics (torch.Tensor): Float32 tensor of shape `(4, 4)` with World-to-Camera (W2C) transformation.
        intrinsics (torch.Tensor): Float32 tensor of shape `(3, 3)` with pinhole camera intrinsic matrix.
        color (torch.Tensor, optional): Float32 tensor of shape `(N, 3)` with running vertex RGB colors on CUDA.
            Defaults to None.
        color_image (torch.Tensor, optional): Float32 tensor of shape `(H, W, 3)` with RGB image on CUDA.
            Defaults to None.
        trunc_margin (float, optional): Truncation distance $\\mu$ in meters. Defaults to 0.04.
        mode (int, optional): Integration mode:
            - 1: True Euclidean ray distance (default).
            - 0: Projective z-depth distance.

    Raises:
        ValueError: If input tensors are not on CUDA or not contiguous.

    Example:
        >>> from conquer3d.ops import single_view_volume_integral
        >>> single_view_volume_integral(grid_verts, sdf, weights, depth, w2c, K, trunc_margin=0.04)
    """
    if not all(t.is_cuda and t.is_contiguous() for t in [grid_vertices, sdf, weight, depth_image]):
        raise ValueError("grid_vertices, sdf, weight, and depth_image must be contiguous CUDA tensors.")
        
    if color is not None and not (color.is_cuda and color.is_contiguous()):
        raise ValueError("color tensor must be a contiguous CUDA tensor.")
        
    if color_image is not None and not (color_image.is_cuda and color_image.is_contiguous()):
        raise ValueError("color_image tensor must be a contiguous CUDA tensor.")

    _single_view_volume_integral(
        grid_vertices,
        sdf,
        weight,
        color,
        depth_image,
        color_image,
        extrinsics,
        intrinsics,
        float(trunc_margin),
        int(mode)
    )
