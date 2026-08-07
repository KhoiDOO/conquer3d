import torch
from typing import Optional

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
    """
    Executes TSDF Volume Integration for a single RGB-D view in-place.

    Args:
        grid_vertices (torch.Tensor): (N, 3) tensor of global grid vertex positions (CUDA, Contiguous).
        sdf (torch.Tensor): (N,) tensor of TSDF values to be updated in-place (CUDA, Contiguous).
        weight (torch.Tensor): (N,) tensor of running average weights to be updated in-place (CUDA, Contiguous).
        depth_image (torch.Tensor): (H, W) tensor of the depth image in meters (CUDA, Contiguous).
        extrinsics (torch.Tensor): (4, 4) World-to-Camera (w2c) extrinsic matrix.
        intrinsics (torch.Tensor): (3, 3) Camera intrinsic matrix.
        color (torch.Tensor, optional): (N, 3) tensor of vertex colors to be updated in-place (CUDA, Contiguous). Defaults to None.
        color_image (torch.Tensor, optional): (H, W, 3) tensor of the RGB image (CUDA, Contiguous). Defaults to None.
        trunc_margin (float, optional): Truncation distance for the TSDF in meters. Defaults to 0.04.
        mode (int, optional): 1 for True Euclidean SDF, 0 for Projective SDF shortcut. Defaults to 1.
    """
    
    # Ensure in-place tensors are properly formatted before passing to C++
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
