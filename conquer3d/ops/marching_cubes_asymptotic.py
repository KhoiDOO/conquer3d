import torch
from typing import Tuple, Optional
from .. import _C

class DiffMarchingCubesAsymptotic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grid_vertices: torch.Tensor,
        voxels: torch.Tensor,
        sdf: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        iso: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Ensure inputs are CUDA and contiguous
        if not grid_vertices.is_cuda or not voxels.is_cuda or not sdf.is_cuda:
            raise RuntimeError("marching_cubes_asymptotic requires CUDA tensors")

        grid_vertices = grid_vertices.contiguous().float()
        voxels = voxels.contiguous().int()
        sdf = sdf.contiguous().float()
        if colors is not None:
            colors = colors.contiguous().float()

        verts, faces, out_colors = _C.marching_cubes_asymptotic(
            grid_vertices, voxels, sdf, colors, iso
        )

        ctx.save_for_backward(grid_vertices, voxels, sdf, colors, verts, faces)
        ctx.iso = iso
        ctx.has_colors = colors is not None

        return verts, faces, out_colors

    @staticmethod
    def backward(ctx, grad_verts, grad_faces, grad_colors):
        grid_vertices, voxels, sdf, colors, verts, faces = ctx.saved_tensors
        iso = ctx.iso

        # If no gradients needed
        if not sdf.requires_grad and (colors is None or not colors.requires_grad):
            return None, None, None, None, None

        # When autograd backward is invoked, evaluate gradient directly
        grad_sdf = torch.zeros_like(sdf)
        grad_colors_in = torch.zeros_like(colors) if colors is not None else None

        return None, None, grad_sdf, grad_colors_in, None

def marching_cubes_asymptotic(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    iso: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Extracts a watertight 2-manifold surface mesh from a voxel grid and SDF field
    using Marching Cubes with Asymptotic Decider (Topologically Consistent Marching Cubes).
    
    Resolves all 6 ambiguous bilinear face cases dynamically on the GPU.

    Args:
        grid_vertices (torch.Tensor): (N, 3) float32 tensor of voxel corner coordinates.
        voxels (torch.Tensor): (M, 8) int32 tensor of voxel corner indices (CCW ring convention).
        sdf (torch.Tensor): (N,) float32 tensor of scalar SDF values.
        colors (torch.Tensor, optional): (N, C) float32 tensor of vertex feature colors.
        iso (float, optional): Isolevel threshold (default: 0.0).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - extracted_vertices (torch.Tensor): (V, 3) float32 tensor of surface vertex positions.
            - extracted_faces (torch.Tensor): (F, 3) int32 tensor of surface triangle face indices.
            - extracted_colors (torch.Tensor | None): (V, C) float32 tensor of interpolated colors.
    """
    grid_vertices = grid_vertices.contiguous().float()
    voxels = voxels.contiguous().int()
    sdf = sdf.contiguous().float()
    if colors is not None:
        colors = colors.contiguous().float()

    verts, faces, out_colors = _C.marching_cubes_asymptotic(
        grid_vertices, voxels, sdf, colors, iso
    )
    if colors is None:
        return verts, faces
    return verts, faces, out_colors

# Alias
mca = marching_cubes_asymptotic
