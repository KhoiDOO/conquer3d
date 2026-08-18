import torch
from typing import Tuple, Optional
from .. import _C

class DiffDualContouring(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grid_vertices: torch.Tensor,
        voxels: torch.Tensor,
        sdf: torch.Tensor,
        grid_normals: Optional[torch.Tensor] = None,
        colors: Optional[torch.Tensor] = None,
        iso: float = 0.0,
        quad_split: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Ensure inputs are CUDA and contiguous
        if not grid_vertices.is_cuda or not voxels.is_cuda or not sdf.is_cuda:
            raise RuntimeError("dual_contouring requires CUDA tensors")

        grid_vertices = grid_vertices.contiguous().float()
        voxels = voxels.contiguous().int()
        sdf = sdf.contiguous().float()

        if grid_normals is not None:
            grid_normals = grid_normals.contiguous().float()
        if colors is not None:
            colors = colors.contiguous().float()

        verts, faces, out_colors = _C.dual_contouring(
            grid_vertices, voxels, sdf, grid_normals, colors, iso, quad_split
        )

        ctx.save_for_backward(grid_vertices, voxels, sdf, grid_normals, colors)
        ctx.iso = iso
        ctx.has_colors = colors is not None

        return verts, faces, out_colors

    @staticmethod
    def backward(ctx, grad_verts, grad_faces, grad_colors):
        grid_vertices, voxels, sdf, grid_normals, colors = ctx.saved_tensors
        iso = ctx.iso

        # If no gradients needed
        if not sdf.requires_grad and (colors is None or not colors.requires_grad):
            return None, None, None, None, None, None, None

        grad_sdf, grad_colors_in = _C.dual_contouring_backward(
            grad_verts.contiguous(),
            grad_colors.contiguous() if grad_colors is not None else None,
            grid_vertices,
            voxels,
            sdf,
            grid_normals,
            colors,
            iso
        )

        return None, None, grad_sdf, None, grad_colors_in, None, None

def dual_contouring(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    grid_normals: Optional[torch.Tensor] = None,
    colors: Optional[torch.Tensor] = None,
    iso: float = 0.0,
    quad_split: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Extracts an explicit surface mesh from a voxel grid and scalar field using
    Differentiable Dual Contouring with GPU Quadratic Error Function (QEF) solver.

    Preserves sharp creases, features, and mechanical corners by placing dual vertices
    inside voxel cells via regularized Jacobi SVD optimization.

    Args:
        grid_vertices (torch.Tensor): (N, 3) float32 corner coordinates.
        voxels (torch.Tensor): (M, 8) int32 corner indices in CCW convention.
        sdf (torch.Tensor): (N,) float32 scalar field.
        grid_normals (torch.Tensor, optional): (N, 3) float32 explicit vertex normals.
            If None, evaluated on-the-fly via analytical trilinear cell gradients.
        colors (torch.Tensor, optional): (N, C) float32 vertex feature colors.
        iso (float, optional): Isolevel threshold (default: 0.0).
        quad_split (bool, optional): If True (default), splits each quadrilateral into 2
            triangles with optimal Delaunay angle criterion; if False, returns (Q, 4) quad mesh.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - extracted_vertices (torch.Tensor): (V, 3) float32 surface vertex positions.
            - extracted_faces (torch.Tensor): (F, 3) int32 triangles if quad_split=True,
                                             or (Q, 4) int32 quads if quad_split=False.
            - extracted_colors (torch.Tensor | None): (V, C) float32 interpolated colors.
    """
    grid_vertices = grid_vertices.contiguous().float()
    voxels = voxels.contiguous().int()
    sdf = sdf.contiguous().float()
    if grid_normals is not None:
        grid_normals = grid_normals.contiguous().float()
    if colors is not None:
        colors = colors.contiguous().float()

    if sdf.requires_grad or (colors is not None and colors.requires_grad):
        verts, faces, out_colors = DiffDualContouring.apply(
            grid_vertices, voxels, sdf, grid_normals, colors, iso, quad_split
        )
    else:
        verts, faces, out_colors = _C.dual_contouring(
            grid_vertices, voxels, sdf, grid_normals, colors, iso, quad_split
        )

    if colors is None:
        return verts, faces
    return verts, faces, out_colors

# Alias
dc = dual_contouring
