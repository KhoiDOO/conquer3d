import torch
from typing import Tuple, Optional
from .. import _C

class DiffDualMarchingCubes(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grid_vertices: torch.Tensor,
        voxels: torch.Tensor,
        sdf: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        iso: float = 0.0,
        quad_split: bool = True,
        project_iters: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Ensure inputs are CUDA and contiguous
        if not grid_vertices.is_cuda or not voxels.is_cuda or not sdf.is_cuda:
            raise RuntimeError("dual_marching_cubes requires CUDA tensors")

        grid_vertices = grid_vertices.contiguous().float()
        voxels = voxels.contiguous().int()
        sdf = sdf.contiguous().float()

        if colors is not None:
            colors = colors.contiguous().float()

        verts, faces, out_colors = _C.dual_marching_cubes(
            grid_vertices, voxels, sdf, colors, iso, quad_split, project_iters
        )

        ctx.save_for_backward(grid_vertices, voxels, sdf, colors)
        ctx.iso = iso
        ctx.project_iters = project_iters
        ctx.has_colors = colors is not None

        return verts, faces, out_colors

    @staticmethod
    def backward(ctx, grad_verts, grad_faces, grad_colors):
        grid_vertices, voxels, sdf, colors = ctx.saved_tensors
        iso = ctx.iso
        project_iters = ctx.project_iters

        # If no gradients needed
        if not sdf.requires_grad and (colors is None or not colors.requires_grad):
            return None, None, None, None, None, None, None

        grad_sdf, grad_colors_in = _C.dual_marching_cubes_backward(
            grad_verts.contiguous(),
            grad_colors.contiguous() if grad_colors is not None else None,
            grid_vertices,
            voxels,
            sdf,
            colors,
            iso,
            project_iters
        )

        return None, None, grad_sdf, grad_colors_in, None, None, None

def dual_marching_cubes(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    iso: float = 0.0,
    quad_split: bool = True,
    project_iters: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Extracts a 2-manifold surface mesh from a voxel grid and scalar field using
    Differentiable Dual Marching Cubes (DMC).

    Unlike standard Dual Contouring which produces at most 1 dual vertex per voxel cell,
    Dual Marching Cubes extracts multiple dual vertices per cell (one per independent MC contour),
    guaranteeing strictly 2-manifold surfaces without topological pinch points or self-intersections.

    Args:
        grid_vertices (torch.Tensor): (N, 3) float32 corner coordinates.
        voxels (torch.Tensor): (M, 8) int32 corner indices in CCW convention.
        sdf (torch.Tensor): (N,) float32 scalar field.
        colors (torch.Tensor, optional): (N, C) float32 vertex feature colors.
        iso (float, optional): Isolevel threshold (default: 0.0).
        quad_split (bool, optional): If True (default), splits each quadrilateral into 2
            triangles with optimal Delaunay angle criterion; if False, returns (Q, 4) quad mesh.
        project_iters (int, optional): Number of Newton-Raphson level-set projection iterations (default: 5).

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
    if colors is not None:
        colors = colors.contiguous().float()

    if sdf.requires_grad or (colors is not None and colors.requires_grad):
        verts, faces, out_colors = DiffDualMarchingCubes.apply(
            grid_vertices, voxels, sdf, colors, iso, quad_split, project_iters
        )
    else:
        verts, faces, out_colors = _C.dual_marching_cubes(
            grid_vertices, voxels, sdf, colors, iso, quad_split, project_iters
        )

    if colors is None:
        return verts, faces
    return verts, faces, out_colors

# Alias
dmc = dual_marching_cubes

