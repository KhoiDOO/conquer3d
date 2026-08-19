"""GPU-accelerated Differentiable Dual Contouring (DC) with Jacobi QEF solvers.

Dual Contouring (Ju et al., 2002) extracts explicit surface meshes from volumetric scalar
fields while preserving sharp features, creases, and corners. For each active voxel cell
intersected by the isosurface, a single dual vertex is placed at the feature point minimizing
the Quadratic Error Function (QEF):

$$E(v) = \\sum_{i} \\left( n_i \\cdot (v - p_i) \\right)^2$$

where $p_i$ are Hermite edge intersection points and $n_i$ are the corresponding surface normals.
The QEF minimum is solved in parallel on the GPU using cyclic Jacobi Singular Value
Decomposition (SVD) on register arrays.

Example:
    >>> import torch
    >>> from conquer3d.ops import dual_contouring
    >>> # Extract sharp mesh from voxel grid and signed distance field
    >>> verts, faces = dual_contouring(grid_vertices, voxels, sdf, iso=0.0, quad_split=True)
"""

from typing import Tuple, Optional, Union
import torch
from .. import _C


class DiffDualContouring(torch.autograd.Function):
    """PyTorch autograd Function for Differentiable Dual Contouring on CUDA.

    Implements forward surface extraction with Jacobi QEF solvers and backward
    adjoint gradient propagation with respect to input scalar SDF values and vertex colors.
    """

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
        """Forward pass extracting dual vertices and quadrilateral/triangle connectivity.

        Args:
            ctx: PyTorch autograd context object.
            grid_vertices (torch.Tensor): Float32 tensor of shape `(N, 3)` containing 3D grid
                vertex coordinates on CUDA.
            voxels (torch.Tensor): Int32 tensor of shape `(M, 8)` containing 8 corner vertex
                indices per voxel cell in counter-clockwise convention on CUDA.
            sdf (torch.Tensor): Float32 tensor of shape `(N,)` containing scalar SDF values on CUDA.
            grid_normals (torch.Tensor, optional): Float32 tensor of shape `(N, 3)` containing
                explicit vertex normal vectors. If None, evaluated on-the-fly via analytical
                trilinear cell gradients. Defaults to None.
            colors (torch.Tensor, optional): Float32 tensor of shape `(N, C)` containing vertex
                colors or feature embeddings on CUDA. Defaults to None.
            iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.
            quad_split (bool, optional): If True, splits dual quadrilaterals into 2 triangles
                according to the optimal Delaunay min-angle criterion; if False, returns quads `(Q, 4)`.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - extracted_vertices: Float32 tensor of shape `(V, 3)` with surface vertex positions.
                - extracted_faces: Int32 tensor of shape `(F, 3)` (triangles) or `(Q, 4)` (quads).
                - extracted_colors: Float32 tensor of shape `(V, C)` or None if colors was None.

        Raises:
            RuntimeError: If `grid_vertices`, `voxels`, or `sdf` are not on CUDA.
        """
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
        """Backward pass evaluating analytical adjoint gradients w.r.t. input SDF and colors.

        Args:
            ctx: PyTorch autograd context object containing saved forward tensors.
            grad_verts (torch.Tensor): Float32 tensor of shape `(V, 3)` containing upstream vertex gradients.
            grad_faces (torch.Tensor): Upstream face gradients (unused, non-differentiable discrete topology).
            grad_colors (torch.Tensor, optional): Float32 tensor of shape `(V, C)` containing upstream color gradients.

        Returns:
            Tuple[None, None, torch.Tensor, None, Optional[torch.Tensor], None, None]:
                Gradients corresponding to (grid_vertices, voxels, sdf, grid_normals, colors, iso, quad_split).
        """
        grid_vertices, voxels, sdf, grid_normals, colors = ctx.saved_tensors
        iso = ctx.iso

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
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Extracts sharp surface meshes from volumetric scalar fields using Differentiable Dual Contouring.

    Preserves sharp creases, mechanical edges, and corners by positioning dual vertices
    at the optimal Quadratic Error Function (QEF) minimizer using register-level Jacobi SVD.

    Args:
        grid_vertices (torch.Tensor): Float32 tensor of shape `(N, 3)` containing grid vertex
            coordinates on CUDA. Must be contiguous.
        voxels (torch.Tensor): Int32 tensor of shape `(M, 8)` containing 8 corner vertex indices
            per voxel cell on CUDA.
        sdf (torch.Tensor): Float32 tensor of shape `(N,)` containing scalar SDF values on CUDA.
        grid_normals (torch.Tensor, optional): Float32 tensor of shape `(N, 3)` containing explicit
            surface normals at grid vertices. If None, evaluated on-the-fly via analytical
            trilinear cell gradients. Defaults to None.
        colors (torch.Tensor, optional): Float32 tensor of shape `(N, C)` containing vertex feature
            colors on CUDA. Defaults to None.
        iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.
        quad_split (bool, optional): If True, splits dual quadrilaterals into 2 triangles using
            the optimal Delaunay angle criterion; if False, returns quads of shape `(Q, 4)`.
            Defaults to True.

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            - If `colors` is None: Returns `(vertices, faces)`.
            - If `colors` is provided: Returns `(vertices, faces, colors)`.
            - `vertices`: Float32 tensor of shape `(V, 3)` on CUDA.
            - `faces`: Int32 tensor of shape `(F, 3)` for triangles or `(Q, 4)` for quads on CUDA.
            - `colors`: Float32 tensor of shape `(V, C)` on CUDA.

    Raises:
        RuntimeError: If inputs are not on CUDA or not contiguous.

    Example:
        >>> import torch
        >>> from conquer3d.ops import dual_contouring
        >>> # Extract sharp 3D surface mesh
        >>> verts, faces = dual_contouring(grid_vertices, voxels, sdf, iso=0.0)
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


# Public alias
dc = dual_contouring
