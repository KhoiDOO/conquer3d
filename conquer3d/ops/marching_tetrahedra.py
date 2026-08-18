import torch
from typing import Tuple, Optional, Union

# Try to import compiled C++ / CUDA extension
try:
    from .._C import marching_tetrahedra as _c_marching_tetrahedra
    from .._C import marching_tetrahedra_backward as _c_marching_tetrahedra_backward
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False

triangle_table = torch.tensor([
    [-1, -1, -1, -1, -1, -1],
    [1, 0, 2, -1, -1, -1],
    [4, 0, 3, -1, -1, -1],
    [1, 4, 2, 1, 3, 4],
    [3, 1, 5, -1, -1, -1],
    [2, 3, 0, 2, 5, 3],
    [1, 4, 0, 1, 5, 4],
    [4, 2, 5, -1, -1, -1],
    [4, 5, 2, -1, -1, -1],
    [4, 1, 0, 4, 5, 1],
    [3, 2, 0, 3, 5, 2],
    [1, 3, 5, -1, -1, -1],
    [4, 1, 2, 4, 3, 1],
    [3, 0, 4, -1, -1, -1],
    [2, 0, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=torch.long)

num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)
base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)
v_id = torch.pow(2, torch.arange(4, dtype=torch.long))

def _sort_edges(edges):
    """sort last dimension of edges of shape (E, 2)"""
    with torch.no_grad():
        order = (edges[:, 0] > edges[:, 1]).long()
        order = order.unsqueeze(dim=1)

        a = torch.gather(input=edges, index=order, dim=1)
        b = torch.gather(input=edges, index=1 - order, dim=1)

    return torch.stack([a, b], -1)


class DiffMarchingTetrahedra(torch.autograd.Function):
    """
    Differentiable Marching Tetrahedra autograd Function with analytical CUDA backward pass.
    """
    @staticmethod
    def forward(
        ctx,
        vertices: torch.Tensor,
        tets: torch.Tensor,
        sdfs: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        iso: float = 0.0
    ):
        vertices = vertices.contiguous()
        tets = tets.contiguous().to(torch.int32)
        sdfs = sdfs.contiguous()
        if colors is not None:
            colors = colors.contiguous()

        out_vertices, out_triangles, out_normals, out_colors, unique_edges = _c_marching_tetrahedra(
            vertices,
            tets,
            sdfs,
            None,
            colors,
            iso,
            True # return_unique_edges
        )

        ctx.iso = iso
        ctx.has_colors = colors is not None

        if ctx.has_colors:
            ctx.save_for_backward(unique_edges, vertices, sdfs, colors)
        else:
            ctx.save_for_backward(unique_edges, vertices, sdfs)

        return out_vertices, out_triangles, out_colors

    @staticmethod
    def backward(ctx, grad_out_vertices, grad_out_triangles, grad_out_colors):
        if ctx.has_colors:
            unique_edges, vertices, sdfs, colors = ctx.saved_tensors
        else:
            unique_edges, vertices, sdfs = ctx.saved_tensors
            colors = None

        iso = ctx.iso
        grad_sdfs = torch.zeros_like(sdfs)
        grad_colors = None
        if ctx.has_colors:
            grad_colors = torch.zeros_like(colors)

        if unique_edges is not None and unique_edges.shape[0] > 0 and grad_out_vertices is not None:
            grad_out_vertices = grad_out_vertices.contiguous()
            if grad_out_colors is not None:
                grad_out_colors = grad_out_colors.contiguous()

            _c_marching_tetrahedra_backward(
                unique_edges,
                vertices,
                colors,
                sdfs,
                grad_out_vertices,
                grad_out_colors,
                grad_sdfs,
                grad_colors,
                iso
            )

        return None, None, grad_sdfs, grad_colors, None


def _marching_tetrahedra_pure_torch(vertices, tets, sdfs, colors=None, return_tet_idx=False, iso=0.0):
    """
    Pure-PyTorch differentiable reference implementation of Marching Tetrahedra.
    """
    device = vertices.device
    with torch.no_grad():
        occ_n = sdfs < iso
        occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)
        occ_sum = occ_sum[valid_tets]

        # find all edges
        all_edges = tets[valid_tets][:, base_tet_edges.to(device)].reshape(-1, 2)
        all_edges = _sort_edges(all_edges)
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
        idx_map = mapping[idx_map]

        interp_v = unique_edges[mask_edges].reshape(-1)

    if interp_v.numel() == 0:
        empty_verts = torch.empty((0, 3), dtype=vertices.dtype, device=device)
        empty_faces = torch.empty((0, 3), dtype=torch.long, device=device)
        if return_tet_idx:
            if colors is not None:
                return empty_verts, empty_faces, torch.empty((0,), dtype=torch.long, device=device), torch.empty((0, colors.shape[-1]), dtype=colors.dtype, device=device)
            return empty_verts, empty_faces, torch.empty((0,), dtype=torch.long, device=device)
        if colors is not None:
            return empty_verts, empty_faces, torch.empty((0, colors.shape[-1]), dtype=colors.dtype, device=device)
        return empty_verts, empty_faces

    edges_to_interp = torch.index_select(vertices, 0, interp_v).reshape(-1, 2, 3)
    edges_to_interp_sdfs = torch.index_select(sdfs, 0, interp_v).reshape(-1, 2, 1)

    edges_to_interp0, edges_to_interp1 = edges_to_interp.unbind(dim=1)
    edges_to_interp_sdfs0, edges_to_interp_sdfs1 = edges_to_interp_sdfs.unbind(dim=1)

    diff = edges_to_interp_sdfs1 - edges_to_interp_sdfs0
    t = torch.clamp((iso - edges_to_interp_sdfs0) / (diff + 1e-12), 0.0, 1.0)
    verts = edges_to_interp0 + t * (edges_to_interp1 - edges_to_interp0)
    
    if colors is not None:
        edges_to_interp_colors = torch.index_select(colors, 0, interp_v).reshape(-1, 2, colors.shape[-1])
        edges_to_interp_colors0, edges_to_interp_colors1 = edges_to_interp_colors.unbind(dim=1)
        verts_colors = edges_to_interp_colors0 + t * (edges_to_interp_colors1 - edges_to_interp_colors0)

    idx_map = idx_map.reshape(-1, 6)

    tetindex = (occ_fx4[valid_tets] * v_id.to(device).unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table.to(device)[tetindex]
    triangle_table_device = triangle_table.to(device)

    # Generate triangle indices
    faces = torch.cat((
        torch.gather(input=idx_map[num_triangles == 1], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 2], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
    ), dim=0)

    if return_tet_idx:
        tet_idx = torch.arange(tets.shape[0], device=device)[valid_tets]
        tet_idx = torch.cat((tet_idx[num_triangles == 1], tet_idx[num_triangles ==
                            2].unsqueeze(-1).expand(-1, 2).reshape(-1)), dim=0)
        if colors is not None:
            return verts, faces, tet_idx, verts_colors
        return verts, faces, tet_idx
        
    if colors is not None:
        return verts, faces, verts_colors
    return verts, faces


def marching_tetrahedra(
    vertices: torch.Tensor,
    tets: torch.Tensor,
    sdfs: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    return_tet_idx: bool = False,
    iso: float = 0.0,
    use_cuda: Optional[bool] = None
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Extracts a triangulated isosurface from an arbitrary tetrahedral mesh using Marching Tetrahedra.
    
    Automatically leverages the high-performance CUDA backend when running on GPU tensors.
    
    Args:
        vertices (torch.Tensor): (N, 3) tensor of 3D vertex positions.
        tets (torch.Tensor): (T, 4) tensor of tetrahedron vertex indices.
        sdfs (torch.Tensor): (N,) tensor of scalar/SDF values at each vertex.
        colors (torch.Tensor, optional): (N, C) optional tensor of vertex features/colors. Defaults to None.
        return_tet_idx (bool, optional): If True, returns original tetrahedron indices for each triangle face.
        iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.
        use_cuda (bool, optional): Force CUDA or pure PyTorch backend. Defaults to auto-detect.

    Returns:
        tuple: (verts, faces) or (verts, faces, verts_colors)
    """
    should_use_cuda = (use_cuda if use_cuda is not None 
                       else (_CUDA_AVAILABLE and vertices.is_cuda and not return_tet_idx))

    if should_use_cuda:
        out_vertices, out_triangles, out_colors = DiffMarchingTetrahedra.apply(
            vertices, tets, sdfs, colors, iso
        )
        if colors is not None:
            return out_vertices, out_triangles.to(torch.long), out_colors
        return out_vertices, out_triangles.to(torch.long)
    else:
        return _marching_tetrahedra_pure_torch(
            vertices, tets, sdfs, colors=colors, return_tet_idx=return_tet_idx, iso=iso
        )