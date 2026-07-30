import torch

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

def marching_tetrahedra(vertices, tets, sdfs, colors=None, return_tet_idx=False):
    """
    Extracts a triangulated isosurface from a tetrahedral grid using the Marching Tetrahedra algorithm.
    
    This is a pure-PyTorch, fully vectorized implementation that is mathematically differentiable.
    It identifies tetrahedra that intersect the zero-isosurface, finds the exact intersection points 
    on the edges using linear interpolation based on SDF values, and generates the resulting triangles.
    
    Args:
        vertices (torch.Tensor): A tensor of shape (N, 3) containing the 3D coordinates of the grid vertices.
        tets (torch.Tensor): A tensor of shape (T, 4) containing the vertex indices for each tetrahedron.
        sdfs (torch.Tensor): A tensor of shape (N,) containing the Signed Distance Field values at each vertex.
        colors (torch.Tensor, optional): A tensor of shape (N, C) containing color or feature values at each vertex. 
            If provided, these features are linearly interpolated to the isosurface exactly like the geometry. 
            Defaults to None.
        return_tet_idx (bool, optional): If True, also returns the indices of the tetrahedra that produced 
            each generated face. Defaults to False.

    Returns:
        tuple: 
            - verts (torch.Tensor): The extracted surface vertices of shape (V, 3). 
            - faces (torch.Tensor): The triangular faces of the extracted surface of shape (F, 3).
            - tet_idx (torch.Tensor, optional): The original tetrahedron index corresponding to each face.
              Returned only if `return_tet_idx` is True.
            - verts_colors (torch.Tensor, optional): The interpolated colors at the surface vertices of shape (V, C).
              Returned only if `colors` is provided.
    """
    device = vertices.device
    with torch.no_grad():
        occ_n = sdfs > 0
        occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)
        occ_sum = occ_sum[valid_tets]

        # find all vertices
        all_edges = tets[valid_tets][:, base_tet_edges.to(device)].reshape(-1, 2)
        all_edges = _sort_edges(all_edges)
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
        idx_map = mapping[idx_map]

        interp_v = unique_edges[mask_edges].reshape(-1)

    edges_to_interp = torch.index_select(vertices, 0, interp_v).reshape(-1, 2, 3)
    edges_to_interp_sdfs = torch.index_select(sdfs, 0, interp_v).reshape(-1, 2, 1)

    edges_to_interp0, edges_to_interp1 = edges_to_interp.unbind(dim=1)

    edges_to_interp_sdfs0, edges_to_interp_sdfs1 = edges_to_interp_sdfs.unbind(dim=1)
    edges_to_interp_sdfs1 = -edges_to_interp_sdfs1

    verts = (edges_to_interp0 * edges_to_interp_sdfs1 + edges_to_interp1 * edges_to_interp_sdfs0)
    verts = verts / (edges_to_interp_sdfs0 + edges_to_interp_sdfs1)
    
    if colors is not None:
        edges_to_interp_colors = torch.index_select(colors, 0, interp_v).reshape(-1, 2, colors.shape[-1])
        edges_to_interp_colors0, edges_to_interp_colors1 = edges_to_interp_colors.unbind(dim=1)
        verts_colors = (edges_to_interp_colors0 * edges_to_interp_sdfs1 + edges_to_interp_colors1 * edges_to_interp_sdfs0)
        verts_colors = verts_colors / (edges_to_interp_sdfs0 + edges_to_interp_sdfs1)

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