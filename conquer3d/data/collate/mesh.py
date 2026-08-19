"""Batch collation utilities for 3D triangle mesh datasets.

This module provides custom `collate_fn` implementations for PyTorch DataLoader,
packing variable-sized mesh vertices and faces into `BTriangleMesh` containers.
"""

from typing import List, Tuple, Any
import torch
from conquer3d.data_structure.bmesh import BTriangleMesh


def bmesh_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Any]]) -> Tuple[BTriangleMesh, torch.Tensor]:
    """Collates a list of `(vertices, faces, label)` tuples into a `BTriangleMesh`.

    Concatenates variable-sized vertices and faces along the 0-th dimension,
    creating batch ID mapping tensors (`vertbids`, `facebids`) to track sample ownership.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, Any]]): List of samples from a mesh dataset,
            where each item is a tuple `(vertices, faces, label)`.

    Returns:
        Tuple[BTriangleMesh, torch.Tensor]:
            - bmesh (BTriangleMesh): Batched mesh container holding concatenated geometry and batch index maps.
            - batched_labels (torch.Tensor): Int64 tensor of labels of shape `(B,)`.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> from conquer3d.data.collate import bmesh_collate_fn
        >>> loader = DataLoader(dataset, batch_size=4, collate_fn=bmesh_collate_fn)
    """
    all_vertices = []
    all_faces = []
    all_vertbids = []
    all_facebids = []
    all_labels = []
    
    for b, (v, f, l) in enumerate(batch):
        all_vertices.append(v)
        
        # We do not offset the face indices because we will process them per-mesh locally
        all_faces.append(f)
        
        all_vertbids.append(torch.full((v.shape[0],), b, dtype=torch.int32))
        all_facebids.append(torch.full((f.shape[0],), b, dtype=torch.int32))
        all_labels.append(l)
        
    batched_vertices = torch.cat(all_vertices, dim=0)
    batched_faces = torch.cat(all_faces, dim=0)
    batched_vertbids = torch.cat(all_vertbids, dim=0)
    batched_facebids = torch.cat(all_facebids, dim=0)
    
    batched_labels = torch.tensor(all_labels, dtype=torch.long)
    
    bmesh = BTriangleMesh(
        vertices=batched_vertices,
        faces=batched_faces,
        vertbids=batched_vertbids,
        facebids=batched_facebids,
        batch_size=len(batch)
    )
    
    return bmesh, batched_labels
