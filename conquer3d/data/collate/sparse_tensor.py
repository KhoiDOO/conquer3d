"""Batch collation utilities for sparse 3D tensor datasets.

This module provides collation routines formatting discrete voxel coordinates
into `(batch_idx, x, y, z)` 4D coordinates compatible with sparse convolution engines
such as SpConv, TorchSparse, and MinkowskiEngine.
"""

from typing import List, Tuple, Any
import torch


def sparse_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates variable-sized discrete grid coordinates and features into a unified sparse batch.
    
    Prepends the batch sample index to each coordinate row to construct the 4D `(batch_idx, x, y, z)` layout.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, Any]]): List of samples where each element
            is a tuple `(idx_grids, features, label)`.
            - `idx_grids`: Int tensor of shape `(N_i, 3)`.
            - `features`: Feature tensor of shape `(N_i, C)`.
            - `label`: Scalar or tensor label.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - batched_coords (torch.Tensor): Int32 coordinate tensor of shape `(Total_N, 4)` in `(batch_idx, x, y, z)`.
            - batched_features (torch.Tensor): Concatenated feature tensor of shape `(Total_N, C)`.
            - batched_labels (torch.Tensor): Batched label tensor of shape `(Batch_Size,)`.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> from conquer3d.data.collate import sparse_collate_fn
        >>> loader = DataLoader(dataset, batch_size=8, collate_fn=sparse_collate_fn)
    """
    batched_coords = []
    batched_features = []
    batched_labels = []

    for batch_idx, item in enumerate(batch):
        idx_grids, features, label = item
        
        # Create a batch index column of shape [N, 1]
        b_col = torch.full((idx_grids.shape[0], 1), batch_idx, dtype=idx_grids.dtype, device=idx_grids.device)
        
        # Concatenate to get [N, 4] formatted as (batch_idx, x, y, z)
        coords_with_batch = torch.cat([b_col, idx_grids], dim=1)
        
        batched_coords.append(coords_with_batch)
        batched_features.append(features)
        batched_labels.append(label)

    batched_coords = torch.cat(batched_coords, dim=0)
    batched_features = torch.cat(batched_features, dim=0)
    
    if isinstance(batched_labels[0], torch.Tensor):
        batched_labels = torch.stack(batched_labels, dim=0)
    else:
        batched_labels = torch.tensor(batched_labels)

    return batched_coords, batched_features, batched_labels
