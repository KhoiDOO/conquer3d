import torch

def sparse_collate_fn(batch):
    """
    Collate function for sparse tensor dataloaders (e.g., MinkowskiEngine, spconv, torchsparse).
    Takes a batch of samples where each sample is (idx_grids, features, ...).
    Appends the batch index to the idx_grids to create (batch_idx, x, y, z).
    
    Args:
        batch (list): A list of tuples (idx_grids, features, label)
        
    Returns:
        tuple: (batched_coords, batched_features, batched_labels)
            batched_coords (torch.Tensor): Shape (Total_N, 4)
            batched_features (torch.Tensor): Shape (Total_N, ...)
            batched_labels (torch.Tensor): Shape (Batch_Size,)
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
