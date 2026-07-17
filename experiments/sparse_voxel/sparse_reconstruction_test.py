import torch
import numpy as np
from tqdm.auto import tqdm
import os
import json

import conquer3d as c3d
from conquer3d.data.dataset.digit3d import Digit3D
from conquer3d.data.collate.mesh import bmesh_collate_fn
from conquer3d.conversion.mesh import mesh2sparse
from conquer3d.ops.distance import chamfer_distance
from torch.utils.data import DataLoader
from torchsparse import SparseTensor

from experiments.sparse_voxel.sparse_vae import SimpleSparseVAE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Dataset (batch_size=1 for accurate per-sample extraction)
    print("Initializing Test Dataset...")
    test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=False, cached=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=bmesh_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 2. Load VAE
    print("Loading VAE Model...")
    model = SimpleSparseVAE(in_channels=8, hidden_channels=32, latent_channels=16, out_channels=8, num_layers=3).to(device)
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_reconstruction.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Warning: {ckpt_path} not found. Testing with untrained weights.")
    model.eval()

    per_sample_cd = []
    
    print("Computing Chamfer Distance...")
    # 3. Test Loop
    with torch.no_grad():
        for i, (bmesh, batched_labels) in enumerate(tqdm(test_loader, desc="Testing VAE")):
            bmesh = bmesh.cuda(non_blocking=True)
            
            orig_points = bmesh.vertices.float()
            
            batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
            batched_coords = batched_coords.to(device, non_blocking=True)
            batched_sdf = batched_sdf.to(device, non_blocking=True)
            
            x = SparseTensor(coords=batched_coords, feats=batched_sdf)
            
            # Forward pass
            pred_feats, _ = model(x)
            
            # Reconstruct predicted mesh
            pred_unique_vertices, pred_local_voxels, pred_merged_sdfs = c3d.conversion.grid.sparse2voxel(
                batched_coords, pred_feats, grid_min=[-1.2, -1.2, -1.2], grid_max=[1.2, 1.2, 1.2], res=[32, 32, 32]
            )
            pred_vert, pred_tri, _, _ = c3d.ops.diff_marching_cubes(pred_unique_vertices, pred_local_voxels, pred_merged_sdfs, iso=0.0)
            
            if len(pred_vert) == 0:
                cd_val = float('inf')
            else:
                pred_vert = pred_vert.float()
                # Compute Chamfer Distance
                cd_val = chamfer_distance(orig_points, pred_vert, squared=True).item()
            
            per_sample_cd.append({
                "sample_idx": i,
                "label": int(batched_labels[0].item()),
                "chamfer_distance": cd_val
            })

if __name__ == "__main__":
    main()
