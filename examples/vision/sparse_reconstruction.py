import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import os
import json
import argparse
import sys
import open3d as o3d

import conquer3d as c3d
from conquer3d.data.dataset.digit3d import SparseDigit3D
from conquer3d.data.collate.sparse_tensor import sparse_collate_fn
from torch.utils.data import DataLoader

import torchsparse.nn as spnn
from torchsparse import SparseTensor

from torch.amp import autocast, GradScaler

# Import the VAE from sparse_vae.py
from sparse_vae import SimpleSparseVAE

# 1. Initialize Datasets & DataLoaders
print("Initializing Datasets...")
train_dataset = SparseDigit3D(root="~/.conquer3d/", train=True, download=False, cached=True)
test_dataset = SparseDigit3D(root="~/.conquer3d/", train=False, download=False, cached=True)

train_loader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=sparse_collate_fn,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=16, 
    shuffle=False, 
    collate_fn=sparse_collate_fn,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True
)

# 2. Define the Model, Loss, and Optimizer
model = SimpleSparseVAE(in_channels=8, hidden_channels=32, latent_channels=16, out_channels=8, num_layers=3).cuda()
mse_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Debug mode: reconstruct first sample and exit")
args, unknown = parser.parse_known_args()

num_epochs = 10
kl_weight = 1e-4

if args.debug:
    print("Debug mode: running reconstruction on the first sample...")
    # Just grab the first batch
    for batch in train_loader:
        batched_coords, batched_sdf, _ = batch
        break
    
    # We just visualize the first sample in the batch (batch_idx == 0)
    mask = (batched_coords[:, 0] == 0)
    sparse_idx_grids = batched_coords[mask, 1:]
    sparse_sdfs = batched_sdf[mask]
    
    b_col = torch.zeros((sparse_idx_grids.size(0), 1), dtype=sparse_idx_grids.dtype)
    sparse_coords = torch.cat([b_col, sparse_idx_grids], dim=1).cuda()
    feats = sparse_sdfs.cuda()
    
    unique_vertices, local_voxels, merged_sdfs = c3d.data_structure.sparse2mesh_topology(
        sparse_coords, feats, grid_min=[-1.2, -1.2, -1.2], grid_max=[1.2, 1.2, 1.2], res=[32, 32, 32]
    )
    vert, tri, _, _ = c3d.ops.diff_marching_cubes(unique_vertices, local_voxels, merged_sdfs, iso=0.0)
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert.detach().cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(tri.detach().cpu().numpy())
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_vae.obj")
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Saved {out_path}")
    sys.exit(0)

print("Starting Training Loop...")

history = []
best_test_loss = float('inf')

scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for batch_idx, (batched_coords, batched_sdf, batched_labels) in enumerate(progress_bar):
        # Send to GPU
        batched_coords = batched_coords.cuda(non_blocking=True)
        batched_sdf = batched_sdf.cuda(non_blocking=True)
        
        # Construct TorchSparse SparseTensor
        x = SparseTensor(coords=batched_coords, feats=batched_sdf)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            pred_feats, posterior = model(x)
            
            recon_loss = mse_criterion(pred_feats, batched_sdf)
            kl_loss = posterior.kl(dims=-1).mean()
            loss = recon_loss + kl_weight * kl_loss
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_train_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}", 
            'Recon': f"{recon_loss.item():.4f}",
            'KL': f"{kl_loss.item():.4f}"
        })

    # Evaluate on Test Set
    model.eval()
    test_recon_loss = 0.0
    test_kl_loss = 0.0
    test_loss = 0.0
    
    with torch.no_grad():
        for batched_coords, batched_sdf, batched_labels in test_loader:
            batched_coords = batched_coords.cuda(non_blocking=True)
            batched_sdf = batched_sdf.cuda(non_blocking=True)
            
            x = SparseTensor(coords=batched_coords, feats=batched_sdf)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                pred_feats, posterior = model(x)
                
                r_loss = mse_criterion(pred_feats, batched_sdf)
                k_loss = posterior.kl(dims=-1).mean()
                t_loss = r_loss + kl_weight * k_loss
                
            test_recon_loss += r_loss.item()
            test_kl_loss += k_loss.item()
            test_loss += t_loss.item()
            
    train_loss_avg = total_train_loss / len(train_loader)
    test_loss_avg = test_loss / len(test_loader)
    
    print(f"==> Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss_avg:.4f}, Test Loss: {test_loss_avg:.4f}")
    
    if test_loss_avg < best_test_loss:
        best_test_loss = test_loss_avg
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_reconstruction.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[*] Saved new best model to {ckpt_path} with test loss: {best_test_loss:.4f}")
    
    # Save statistics to JSON
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss_avg,
        "test_loss": test_loss_avg
    })
    
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_reconstruction.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=4)

print("Training finished.")
