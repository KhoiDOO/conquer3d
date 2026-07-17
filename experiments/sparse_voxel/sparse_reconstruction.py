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
from conquer3d.data.dataset.digit3d import Digit3D
from conquer3d.data.collate.mesh import bmesh_collate_fn
from torch.utils.data import DataLoader

import torchsparse.nn as spnn
from torchsparse import SparseTensor

from torch.amp import autocast, GradScaler

# Import the VAE from sparse_vae.py
from experiments.sparse_voxel.sparse_vae import SimpleSparseVAE, DiagonalGaussianDistribution

from conquer3d.conversion.mesh import mesh2sparse, mesh2sparse_with_dense

# 1. Initialize Datasets & DataLoaders
print("Initializing Datasets...")
train_dataset = Digit3D(root="~/.conquer3d/", train=True, download=False, cached=True)
test_dataset = Digit3D(root="~/.conquer3d/", train=False, download=False, cached=True)

train_loader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=bmesh_collate_fn,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=16, 
    shuffle=False, 
    collate_fn=bmesh_collate_fn,
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
    for batch_idx, (bmesh, batched_labels) in enumerate(progress_bar):
        bmesh = bmesh.cuda(non_blocking=True)
        bmesh.vertices = bmesh.vertices.float()
        batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
        # Send to GPU
        batched_coords = batched_coords.cuda(non_blocking=True)
        batched_sdf = batched_sdf.cuda(non_blocking=True)
        
        # We need dense coords for the full bounding box loss to prevent solid block artifacts
        bmesh = bmesh.cuda(non_blocking=True)
        _, _, dense_coords, dense_sdfs = mesh2sparse_with_dense(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
        dense_coords = dense_coords.to(batched_coords.device)
        dense_sdfs = dense_sdfs.to(batched_sdf.device)
        
        # Construct TorchSparse SparseTensor
        x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # Encode
            x._caches.cmaps.setdefault(x.stride, x.coords)
            h = model.stem(x)
            for layer in model.enc_layers:
                h = layer(h)
            enc_out = model.enc_out(h)
            posterior = DiagonalGaussianDistribution(enc_out.feats, feat_dim=1)
            z_feats = posterior.sample()
            
            gt_z_coords = enc_out.coords
            
            if len(gt_z_coords) > 0:
                # Build full 32x32x32 dense cache to ensure all coordinates are covered
                batch_size = bmesh.batch_size if hasattr(bmesh, 'batch_size') else len(torch.unique(gt_z_coords[:, 0]))
                if batch_size == 0:
                    batch_size = 1
                    
                all_b = torch.arange(batch_size, device=x.coords.device)
                all_x = torch.arange(32, device=x.coords.device)
                all_y = torch.arange(32, device=x.coords.device)
                all_z = torch.arange(32, device=x.coords.device)
                grid_b, grid_x, grid_y, grid_z = torch.meshgrid(all_b, all_x, all_y, all_z, indexing='ij')
                dummy_coords = torch.stack([grid_b, grid_x, grid_y, grid_z], dim=-1).view(-1, 4).int()
                
                if len(dummy_coords) > 0:
                    dummy_feats = torch.zeros((len(dummy_coords), 8), dtype=torch.float32, device=x.feats.device)
                    dummy_x = SparseTensor(coords=dummy_coords.contiguous(), feats=dummy_feats.contiguous())
                    dummy_x._caches.cmaps.setdefault(dummy_x.stride, dummy_x.coords)
                    
                    h_dummy = model.stem(dummy_x)
                    for layer in model.enc_layers:
                        h_dummy = layer(h_dummy)
                        
                    dummy_z_coords = dummy_x._caches.cmaps[(4, 4, 4)]
                    if isinstance(dummy_z_coords, tuple):
                        dummy_z_coords = dummy_z_coords[0]
                        
                    aligned_z_feats = torch.zeros((len(dummy_z_coords), z_feats.shape[-1]), device=x.feats.device, dtype=torch.float32)
                    
                    flat_gt_z = gt_z_coords[:, 0] * 32768 + gt_z_coords[:, 1] * 1024 + gt_z_coords[:, 2] * 32 + gt_z_coords[:, 3]
                    flat_dummy_z = dummy_z_coords[:, 0] * 32768 + dummy_z_coords[:, 1] * 1024 + dummy_z_coords[:, 2] * 32 + dummy_z_coords[:, 3]
                    
                    sort_idx_dummy = torch.argsort(flat_dummy_z)
                    sorted_dummy_z = flat_dummy_z[sort_idx_dummy]
                    
                    idx_in_dummy = torch.searchsorted(sorted_dummy_z, flat_gt_z)
                    valid_mask = (idx_in_dummy < len(sorted_dummy_z)) & (sorted_dummy_z[torch.clamp(idx_in_dummy, max=len(sorted_dummy_z)-1)] == flat_gt_z)
                    
                    target_indices = sort_idx_dummy[idx_in_dummy[valid_mask]]
                    aligned_z_feats[target_indices] = z_feats[valid_mask]
                    
                    z = SparseTensor(coords=dummy_z_coords.contiguous(), feats=aligned_z_feats.contiguous(), stride=(4, 4, 4))
                    z._caches = dummy_x._caches
                    
                    # Decode
                    h_dec = model.dec_in(z)
                    for layer in model.dec_layers:
                        h_dec = layer(h_dec)
                    pred_feats = model.dec_out(h_dec)
                    
                    flat_dense = dense_coords[:, 0] * 32768 + dense_coords[:, 1] * 1024 + dense_coords[:, 2] * 32 + dense_coords[:, 3]
                    flat_pred = pred_feats.coords[:, 0] * 32768 + pred_feats.coords[:, 1] * 1024 + pred_feats.coords[:, 2] * 32 + pred_feats.coords[:, 3]
                    
                    sort_idx = torch.argsort(flat_dense)
                    sorted_dense = flat_dense[sort_idx]
                    idx = torch.searchsorted(sorted_dense, flat_pred)
                    valid_mask_dense = (idx < len(sorted_dense)) & (sorted_dense[torch.clamp(idx, max=len(sorted_dense)-1)] == flat_pred)
                    
                    dummy_indices = sort_idx[idx[valid_mask_dense]]
                    dummy_target_sdf = dense_sdfs[dummy_indices]
                    pred_feats_valid = pred_feats.feats[valid_mask_dense]
                    
                    if len(pred_feats_valid) > 0:
                        recon_loss = mse_criterion(pred_feats_valid, dummy_target_sdf)

                    else:
                        recon_loss = mse_criterion(pred_feats.feats, batched_sdf)
                else:
                    pred_feats, posterior = model(x)
                    recon_loss = mse_criterion(pred_feats, batched_sdf)
            else:
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
        for bmesh, batched_labels in test_loader:
            bmesh = bmesh.cuda(non_blocking=True)
            bmesh.vertices = bmesh.vertices.float()
            batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
            batched_coords = batched_coords.cuda(non_blocking=True)
            batched_sdf = batched_sdf.cuda(non_blocking=True)
            
            x = SparseTensor(coords=batched_coords, feats=batched_sdf)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # Just use normal forward for validation to save time since we just want to track convergence
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
