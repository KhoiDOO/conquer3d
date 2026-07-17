import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import os

import conquer3d as c3d
from conquer3d.data.dataset.digit3d import Digit3D
from conquer3d.data.collate.mesh import bmesh_collate_fn
from conquer3d.conversion.mesh import mesh2sparse
from torch.utils.data import DataLoader

from torchsparse import SparseTensor

from torch.amp import autocast, GradScaler

import argparse
import sys
import open3d as o3d
import json

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

# 3. Define the Sparse ResNet Architecture
from experiments.sparse_voxel.sparse_classifier import SparseClassifier

model = SparseClassifier(num_classes=10).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Debug mode: reconstruct first sample and exit")
args, unknown = parser.parse_known_args()

# 4. Training Loop
num_epochs = 20
warmup_epochs = 5

def lr_lambda(epoch):
    import math
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

if args.debug:
    print("Debug mode: reconstructing the first sample from train_dataset...")
    batch = bmesh_collate_fn([train_dataset[0]])
    bmesh, label = batch
    sparse_coords, feats = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
    sparse_coords = sparse_coords.cuda()
    feats = feats.cuda()
    
    unique_vertices, local_voxels, merged_sdfs = c3d.data_structure.sparse2mesh_topology(
        sparse_coords, feats, grid_min=[-1.2, -1.2, -1.2], grid_max=[1.2, 1.2, 1.2], res=[32, 32, 32]
    )
    vert, tri, _, _ = c3d.ops.diff_marching_cubes(unique_vertices, local_voxels, merged_sdfs, iso=0.0)
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert.detach().cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(tri.detach().cpu().numpy())
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug.obj")
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Saved {out_path}")
    sys.exit(0)

print("Starting Training Loop...")

history = []
best_test_acc = 0.0

scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for batch_idx, (bmesh, batched_labels) in enumerate(progress_bar):
        bmesh = bmesh.cuda(non_blocking=True)
        bmesh.vertices = bmesh.vertices.float()
        batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
        # Send to GPU
        batched_coords = batched_coords.cuda(non_blocking=True)
        batched_sdf = batched_sdf.cuda(non_blocking=True) # Voxel-centric features already have 8 channels: [Total_N, 8]
        batched_labels = batched_labels.cuda(non_blocking=True)
        
        # Construct TorchSparse SparseTensor exactly as required
        x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, batched_labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += batched_labels.size(0)
        correct += predicted.eq(batched_labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}", 
            'Acc': f"{100.*correct/total:.2f}%"
        })

    # Evaluate on Test Set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for bmesh, batched_labels in test_loader:
            bmesh = bmesh.cuda(non_blocking=True)
            bmesh.vertices = bmesh.vertices.float()
            batched_coords, batched_sdf = mesh2sparse(bmesh, res=[32, 32, 32], grid_bound=1.2, iso=0.0)
            batched_coords = batched_coords.cuda(non_blocking=True)
            batched_sdf = batched_sdf.cuda(non_blocking=True)
            batched_labels = batched_labels.cuda(non_blocking=True)
            
            x = SparseTensor(coords=batched_coords.contiguous(), feats=batched_sdf.contiguous())
            
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                
            _, predicted = logits.max(1)
            test_total += batched_labels.size(0)
            test_correct += predicted.eq(batched_labels).sum().item()
            
    train_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total
    test_acc = 100. * test_correct / test_total
    
    print(f"==> Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_classification.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[*] Saved new best model to {ckpt_path} with test acc: {best_test_acc:.2f}%")
    
    # Save statistics to JSON
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_acc": test_acc
    })
    
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_classification.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=4)
        
    scheduler.step()

print("Training finished.")
