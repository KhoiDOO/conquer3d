"""Differentiable point cloud fitting to surface mesh using Conquer3D Chamfer distance.

This example demonstrates how to deform a 3D random point cloud sampled from a solid ball
towards target surface points sampled from the Stanford Armadillo benchmark mesh using
Conquer3D's GPU KD-Tree accelerated differentiable Chamfer distance.

The initial source points are generated using `create_random_points_ball`, and optimized
via gradient descent (Adam) minimizing symmetric bidirectional Chamfer distance.

Example:
    $ python chamfer.py --steps 300 --n_points 10000
"""

import os
import argparse
import time
import torch
import trimesh
import numpy as np
from tqdm import tqdm

from conquer3d.data.assets import Armadillo
from conquer3d.data_structure.grid import create_random_points_ball
from conquer3d.ops import chamfer_distance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Differentiable point cloud fitting to mesh surface using Conquer3D Chamfer distance."
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=10000,
        help="Number of points to sample from mesh surface and initial ball (default: 10000)."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of gradient descent optimization steps (default: 300)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="Initial learning rate for Adam optimizer (default: 0.02)."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.8,
        help="Radius of the initial random source points ball (default: 0.8)."
    )
    parser.add_argument(
        "--squared",
        action="store_true",
        default=True,
        help="Use squared Euclidean metric L2^2 (default: True)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run optimization on ('cuda' or 'cpu')."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"=== Conquer3D Differentiable Chamfer Distance Fitting ===")
    print(f"Device:            {device}")
    print(f"Points:            {args.n_points}")
    print(f"Optimization Steps:{args.steps}")
    print(f"Learning Rate:     {args.lr}")
    print(f"Metric:            {'Squared L2^2' if args.squared else 'Euclidean L2'}")

    # Output directory matches the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_ply_path = os.path.join(script_dir, "chamfer_pred.ply")
    gt_ply_path = os.path.join(script_dir, "chamfer_gt.ply")

    # 1. Load the Stanford Armadillo mesh asset
    print("\n[1/4] Loading Armadillo asset...")
    armadillo = Armadillo()
    verts, faces, _ = armadillo.get()

    # Center and normalize Armadillo coordinates to [-1, 1] for uniform optimization scale
    center = (verts.max(dim=0).values + verts.min(dim=0).values) / 2.0
    scale = (verts.max(dim=0).values - verts.min(dim=0).values).max()
    verts_norm = (verts - center) / (scale * 0.5)

    # 2. Sample ground-truth points uniformly from the mesh surface
    print(f"[2/4] Sampling {args.n_points} ground-truth points from Armadillo surface...")
    mesh = trimesh.Trimesh(
        vertices=verts_norm.cpu().numpy(),
        faces=faces.cpu().numpy(),
        process=False
    )
    gt_points_np, _ = trimesh.sample.sample_surface(mesh, args.n_points)
    gt_points = torch.tensor(gt_points_np, dtype=torch.float32, device=device)

    # 3. Create initial source point cloud as a solid random ball
    print(f"[3/4] Initializing source points ball (radius={args.radius})...")
    source_points = create_random_points_ball(
        n_points=args.n_points,
        radius=args.radius,
        device=device
    ).requires_grad_(True)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam([source_points], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-4)

    init_loss = chamfer_distance(source_points, gt_points, squared=args.squared).item()
    print(f"Initial Chamfer Distance Loss: {init_loss:.6f}")

    # 4. Optimization Loop
    print(f"\n[4/4] Fitting source point cloud via Conquer3D Chamfer distance...")
    start_time = time.time()
    pbar = tqdm(range(args.steps), desc="Optimizing", unit="step")

    for step in pbar:
        optimizer.zero_grad()
        loss = chamfer_distance(source_points, gt_points, squared=args.squared)
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix({
            "Chamfer Loss": f"{loss.item():.6f}",
            "lr": f"{scheduler.get_last_lr()[0]:.4f}"
        })

    elapsed = time.time() - start_time
    final_loss = loss.item()
    print(f"\nOptimization completed in {elapsed:.2f}s ({args.steps / elapsed:.1f} steps/s)!")
    print(f"Initial Loss: {init_loss:.6f} -> Final Loss: {final_loss:.6f} (reduced by {(1.0 - final_loss / init_loss) * 100:.2f}%)")

    # 5. Export results as Stanford PLY files
    print(f"\nSaving results to PLY files:")
    print(f"  - Predicted fit:     {pred_ply_path}")
    print(f"  - Ground-truth scan: {gt_ply_path}")

    # Assign distinct colors for visualization (Blue for prediction, Grey for GT)
    pred_colors = np.full((args.n_points, 4), [65, 105, 225, 255], dtype=np.uint8)  # Royal Blue
    gt_colors = np.full((args.n_points, 4), [180, 180, 180, 255], dtype=np.uint8)   # Light Slate Grey

    pred_pc = trimesh.PointCloud(vertices=source_points.detach().cpu().numpy(), colors=pred_colors)
    gt_pc = trimesh.PointCloud(vertices=gt_points.detach().cpu().numpy(), colors=gt_colors)

    pred_pc.export(pred_ply_path)
    gt_pc.export(gt_ply_path)
    print("Export complete! Results are ready for 3D inspection in MeshLab, CloudCompare, or 3D viewer.")


if __name__ == "__main__":
    main()
