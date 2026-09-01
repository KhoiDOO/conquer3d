"""Differentiable surface reconstruction and rendering example using Conquer3D."""

import os
import argparse
from typing import Tuple, Optional
import torch
from tqdm import tqdm

import conquer3d as c3d
import conquer3d.data_structure as ds
from conquer3d.io.obj import write_obj
from conquer3d.ops import diff_marching_cubes, marching_tetrahedra, tetrahedralize, dpsr
from conquer3d.ops.dpsr import grid_interp
from conquer3d.data.assets import Armadillo

import kaolin as kal
import render
import nvdiffrast.torch as dr


def dpsr_to_mesh(
    points: torch.Tensor,
    normals: torch.Tensor,
    tet_points: Optional[torch.Tensor] = None,
    tets: Optional[torch.Tensor] = None,
    res: int = 64,
    sig: float = 10.0,
    iso: float = 0.0,
    grid_min = [-1.0, -1.0, -1.0],
    grid_max = [1.0, 1.0, 1.0]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts a surface mesh from points and normals via DPSR, Delaunay tetrahedralization, and Marching Tetrahedra.

    Args:
        points (torch.Tensor): Oriented point coordinates `(N, 3)` or `(1, N, 3)`.
        normals (torch.Tensor): Outward normal vectors `(N, 3)` or `(1, N, 3)`.
        tet_points (torch.Tensor, optional): Spatial query vertices used to construct Delaunay triangulation.
            If None, generated within `[grid_min, grid_max]`.
        tets (torch.Tensor, optional): Precomputed Delaunay tetrahedron simplices `(M, 4)`.
            If None, computed via `tetrahedralize(tet_points)`.
        res (int, optional): Grid resolution for DPSR Poisson indicator field. Defaults to 64.
        sig (float, optional): Gaussian smoothing filter degree. Defaults to 10.0.
        iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.
        grid_min: Lower spatial bounds. Defaults to `[-1.0, -1.0, -1.0]`.
        grid_max: Upper spatial bounds. Defaults to `[1.0, 1.0, 1.0]`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Extracted surface vertices `(V, 3)` and triangle faces `(F, 3)`.
    """
    device = points.device
    g_min = torch.as_tensor(grid_min, dtype=torch.float32, device=device)
    g_max = torch.as_tensor(grid_max, dtype=torch.float32, device=device)

    pts_in = points.unsqueeze(0) if points.ndim == 2 else points
    nrms_in = normals.unsqueeze(0) if normals.ndim == 2 else normals

    # 1. Compute DPSR continuous Poisson indicator field on regular grid
    phi = dpsr(pts_in, nrms_in, res=res, sig=sig, grid_min=grid_min, grid_max=grid_max)
    phi = phi.squeeze(0) if phi.ndim == 4 and phi.shape[0] == 1 else phi

    # 2. Delaunay Triangulation (Tetrahedralization)
    if tet_points is None:
        tet_points = torch.rand(2000, 3, device=device) * (g_max - g_min) + g_min
    if tets is None:
        tets = tetrahedralize(tet_points)

    # 3. Trilinear interpolation of DPSR indicator field at tetrahedron vertices
    pts_norm = (tet_points - g_min) / (g_max - g_min)
    phi_b = phi.unsqueeze(0) if phi.ndim == 3 else phi
    pts_b = pts_norm.unsqueeze(0) if pts_norm.ndim == 2 else pts_norm
    sdfs = grid_interp(phi_b, pts_b).squeeze(0)

    # 4. Marching Tetrahedra surface extraction
    verts, faces = marching_tetrahedra(tet_points, tets, sdfs, iso=iso)
    return verts, faces


def lr_schedule(iter_idx):
    """Exponential learning rate decay schedule."""
    return max(0.0, 10 ** (-(iter_idx) * 0.0002))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differentiable Rendering with Conquer3D")
    parser.add_argument("--iter", type=int, default=1000, help="Number of iterations to run optimization.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for camera sampling.")
    parser.add_argument("--train_res", type=int, nargs=2, default=[2048, 2048], help="Resolution of training images.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for Adam optimizer.")
    parser.add_argument("--res", type=int, default=128, help="Resolution of the voxel grid.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu').")
    parser.add_argument("--method", type=str, default="mc", choices=["mc", "dpsr"],
                        help="Differentiable meshing method. Options: 'mc', 'dpsr'.")
    parser.add_argument("--out_dir", type=str, default="./output", help="Output directory to save results.")

    args = parser.parse_args()

    device = args.device
    glctx = dr.RasterizeCudaContext(device=device)

    # 1. Ground Truth Mesh Preparation
    mesh = Armadillo()
    vertices, faces, _ = mesh.get()
    vertices = vertices.float()
    faces = faces.to(torch.int32)
    vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
    scale = 1.8 / torch.max(vmax - vmin).item()
    vertices = vertices - (vmax + vmin) / 2
    vertices = vertices * scale
    vertices = vertices.to(device)
    faces = faces.to(device)

    tm = ds.TriangleMesh(vertices, faces)
    tm.fix_normals()
    gt_mesh = kal.rep.SurfaceMesh(vertices=tm.vertices, faces=tm.triangles)

    # 2. Method Setup & Parameter Initialization
    if args.method == "mc":
        grid_vertices, voxels, _ = c3d.data_structure.create_voxel_grid(
            grid_min=[-1.0, -1.0, -1.0],
            grid_max=[1.0, 1.0, 1.0],
            res=[args.res, args.res, args.res],
            device=device
        )
        sdf = torch.rand_like(grid_vertices[:, 0]) - 0.1
        sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        optimizer = torch.optim.Adam([sdf], lr=args.learning_rate)
    elif args.method == "dpsr":
        # Initialize trainable point cloud and normal vectors on initial sphere
        N_pts = 3000
        phi_angle = torch.rand(N_pts, device=device) * 2.0 * 3.14159265
        theta_angle = torch.acos(torch.rand(N_pts, device=device) * 2.0 - 1.0)
        r = 0.6
        init_pts = torch.stack([
            r * torch.sin(theta_angle) * torch.cos(phi_angle),
            r * torch.sin(theta_angle) * torch.sin(phi_angle),
            r * torch.cos(theta_angle)
        ], dim=-1)
        points = torch.nn.Parameter(init_pts.clone().detach(), requires_grad=True)
        normals = torch.nn.Parameter((init_pts / r).clone().detach(), requires_grad=True)

        # Delaunay tetrahedralization background query points
        tet_res = max(16, args.res // 2)
        tet_grid, _, _ = c3d.data_structure.create_voxel_grid(
            grid_min=[-1.0, -1.0, -1.0],
            grid_max=[1.0, 1.0, 1.0],
            res=[tet_res, tet_res, tet_res],
            device=device
        )
        tets = tetrahedralize(tet_grid)
        optimizer = torch.optim.Adam([points, normals], lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x))

    # 3. Optimization Loop
    pbar = tqdm(range(args.iter), desc=f"Optimizing ({args.method.upper()})")
    for it in pbar:
        optimizer.zero_grad()

        # Sample random camera poses
        cameras = render.get_random_camera_batch(args.batch, iter_res=args.train_res, device=device)

        # Render ground truth mesh from sampled views
        target = render.render_mesh(glctx, gt_mesh, cameras, args.train_res, return_types=["mask", "depth"])

        # Extract differentiable isosurface mesh
        if args.method == "mc":
            mc_vertices, mc_faces = diff_marching_cubes(
                grid_vertices,
                voxels,
                sdf,
                iso=0.0
            )[:2]
        elif args.method == "dpsr":
            mc_vertices, mc_faces = dpsr_to_mesh(
                points,
                normals,
                tet_points=tet_grid,
                tets=tets,
                res=args.res,
                iso=0.0
            )

        if mc_vertices.shape[0] == 0:
            continue

        extracted_mesh = kal.rep.SurfaceMesh(vertices=mc_vertices, faces=mc_faces.int())
        buffers = render.render_mesh(glctx, extracted_mesh, cameras, args.train_res, return_types=["mask", "depth"])

        # Multi-view Silhouette Mask & Depth Losses
        mask_loss = (buffers["mask"] - target["mask"]).abs().mean()
        depth_loss = ((((buffers["depth"] - target["depth"]) * target["mask"]) ** 2).sum(-1) + 1e-8).sqrt().mean() * 10.0

        total_loss = mask_loss + depth_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if (it + 1) % 100 == 0 or it == args.iter - 1:
            pbar.set_postfix({
                "mask_loss": f"{mask_loss.item():.4f}",
                "depth_loss": f"{depth_loss.item():.4f}",
                "total_loss": f"{total_loss.item():.4f}"
            })

    # 4. Export Output Artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    save_name = args.method

    # Save state as .pt file
    save_path = os.path.join(args.out_dir, f"{save_name}.pt")
    if args.method == "mc":
        torch.save(
            {
                "sdf": sdf.detach().cpu(),
                "grid_vertices": grid_vertices.detach().cpu(),
                "voxels": voxels.detach().cpu(),
                "res": args.res,
                "method": "mc",
            },
            save_path
        )
    elif args.method == "dpsr":
        torch.save(
            {
                "points": points.detach().cpu(),
                "normals": normals.detach().cpu(),
                "tet_points": tet_grid.detach().cpu(),
                "tets": tets.detach().cpu(),
                "res": args.res,
                "method": "dpsr",
            },
            save_path
        )
    print(f"Saved optimized model state to: {save_path}")

    # Extract final reconstructed mesh and export as .obj file
    with torch.no_grad():
        if args.method == "mc":
            final_verts, final_faces = diff_marching_cubes(
                grid_vertices,
                voxels,
                sdf,
                iso=0.0
            )[:2]
        elif args.method == "dpsr":
            final_verts, final_faces = dpsr_to_mesh(
                points,
                normals,
                tet_points=tet_grid,
                tets=tets,
                res=args.res,
                iso=0.0
            )

    obj_save_path = os.path.join(args.out_dir, f"{save_name}.obj")
    write_obj(obj_save_path, final_verts.contiguous(), final_faces.contiguous().int())
    print(f"Exported reconstructed 3D mesh ({final_verts.shape[0]:,} vertices, {final_faces.shape[0]:,} faces) to: {obj_save_path}")