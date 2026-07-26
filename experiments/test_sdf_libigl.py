import os
import sys
import time
import numpy as np
import torch
import igl

# Ensure we import from the local repository build first rather than site-packages
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from conquer3d.creation.triangle_creation import create_sphere
from conquer3d.data_structure import TriangleMesh


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("      SDF COMPARISON: conquer3d vs libigl (Sphere Test)")
    print("=" * 70)

    # 1. Create sphere mesh using conquer3d creation functions
    radius = 1.0
    sectors = 64
    stacks = 32
    print(f"[1] Generating Sphere Mesh: radius={radius}, sectors={sectors}, stacks={stacks}")
    vertices_cpu, triangles_cpu = create_sphere(sectors=sectors, stacks=stacks, radius=radius)
    print(f"    -> Vertices: {vertices_cpu.shape[0]:,}, Triangles: {triangles_cpu.shape[0]:,}")

    # 2. Generate test query points (random points around [-2, 2]^3)
    num_queries = 50_000
    print(f"\n[2] Generating {num_queries:,} Random Query Points in [-2.0, 2.0]^3...")
    torch.manual_seed(42)
    query_pts_cpu = (torch.rand(num_queries, 3, dtype=torch.float32) - 0.5) * 4.0

    # Exact analytical SDF for a centered sphere of radius 1
    sdf_analytic_cpu = torch.norm(query_pts_cpu, dim=1) - radius
    sdf_analytic_np = sdf_analytic_cpu.numpy()

    # 3. Compute SDF using libigl (CPU)
    print("\n[3] Computing SDF with libigl (CPU)...")
    V_np = vertices_cpu.to(torch.float64).numpy()
    F_np = triangles_cpu.to(torch.int64).numpy()
    P_np = query_pts_cpu.to(torch.float64).numpy()

    # (a) libigl Pseudonormals
    start_time = time.perf_counter()
    sdf_igl_pseudo, idx_igl_pseudo, _, _ = igl.signed_distance(
        P_np, V_np, F_np, sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )
    time_igl_pseudo = time.perf_counter() - start_time
    print(f"    -> [libigl] Pseudonormals time      : {time_igl_pseudo:.4f} seconds")

    # (b) libigl Fast Winding Number
    start_time = time.perf_counter()
    sdf_igl_fwn, _, _, _ = igl.signed_distance(
        P_np, V_np, F_np, sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    )
    time_igl_fwn = time.perf_counter() - start_time
    print(f"    -> [libigl] Fast Winding Number time: {time_igl_fwn:.4f} seconds")

    # 4. Compute SDF using conquer3d (CUDA)
    print("\n[4] Computing SDF with conquer3d (CUDA)...")
    vertices_gpu = vertices_cpu.to(device)
    triangles_gpu = triangles_cpu.to(torch.int32).to(device)
    query_pts_gpu = query_pts_cpu.to(device)

    mesh_gpu = TriangleMesh(vertices_gpu, triangles_gpu)

    # Warmup BVH build
    _ = mesh_gpu.query_points(query_pts_gpu[:10], return_sdf=True, return_prj_pts=False, sign_mode=2, distance_mode=0)
    torch.cuda.synchronize()

    # (a) conquer3d Pseudonormals (sign_mode=2)
    start_time = time.perf_counter()
    _, idx_c3d_pseudo, prj_pts_c3d_pseudo, sdf_c3d_pseudo = mesh_gpu.query_points(
        query_pts_gpu, return_sdf=True, return_prj_pts=True, sign_mode=2, distance_mode=0
    )
    torch.cuda.synchronize()
    time_c3d_pseudo = time.perf_counter() - start_time
    print(f"    -> [conquer3d] Pseudonormals time (sign_mode=2): {time_c3d_pseudo:.4f} seconds ({time_igl_pseudo / time_c3d_pseudo:.1f}x faster)")
    sdf_c3d_pseudo_np = sdf_c3d_pseudo.cpu().numpy()

    # (b) conquer3d Fast Winding Number (sign_mode=1)
    start_time = time.perf_counter()
    _, _, _, sdf_c3d_fwn = mesh_gpu.query_points(
        query_pts_gpu, return_sdf=True, return_prj_pts=False, sign_mode=1, distance_mode=0
    )
    torch.cuda.synchronize()
    time_c3d_fwn = time.perf_counter() - start_time
    print(f"    -> [conquer3d] Fast Winding Number time (sign_mode=1): {time_c3d_fwn:.4f} seconds ({time_igl_fwn / time_c3d_fwn:.1f}x faster)")
    sdf_c3d_fwn_np = sdf_c3d_fwn.cpu().numpy()

    # (c) conquer3d Ray Casting (sign_mode=0)
    start_time = time.perf_counter()
    _, _, _, sdf_c3d_ray = mesh_gpu.query_points(
        query_pts_gpu, return_sdf=True, return_prj_pts=False, sign_mode=0, distance_mode=0
    )
    torch.cuda.synchronize()
    time_c3d_ray = time.perf_counter() - start_time
    print(f"    -> [conquer3d] Ray Casting time (sign_mode=0): {time_c3d_ray:.4f} seconds")
    sdf_c3d_ray_np = sdf_c3d_ray.cpu().numpy()

    # 5. Comparative Analysis & Evaluation
    print("\n" + "=" * 70)
    print("                     ACCURACY & ERROR ANALYSIS")
    print("=" * 70)

    def print_comparison(label, pred_sdf, ref_sdf):
        diff = np.abs(pred_sdf - ref_sdf)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rms_diff = np.sqrt(np.mean(diff**2))
        
        # Check sign agreement (ignoring near zero boundary points within 1e-5)
        valid_mask = (np.abs(pred_sdf) > 1e-5) & (np.abs(ref_sdf) > 1e-5)
        sign_match = (np.sign(pred_sdf[valid_mask]) == np.sign(ref_sdf[valid_mask])).mean() * 100.0

        print(f"[{label}]")
        print(f"    - Max Abs Diff : {max_diff:.8f}")
        print(f"    - Mean Abs Diff: {mean_diff:.8f}")
        print(f"    - RMS Diff     : {rms_diff:.8f}")
        print(f"    - Sign Match % : {sign_match:.4f}%\n")

    print_comparison("conquer3d Pseudonormals vs libigl Pseudonormals", sdf_c3d_pseudo_np, sdf_igl_pseudo)
    print_comparison("conquer3d Fast Winding vs libigl Fast Winding", sdf_c3d_fwn_np, sdf_igl_fwn)
    print_comparison("conquer3d Pseudonormals vs Analytic Sphere SDF", sdf_c3d_pseudo_np, sdf_analytic_np)
    print_comparison("libigl Pseudonormals vs Analytic Sphere SDF", sdf_igl_pseudo, sdf_analytic_np)

    print("=" * 70)
    print("                       TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
