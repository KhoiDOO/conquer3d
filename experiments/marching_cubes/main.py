import argparse
import sys
import os

if os.path.abspath('../../') not in sys.path:
    sys.path.append(os.path.abspath('../../'))
if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

import torch
from conquer3d.ops.marching_cubes import marching_cubes
from conquer3d.io.obj import write_obj
import conquer3d.data.assets as c3d_assets
import conquer3d.data_structure as ds
from conquer3d.conversion.tmesh import tmesh2sparse

def main():
    parser = argparse.ArgumentParser(description="Standard Marching Cubes Evaluation")
    parser.add_argument('--input', type=str, required=True, help='Name of asset from conquer3d.data.assets')
    parser.add_argument('--res', type=int, required=True, help='Resolution for the voxel grid')
    parser.add_argument('--chunk_size', type=int, default=5000000, help='Chunk size for SDF computation')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Asset
    if not hasattr(c3d_assets, args.input):
        raise ValueError(f"Asset {args.input} not found in conquer3d.data.assets")
    
    asset_class = getattr(c3d_assets, args.input)
    asset = asset_class()
    vertices, faces, _ = asset.get()
    
    vertices = vertices.float()
    faces = faces.to(torch.int32)

    # Normalize vertices to [-0.9, 0.9]
    vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
    scale = 1.8 / torch.max(vmax - vmin).item()
    vertices = vertices - (vmax + vmin) / 2
    vertices = vertices * scale
    
    vertices = vertices.to(device)
    faces = faces.to(device)
    
    print("Building TriangleMesh...")
    tm = ds.TriangleMesh(vertices, faces)
    
    print("--- Mesh Diagnostics ---")
    print(f"Is fully manifold: {tm.is_manifold()}")
    print(f"Is edge manifold: {tm.is_edge_manifold()}")
    print(f"Is vertex manifold: {tm.is_vertex_manifold()}")
    print(f"Has self-intersections: {tm.is_self_intersection()}")
    print("------------------------")
    print("Fixing normals with native CUDA fix_normals()...")
    tm.fix_normals()
    
    r = args.res
    output_dir = str(r)
    os.makedirs(output_dir, exist_ok=True)
    print(f"=========================================")
    print(f"Processing resolution {r}x{r}x{r}...")
    
    print("Creating sparse voxel grid and computing SDF...")
    grid_vertices, active_voxels, idx_grids, sdf = tmesh2sparse(
        tm, r, grid_min=[-1.0, -1.0, -1.0], grid_max=[1.0, 1.0, 1.0], chunk_size=args.chunk_size, device=device
    )
    
    print(f"Active voxels: {active_voxels.shape[0]}")
    
    pt_path = os.path.join(output_dir, f"{args.input.lower()}.pt")
    print(f"Saving active grid info to {pt_path}...")
    torch.save({
        'grid_vertices': grid_vertices,
        'idx_grids': idx_grids,
        'voxels': active_voxels,
        'sdf': sdf
    }, pt_path)
    
    if active_voxels.shape[0] == 0:
        print(f"Warning: No vertices extracted for resolution {r}")
        mc_vertices = torch.empty((0, 3), dtype=torch.float32, device=device)
        mc_faces = torch.empty((0, 3), dtype=torch.int32, device=device)
    else:
        print("Performing standard marching cubes...")
        mc_vertices, mc_faces, _, _ = marching_cubes(grid_vertices, active_voxels, sdf)
    
    if mc_vertices.shape[0] == 0:
        print(f"Warning: No vertices extracted for resolution {r}")
    else:
        obj_path = os.path.join(output_dir, f"{args.input.lower()}.obj")
        print(f"Saving mesh to {obj_path}...")
        write_obj(obj_path, mc_vertices, mc_faces)

    print("Done!")

if __name__ == '__main__':
    main()
