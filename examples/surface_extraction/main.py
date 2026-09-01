import argparse
import json
import os
import torch
import conquer3d.data_structure as ds

from conquer3d.data.dataset import ToyMeshDataset, MeshDataset

from conquer3d.ops import (
    marching_cubes,
    marching_cubes_asymptotic,
    marching_tetrahedra,
    marching_tetrahedra_grid,
    dual_contouring,
    dual_marching_cubes,
    tetrahedralize,
)
import gdel3d_cuda
from conquer3d.io.obj import write_obj
from conquer3d.conversion.tmesh import tmesh2sparse

from tqdm import tqdm


def get_tetrahedra(grid_vertices: torch.Tensor) -> torch.Tensor:
    grid_vertices = grid_vertices.contiguous().float()
    return gdel3d_cuda.tetrahedralize_cuda(grid_vertices)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Standard Marching Cubes Evaluation")
    parser.add_argument('--ds', type=str, required=True, help='Name of datasets')
    parser.add_argument('--ds_root', type=str, default=None, help='Root directory for the dataset')
    parser.add_argument('--method', type=str, required=True, help='Marching method to use', choices=['mc', 'mca', 'mt', 'mtc', 'mtg', 'dc', 'dmc'])
    parser.add_argument('--res', type=int, required=True, help='Resolution for the voxel grid')
    parser.add_argument('--chunk_size', type=int, default=5000000, help='Chunk size for SDF computation')
    parser.add_argument('--recompute', action='store_true', help='Force recomputation even if result files exist')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.ds == 'toy':
        dataset = ToyMeshDataset(return_hash_id=True)
    else:
        dataset = MeshDataset(root=args.ds_root, return_hash_id=True)
    num_samples = len(dataset)
    
    save_main_dir = 'outputs'
    save_dir = os.path.join(save_main_dir, args.ds, args.method, str(args.res))
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in tqdm(range(num_samples), desc="Processing samples"):
        sample_name = dataset.names[idx] if hasattr(dataset, 'names') and idx < len(dataset.names) else f"sample_{idx}"
        try:
            vertices, faces, hash_id  = dataset[idx]
        except Exception as e:
            if args.verbose:
                print(f"Skipping sample {idx} ({sample_name}) due to load error: {e}")
            continue

        properties_save_path = os.path.join(save_dir, f"{hash_id}_properties.json")
        mesh_save_path = os.path.join(save_dir, f"{hash_id}.obj")
        if not args.recompute and os.path.exists(properties_save_path) and os.path.exists(mesh_save_path):
            if args.verbose:
                print(f"Skipping {hash_id}: result files already exist.")
            continue
        
        vertices = vertices.float()
        faces = faces.to(torch.int32)

        # Normalize vertices to [-0.9, 0.9]
        vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
        scale = 1.8 / torch.max(vmax - vmin).item()
        vertices = vertices - (vmax + vmin) / 2
        vertices = vertices * scale
        
        vertices = vertices.to(device)
        faces = faces.to(device)
        
        if args.verbose:
            print("Building TriangleMesh...")
        tm = ds.TriangleMesh(vertices, faces)
        
        if args.verbose:
            print("Fixing normals with native CUDA fix_normals()...")
        tm.fix_normals()
        
        if args.verbose:
            print("Creating sparse voxel grid and computing SDF...")
        
        grid_vertices, active_voxels, sdf = tmesh2sparse(
            tm, 
            args.res, 
            grid_min=[-1.0, -1.0, -1.0], 
            grid_max=[1.0, 1.0, 1.0], 
            chunk_size=args.chunk_size, 
            device=device,
            show_progress=args.verbose,
            sign_mode=5
        )
        
        with torch.inference_mode():
            if args.method == 'mc':
                vertices, faces, _, _ = marching_cubes(grid_vertices, active_voxels, sdf)
            elif args.method == 'mt':
                tets = get_tetrahedra(grid_vertices)
                vertices, faces = marching_tetrahedra(grid_vertices, tets, sdf)
            elif args.method == 'mtc':
                tets = get_tetrahedra(grid_vertices)
                vertices, faces = marching_tetrahedra(grid_vertices, tets, sdf, use_cuda=True)
            elif args.method == 'mtg':
                vertices, faces, _, _ = marching_tetrahedra_grid(grid_vertices, active_voxels, sdf)
            elif args.method == 'dc':
                vertices, faces = dual_contouring(grid_vertices, active_voxels, sdf)
            elif args.method == 'dmc':
                vertices, faces = dual_marching_cubes(grid_vertices, active_voxels, sdf)
            elif args.method == 'mca':
                vertices, faces = marching_cubes_asymptotic(grid_vertices, active_voxels, sdf)
            else:
                raise NotImplementedError(f"Method {args.method} is not implemented in this script.")
        
        properties = {
            "#grid_vertices": grid_vertices.shape[0],
            "#active_voxels": active_voxels.shape[0],
            "#vertices": vertices.shape[0],
            "#faces": faces.shape[0],
        }
        
        properties_save_path = os.path.join(save_dir, f"{hash_id}_properties.json")
        
        with open(properties_save_path, 'w') as f:
            json.dump(properties, f, indent=4)
            
        mesh_save_path = os.path.join(save_dir, f"{hash_id}.obj")
        write_obj(mesh_save_path, vertices.cpu().numpy(), faces.cpu().numpy())