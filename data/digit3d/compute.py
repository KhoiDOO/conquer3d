import os
import glob
import torch
import numpy as np
import conquer3d as c3d
from tqdm import tqdm

def process_meshes(src_dir, dest_dir, grid_res=32, grid_bound=1.2):
    """
    Reads OBJ files from src_dir, computes active voxel indices & SDF using conquer3d on GPU,
    and saves them as compressed .npz files in dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)
    obj_files = glob.glob(os.path.join(src_dir, "*.obj"))
    
    device = "cuda"
    
    for obj_path in tqdm(obj_files, desc=f"Processing {os.path.basename(src_dir)}"):
        basename = os.path.basename(obj_path)
        name, _ = os.path.splitext(basename)
        npz_path = os.path.join(dest_dir, f"{name}.npz")
        
        # Skip if already processed
        if os.path.exists(npz_path):
            continue
            
        # Extremely fast manual OBJ parsing
        with open(obj_path, 'r') as f:
            lines = f.readlines()
            
        vertices = []
        faces = []
        for line in lines:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()
                # Wavefront OBJs are 1-indexed
                faces.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
                
        if len(vertices) == 0:
            continue
            
        # Move directly to CUDA to utilize fast conquer3d kernels
        vertices_t = torch.tensor(vertices, dtype=torch.float32, device=device)
        faces_t = torch.tensor(faces, dtype=torch.int32, device=device)
        
        # Instantiate BVH tree
        mesh = c3d._C.TriangleMesh(vertices_t, faces_t)
        mesh.build_bvh()
        
        # Create dense grid coordinates
        grid_vertices, voxels, idx_grids = c3d.data_structure.create_voxel_grid(
            grid_min=[-grid_bound] * 3, 
            grid_max=[grid_bound] * 3, 
            res=[grid_res] * 3, 
            device=device
        )
        
        # Query SDF from BVH tree
        _, _, _, sdf = mesh.query_points(grid_vertices, return_sdf=True, return_prj_pts=False)
        
        # Compute active voxels (narrow band)
        active_voxel_indices = c3d.data_structure.compute_active_voxels(voxels, sdf, iso=0.0)
        active_vertices = torch.unique(voxels[active_voxel_indices])
        
        # Extract sparse features and map back to CPU as NumPy arrays
        sparse_idx_grids = idx_grids[active_vertices].cpu().numpy()
        sparse_sdf = sdf[active_vertices].cpu().numpy()
        
        # Save as compressed NumPy archive to save disk space
        np.savez_compressed(npz_path, idx_grids=sparse_idx_grids, sdf=sparse_sdf)

def main():
    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_train = os.path.join(base_dir, "src/train")
    src_test = os.path.join(base_dir, "src/test")
    
    sdf_train = os.path.join(base_dir, "sdf/train")
    sdf_test = os.path.join(base_dir, "sdf/test")
    
    if not os.path.exists(src_train) or not os.path.exists(src_test):
        print("Source directories not found. Make sure data/digit3d/src exists!")
        return

    print("Starting Offline Voxelization Pipeline...")
    process_meshes(src_train, sdf_train)
    process_meshes(src_test, sdf_test)
    print("Precomputation Complete!")

if __name__ == "__main__":
    main()
