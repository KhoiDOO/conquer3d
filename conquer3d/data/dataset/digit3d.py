import os
import zipfile
import torch
import numpy as np
import conquer3d as c3d
import trimesh
import meshlib.mrmeshpy as mr
import meshlib.mrmeshnumpy as mrnp

from .base_mesh import BaseMeshDataset

class Digit3D(BaseMeshDataset):
    """
    Digit3D Mesh Dataset containing 3D MNIST digits.
    """
    def __init__(self, root: str = "~/.conquer3d/", train: bool = True, transform=None, download: bool = False, cached: bool = False):
        root = os.path.expanduser(root)
        super().__init__(root, transform)
        self.train = train
        self.zip_path = os.path.join(root, "digit3d.zip")
        self.split_dir = "src/train" if train else "src/test"
        self.cached = cached
        self._cache = {}
        
        if download:
            self.download()
            
        if not os.path.exists(self.zip_path):
            raise RuntimeError(f"Dataset not found at {self.zip_path}. You can use download=True to download it.")
            
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            self.all_files = [f for f in z.namelist() if f.startswith(self.split_dir) and f.endswith(".obj")]

    def download(self):
        if os.path.exists(self.zip_path):
            return
        os.makedirs(self.root, exist_ok=True)
        url = "https://drive.google.com/uc?id=1Vry0-sflcSmpwZnjn8yBbF2vBfuW1T_W"
        try:
            import gdown
        except ImportError:
            raise ImportError("gdown is required to download the dataset. Please install it using 'pip install gdown'.")
        print(f"Downloading Digit3D dataset to {self.zip_path}...")
        gdown.download(url, self.zip_path, quiet=False)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int):
        if self.cached and idx in self._cache:
            vertices_t, faces_t, label = self._cache[idx]
            if self.transform:
                vertices_t = self.transform(vertices_t.clone())
            return vertices_t, faces_t, label
            
        f_path = self.all_files[idx]
        basename = os.path.basename(f_path)
        label = int(basename.split("_")[0])
        
        # Read directly from zip stream to avoid file descriptor and extraction I/O overhead
        if getattr(self, '_zip', None) is None:
            self._zip = zipfile.ZipFile(self.zip_path, 'r')
            
        with self._zip.open(f_path, 'r') as f:
            content = f.read().decode('utf-8')
                
        vertices = []
        faces = []
        for line in content.splitlines():
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()
                # Wavefront OBJ faces are 1-indexed, so we subtract 1 for 0-indexed tensors
                faces.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
                
        vertices_t = torch.tensor(vertices, dtype=torch.float32)
        faces_t = torch.tensor(faces, dtype=torch.int32)
        
        if self.cached:
            self._cache[idx] = (vertices_t, faces_t, label)
        
        if self.transform:
            vertices_t = self.transform(vertices_t.clone())
            
        return vertices_t, faces_t, label


class SparseDigit3D(Digit3D):
    """
    Digit3D Dataset that constructs a sparse SDF voxel grid from the mesh on-the-fly using CPU (Open3D).
    This allows arbitrary geometric augmentations on the mesh before voxelization without CUDA IPC issues.
    """
    def __init__(self, root: str = "~/.conquer3d/", train: bool = True, transform=None, download: bool = False, 
                 grid_res: int = 32, grid_bound: float = 1.2, cached: bool = False):
        super().__init__(root, train, transform, download, cached=cached)
        self.grid_res = grid_res
        self.grid_bound = grid_bound
        
    def __getitem__(self, idx: int):
        # 1. Obtain vertices, faces, and label from Digit3D
        vertices, faces, label = super().__getitem__(idx)
        
        # 2. Construct voxel grid in CPU
        grid_vertices, voxels, idx_grids = c3d.data_structure.create_voxel_grid(
            grid_min=[-self.grid_bound] * 3, 
            grid_max=[self.grid_bound] * 3, 
            res=[self.grid_res] * 3, 
            device="cpu"
        )
        
        # Construct the mesh using meshlib's numpy interface
        mesh_mr = mrnp.meshFromFacesVerts(faces.numpy(), vertices.numpy())
        
        # Construct point cloud from grid vertices for vectorized distance computation
        pc = mrnp.pointCloudFromPoints(grid_vertices.numpy())
        
        # Compute the signed distance for all grid vertices at once
        dist_scalars = mr.findSignedDistances(mesh_mr, pc.points)
        
        # Convert to tensor
        sdf = torch.tensor(list(dist_scalars), dtype=torch.float32, device="cpu")
        
        # 4. Compute active voxels
        active_voxel_indices = c3d.data_structure.compute_active_voxels(voxels, sdf, iso=0.0)
        
        # 5. Extract purely Voxel-Centric representations using voxel2sparse
        sparse_coords, sparse_sdfs = c3d.data_structure.voxel2sparse(
            active_voxel_indices, voxels, idx_grids, sdf=sdf, batch_idx=0
        )
        
        # We only need the x, y, z for the dataset (collate_fn handles batch_idx)
        sparse_idx_grids = sparse_coords[:, 1:]
        
        return sparse_idx_grids, sparse_sdfs, label
