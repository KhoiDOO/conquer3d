import os
import zipfile
import io
import torch
import trimesh
from PIL import Image
import torchvision.transforms.functional as TF

from .base_mesh import BaseMeshDataset

class Digit3D(BaseMeshDataset):
    """
    Digit3D Mesh Dataset containing 3D MNIST digits.
    """
    def __init__(self, root: str = "~/.conquer3d/", train: bool = True, transform=None, download: bool = False, cached: bool = False, return_img: bool = False):
        root = os.path.expanduser(root)
        super().__init__(root, transform)
        self.train = train
        self.zip_path = os.path.join(root, "digit3d.zip")
        self.split_dir = "src/train" if train else "src/test"
        self.cached = cached
        self.return_img = return_img
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
            cached_data = self._cache[idx]
            if self.return_img:
                vertices_t, faces_t, label, img_t = cached_data
                if self.transform:
                    vertices_t, faces_t = self.transform(vertices_t.clone(), faces_t.clone())
                return vertices_t, faces_t, label, img_t
            else:
                vertices_t, faces_t, label = cached_data[:3]
                if self.transform:
                    vertices_t, faces_t = self.transform(vertices_t.clone(), faces_t.clone())
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
                faces.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
                
        vertices_t = torch.tensor(vertices, dtype=torch.float32)
        faces_t = torch.tensor(faces, dtype=torch.int32)
        
        img_t = None
        if self.return_img:
            img_path = f_path.rsplit(".", 1)[0] + ".png"
            try:
                with self._zip.open(img_path, 'r') as f_img:
                    img_bytes = f_img.read()
                img_pil = Image.open(io.BytesIO(img_bytes))
                img_t = TF.to_tensor(img_pil)
            except KeyError:
                raise FileNotFoundError(f"Image {img_path} not found in {self.zip_path}. Ensure the dataset archive contains PNG images.")
        
        if self.cached:
            if self.return_img:
                self._cache[idx] = (vertices_t, faces_t, label, img_t)
            else:
                self._cache[idx] = (vertices_t, faces_t, label)
        
        if self.transform:
            vertices_t, faces_t = self.transform(vertices_t.clone(), faces_t.clone())
            
        if self.return_img:
            return vertices_t, faces_t, label, img_t
        return vertices_t, faces_t, label

class PointDigit3D(Digit3D):
    """
    Digit3D Dataset that constructs a point cloud by sampling on the mesh.
    """
    def __init__(self, root: str = "~/.conquer3d/", train: bool = True, transform=None, download: bool = False, 
                 cached: bool = False, num_points: int = 512, return_img: bool = False):
        super().__init__(root, train, transform, download, cached=cached, return_img=return_img)
        self.num_points = num_points
        
    def __getitem__(self, idx: int):
        # 1. Obtain data from Digit3D
        if self.return_img:
            vertices, faces, label, img_t = super().__getitem__(idx)
        else:
            vertices, faces, label = super().__getitem__(idx)

        # 2. Construct trimesh object (CPU safe for DataLoader workers)
        mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(), process=False)
        
        # 3. Sample points uniformly over the surface
        points_np, face_indices = trimesh.sample.sample_surface(mesh, self.num_points)
        normals_np = mesh.face_normals[face_indices]
        
        points = torch.tensor(points_np, dtype=torch.float32)
        normals = torch.tensor(normals_np, dtype=torch.float32)
        
        # 4. Combine into features
        features = torch.cat([points, normals], dim=-1)
        
        if self.return_img:
            return points, features, label, img_t
        return points, features, label
