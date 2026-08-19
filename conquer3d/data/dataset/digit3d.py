"""Digit3D: 3D MNIST digit mesh dataset and point cloud generators.

This module provides dataset loaders for 3D MNIST digit meshes, supporting
on-the-fly zip extraction, surface point cloud sampling, and multi-modal image pairing.
"""

from typing import Tuple, Optional, Callable, Union, Any
import os
import zipfile
import io
import torch
import trimesh
from PIL import Image
import torchvision.transforms.functional as TF

from .base_mesh import BaseMeshDataset


class Digit3D(BaseMeshDataset):
    """Digit3D Mesh Dataset containing 3D MNIST digits.

    Attributes:
        root (str): Root cache directory path.
        train (bool): If True, loads training split; if False, test split.
        zip_path (str): File path to the zip archive.
        cached (bool): If True, caches parsed OBJ files in memory.
        return_img (bool): If True, also loads and returns the paired 2D digit image.
    """

    def __init__(
        self,
        root: str = "~/.conquer3d/",
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        cached: bool = False,
        return_img: bool = False
    ) -> None:
        """Initializes the Digit3D dataset instance.

        Args:
            root (str, optional): Root folder for caching dataset archive. Defaults to `"~/.conquer3d/"`.
            train (bool, optional): Whether to load train or test split. Defaults to True.
            transform (Callable, optional): Geometric transform applied to (vertices, faces). Defaults to None.
            download (bool, optional): Automatically download dataset if missing. Defaults to False.
            cached (bool, optional): In-memory sample cache. Defaults to False.
            return_img (bool, optional): Return paired 2D image tensor. Defaults to False.

        Raises:
            RuntimeError: If dataset archive is not found and `download=False`.
        """
        root = os.path.expanduser(root)
        super().__init__(root, transform)
        self.train = train
        self.zip_path = os.path.join(root, "digit3d.zip")
        self.split_dir = "src/train" if train else "src/test"
        self.cached = cached
        self.return_img = return_img
        self._cache = {}
        self._zip = None
        
        if download:
            self.download()
            
        if not os.path.exists(self.zip_path):
            raise RuntimeError(f"Dataset not found at {self.zip_path}. You can use download=True to download it.")
            
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            self.all_files = [f for f in z.namelist() if f.startswith(self.split_dir) and f.endswith(".obj")]

    def download(self) -> None:
        """Downloads the Digit3D dataset archive via Google Drive if not cached locally."""
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
        """int: Total number of digit mesh samples."""
        return len(self.all_files)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]]:
        """Loads and returns 3D mesh vertices, faces, integer class label, and optional 2D image."""
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, 'r')

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
        label = int(os.path.basename(os.path.dirname(f_path)))
        
        with self._zip.open(f_path, 'r') as f_obj:
            content = f_obj.read().decode('utf-8')
            
        vertices = []
        faces = []
        for line in content.splitlines():
            line = line.strip()
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
    """Digit3D Dataset converting 3D digit meshes into surface-sampled point clouds.

    Attributes:
        num_points (int): Number of surface points to sample uniformly per mesh.
    """

    def __init__(
        self,
        root: str = "~/.conquer3d/",
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False, 
        cached: bool = False,
        num_points: int = 512,
        return_img: bool = False
    ) -> None:
        """Initializes the PointDigit3D dataset instance."""
        super().__init__(root, train, transform, download, cached=cached, return_img=return_img)
        self.num_points = num_points
        
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]]:
        """Samples surface point clouds and returns (points, features, label, [image])."""
        if self.return_img:
            vertices, faces, label, img_t = super().__getitem__(idx)
        else:
            vertices, faces, label = super().__getitem__(idx)

        mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(), process=False)
        
        points_np, face_indices = trimesh.sample.sample_surface(mesh, self.num_points)
        normals_np = mesh.face_normals[face_indices]
        
        points = torch.tensor(points_np, dtype=torch.float32)
        normals = torch.tensor(normals_np, dtype=torch.float32)
        
        features = torch.cat([points, normals], dim=-1)
        
        if self.return_img:
            return points, features, label, img_t
        return points, features, label
