"""Generic mesh directory datasets and multi-class folder loaders.

This module provides PyTorch datasets for traversing directories and loading
various mesh formats (.obj, .glb, .gltf, .off, .ply, .stl) into PyTorch tensors.
"""

from typing import Callable, Optional, List, Union, Tuple, Dict, Any
import os
import inspect
import hashlib
import torch
import trimesh
from multiprocessing import Pool
from .base_mesh import BaseMeshDataset
import conquer3d.data.assets.common as common_assets


def _check_watertight(file_path: str) -> bool:
    """Checks if a mesh file represents a watertight 2-manifold surface.

    Args:
        file_path (str): Absolute or relative filesystem path to the 3D file.

    Returns:
        bool: True if the mesh is watertight, False otherwise.
    """
    try:
        mesh = trimesh.load(file_path, process=False, force='mesh', skip_materials=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if isinstance(mesh, (list, tuple)):
            mesh = trimesh.util.concatenate(mesh)
        return bool(getattr(mesh, 'is_watertight', False))
    except Exception:
        return False


class MeshDataset(BaseMeshDataset):
    """Dataset that recursively queries all mesh files inside a root directory across all depths.

    Supports arbitrary 3D file extensions (.obj, .glb, .gltf, .ply, .off, .stl, etc.).

    Attributes:
        root (str): Expanded root directory path.
        types (set): Supported lowercase file extensions.
        cached (bool): Whether samples are cached in host memory.
        return_hash_id (bool): Whether to return a deterministic MD5 hash string per mesh.
        watertight_only (bool): If True, filters out non-watertight meshes via multiprocessing.
        all_files (List[str]): Sorted list of discovered valid mesh paths.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        types: Optional[List[str]] = None,
        cached: bool = False,
        return_hash_id: bool = False,
        watertight_only: bool = False,
    ) -> None:
        """Initializes the MeshDataset.

        Args:
            root (str): Filesystem path to root directory.
            transform (Callable, optional): Geometric transform applied to (vertices, faces). Defaults to None.
            types (List[str], optional): Allowed file extensions (e.g. `['obj', 'ply']`). Defaults to `['obj']`.
            cached (bool, optional): Cache parsed meshes in memory. Defaults to False.
            return_hash_id (bool, optional): Include MD5 hash string in return tuple. Defaults to False.
            watertight_only (bool, optional): Filter out non-watertight models. Defaults to False.
        """
        root = os.path.expanduser(root)
        super().__init__(root, transform)
        
        if types is None:
            types = ["obj"]
        
        self.types = {f".{t.lstrip('.').lower()}" for t in types}
        self.cached = cached
        self._cache = {}
        self.return_hash_id = return_hash_id
        self.watertight_only = watertight_only
        
        self.all_files = self._query_mesh_files()

    def _query_mesh_files(self) -> List[str]:
        mesh_files = []
        if not os.path.exists(self.root):
            return mesh_files
            
        for dirpath, _, filenames in os.walk(self.root):
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in self.types:
                    mesh_files.append(os.path.join(dirpath, f))
                    
        mesh_files.sort()
        if self.watertight_only and mesh_files:
            with Pool() as pool:
                flags = pool.map(_check_watertight, mesh_files)
            mesh_files = [f for f, is_wt in zip(mesh_files, flags) if is_wt]
        return mesh_files

    def __len__(self) -> int:
        """int: Total number of valid mesh files found."""
        return len(self.all_files)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
        """Loads and returns 3D mesh `(vertices, faces, [hash_id])`."""
        f_path = self.all_files[idx]
        hash_id = None
        if self.return_hash_id:
            rel_path = os.path.relpath(f_path, start=self.root).replace("\\", "/")
            hash_id = hashlib.md5(rel_path.encode('utf-8')).hexdigest()

        if self.cached and idx in self._cache:
            vertices_t, faces_t = self._cache[idx]
            if self.transform:
                vertices_t, faces_t = self.transform(vertices_t.clone(), faces_t.clone())
            if self.return_hash_id:
                return vertices_t, faces_t, hash_id
            return vertices_t, faces_t
            
        mesh = trimesh.load(f_path, process=False, force='mesh', skip_materials=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if isinstance(mesh, (list, tuple)):
            mesh = trimesh.util.concatenate(mesh)
            
        vertices_t = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces_t = torch.tensor(mesh.faces, dtype=torch.int32)
        
        if self.cached:
            self._cache[idx] = (vertices_t, faces_t)
            
        if self.transform:
            vertices_t, faces_t = self.transform(vertices_t.clone(), faces_t.clone())
            
        if self.return_hash_id:
            return vertices_t, faces_t, hash_id
            
        return vertices_t, faces_t


class MeshFolderDataset(BaseMeshDataset):
    """Multi-class folder dataset where immediate subdirectories correspond to discrete class labels.

    Example layout:
        root/chair/chair01.obj
        root/table/table01.obj

    Returns tuples of `(vertices, faces, label, [hash_id])`.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        types: Optional[List[str]] = None,
        cached: bool = False,
        return_hash_id: bool = False,
        watertight_only: bool = False,
    ) -> None:
        """Initializes the MeshFolderDataset."""
        root = os.path.expanduser(root)
        super().__init__(root, transform)
        
        if types is None:
            types = ["obj"]
        
        self.types = {f".{t.lstrip('.').lower()}" for t in types}
        self.cached = cached
        self._cache = {}
        self.return_hash_id = return_hash_id
        self.watertight_only = watertight_only
        
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._query_samples()
        self.all_files = [path for path, _ in self.samples]

    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        if not os.path.exists(self.root):
            return [], {}
        classes = sorted([d.name for d in os.scandir(self.root) if d.is_dir()])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _query_samples(self) -> List[Tuple[str, int]]:
        samples = []
        if not os.path.exists(self.root):
            return samples
            
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root, cls_name)
            cls_idx = self.class_to_idx[cls_name]
            for dirpath, _, filenames in os.walk(cls_dir):
                for f in filenames:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in self.types:
                        samples.append((os.path.join(dirpath, f), cls_idx))
                        
        samples.sort(key=lambda x: x[0])
        if self.watertight_only and samples:
            paths = [p for p, _ in samples]
            with Pool() as pool:
                flags = pool.map(_check_watertight, paths)
            samples = [s for s, is_wt in zip(samples, flags) if is_wt]
        return samples

    def __len__(self) -> int:
        """int: Total number of labeled mesh samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, int, str]]:
        """Loads and returns `(vertices, faces, class_label, [hash_id])`."""
        f_path, label = self.samples[idx]
        hash_id = None
        if self.return_hash_id:
            rel_path = os.path.relpath(f_path, start=self.root).replace("\\", "/")
            hash_id = hashlib.md5(rel_path.encode('utf-8')).hexdigest()

        if self.cached and idx in self._cache:
            vertices_t, faces_t, label = self._cache[idx]
            if self.transform:
                vertices_t, faces_t = self.transform(vertices_t.clone(), faces_t.clone())
            if self.return_hash_id:
                return vertices_t, faces_t, label, hash_id
            return vertices_t, faces_t, label
            
        mesh = trimesh.load(f_path, process=False, force='mesh', skip_materials=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if isinstance(mesh, (list, tuple)):
            mesh = trimesh.util.concatenate(mesh)
            
        vertices_t = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces_t = torch.tensor(mesh.faces, dtype=torch.int32)
        
        if self.cached:
            self._cache[idx] = (vertices_t, faces_t, label)
            
        if self.transform:
            vertices_t, faces_t = self.transform(vertices_t.clone(), faces_t.clone())
            
        if self.return_hash_id:
            return vertices_t, faces_t, label, hash_id
            
        return vertices_t, faces_t, label


class ToyMeshDataset(BaseMeshDataset):
    """Benchmark test dataset aggregating standard 3D meshes from `conquer3d.data.assets.common`."""

    def __init__(
        self,
        root: str = "~/.conquer3d/",
        transform: Optional[Callable] = None,
        cached: bool = False,
        return_hash_id: bool = False,
    ) -> None:
        """Initializes the ToyMeshDataset."""
        root = os.path.expanduser(root)
        super().__init__(root, transform)
        self.cached = cached
        self._cache = {}
        self.return_hash_id = return_hash_id
        
        exclude_names = {"woody", "alligator", "iphigenia", "iphiagenia", "beetle", 'beetlealt'}
        self.mesh_classes = []
        for name, cls in inspect.getmembers(common_assets, inspect.isclass):
            if issubclass(cls, common_assets.Common3D) and cls is not common_assets.Common3D:
                if name.lower() not in exclude_names:
                    self.mesh_classes.append((name, cls))
                    
        self.mesh_classes.sort(key=lambda x: x[0])
        self.names = [name for name, _ in self.mesh_classes]

    def __len__(self) -> int:
        """int: Number of benchmark models."""
        return len(self.mesh_classes)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
        """Retrieves `(vertices, faces, [asset_name])`."""
        if self.cached and idx in self._cache:
            vertices_t, faces_t = self._cache[idx]
            if self.transform:
                vertices_t, faces_t = self.transform(vertices_t.clone(), faces_t.clone())
            if self.return_hash_id:
                return vertices_t, faces_t, self.names[idx]
            return vertices_t, faces_t
            
        _, cls_type = self.mesh_classes[idx]
        asset = cls_type(download_dir=self.root)
        
        vertices_t = asset.vertices.clone().to(torch.float32)
        faces_t = asset.faces.clone().to(torch.int32)
        
        if self.cached:
            self._cache[idx] = (vertices_t, faces_t)
            
        if self.transform:
            vertices_t, faces_t = self.transform(vertices_t.clone(), faces_t.clone())
            
        if self.return_hash_id:
            return vertices_t, faces_t, self.names[idx]
            
        return vertices_t, faces_t