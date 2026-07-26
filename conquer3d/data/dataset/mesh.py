import os
import inspect
import hashlib
import torch
import trimesh
from multiprocessing import Pool
from typing import Callable, Optional, List, Union
from .base_mesh import BaseMeshDataset
import conquer3d.data.assets.common as common_assets


def _check_watertight(file_path: str) -> bool:
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
    """
    Dataset that queries all mesh files inside a root directory across all subdirectory depth levels.
    Supports file types such as obj, glb, gltf, etc.
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        types: Optional[List[str]] = None,
        cached: bool = False,
        return_hash_id: bool = False,
        watertight_only: bool = False,
    ):
        root = os.path.expanduser(root)
        super().__init__(root, transform)
        
        if types is None:
            types = ["obj"]
        
        # Normalize extensions to always start with a dot and be lowercase
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
                    
        # Sort for deterministic ordering across operating systems and runs
        mesh_files.sort()
        if self.watertight_only and mesh_files:
            with Pool() as pool:
                flags = pool.map(_check_watertight, mesh_files)
            mesh_files = [f for f, is_wt in zip(mesh_files, flags) if is_wt]
        return mesh_files

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int):
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
            
        # Safely load mesh using trimesh
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
    """
    A generic folder dataset where subfolders inside the root directory correspond to different classes.
    For example:
        root/chair/chair01.obj
        root/table/table01.obj
    Returns tuples of (vertices, faces, label) where label is the integer index of the subfolder class.
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        types: Optional[List[str]] = None,
        cached: bool = False,
        return_hash_id: bool = False,
        watertight_only: bool = False,
    ):
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

    def _find_classes(self) -> tuple[List[str], dict[str, int]]:
        if not os.path.exists(self.root):
            return [], {}
        classes = sorted([d.name for d in os.scandir(self.root) if d.is_dir()])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _query_samples(self) -> List[tuple[str, int]]:
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
                        
        # Sort samples by file path for deterministic ordering across runs and platforms
        samples.sort(key=lambda x: x[0])
        if self.watertight_only and samples:
            paths = [p for p, _ in samples]
            with Pool() as pool:
                flags = pool.map(_check_watertight, paths)
            samples = [s for s, is_wt in zip(samples, flags) if is_wt]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
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
            
        # Safely load mesh using trimesh
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
    """
    Dataset containing available test meshes from conquer3d.data.assets.common
    except 'woody', 'alligator', and 'iphigenia'.
    """
    def __init__(
        self,
        root: str = "~/.conquer3d/",
        transform: Optional[Callable] = None,
        cached: bool = False,
        return_hash_id: bool = False,
    ):
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
                    
        # Sort by class name for deterministic dataset ordering
        self.mesh_classes.sort(key=lambda x: x[0])
        self.names = [name for name, _ in self.mesh_classes]

    def __len__(self) -> int:
        return len(self.mesh_classes)

    def __getitem__(self, idx: int):
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