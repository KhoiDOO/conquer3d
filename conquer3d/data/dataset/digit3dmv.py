"""Digit3DMV: 3D MNIST Multi-View Normal and Depth Map Dataset.

This module provides dataset loaders for Digit3DMV, offering multi-view normal
and depth maps rendered under a right-handed Z-upward coordinate system with 
canonical camera-to-world (c2w) matrices, flexible view selections, and 
on-the-fly zip extraction.
"""

from typing import Tuple, Optional, Callable, Union, Dict, Any, List, Literal
import os
import zipfile
import io
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class Digit3DMV(Dataset):
    """Digit3DMV Dataset containing multi-view normal and depth maps for 3D MNIST digits.

    Attributes:
        root (str): Root cache directory path containing archives or uncompressed folders.
        train (bool): If True, loads the training split (60,000 samples); if False, test split (10,000 samples).
        resolution (int): Rendered image resolution (64 or 128).
        modality (str): Image modality to load: `'normal'`, `'depth'`, or `'both'`.
        all_360 (bool): If True, returns all 12 diagonal 360 views; if False, returns 4 equatorial views.
        return_front_variation (bool): If True, also returns the 8 perturbed front views.
        return_c2w (bool): If True, returns 4x4 camera-to-world (c2w) transformation matrices.
        transform (Callable, optional): Optional transformation applied to loaded image tensors.
        cached (bool): If True, caches parsed sample dicts in memory for fast training.
        use_zip (bool): Whether data is loaded directly from a .zip archive or an uncompressed directory.
    """

    # Google Drive File IDs for automatic download
    GDRIVE_IDS = {
        64: "1Ix017GWt6H9RIZJ7MIKpa9vQTJWjLizu",
        128: "1sz3w8KP1snu6HpdqNqLMwhXn2sHQJFJP",
    }

    def __init__(
        self,
        root: str = "~/.conquer3d/",
        train: bool = True,
        resolution: int = 64,
        modality: Literal["normal", "depth", "both"] = "normal",
        all_360: bool = False,
        return_front_variation: bool = False,
        return_c2w: bool = False,
        transform: Optional[Callable] = None,
        use_zip: Optional[bool] = None,
        cached: bool = False,
        download: bool = False,
    ) -> None:
        """Initializes the Digit3DMV dataset instance.

        Args:
            root (str, optional): Root folder containing dataset archives/directories. Defaults to `"~/.conquer3d/"`.
            train (bool, optional): Whether to load train or test split. Defaults to True.
            resolution (int, optional): Image resolution (64 or 128). Defaults to 64.
            modality (str, optional): Target modality ('normal', 'depth', or 'both'). Defaults to 'normal'.
            all_360 (bool, optional): Return 12 360 views if True, or 4 equatorial views if False. Defaults to False.
            return_front_variation (bool, optional): Return 8 front angle variation views. Defaults to False.
            return_c2w (bool, optional): Return camera-to-world transformation matrices. Defaults to False.
            transform (Callable, optional): Transform function applied to image tensors. Defaults to None.
            use_zip (bool, optional): Force zip loading vs directory mode. If None, auto-detects. Defaults to None.
            cached (bool, optional): In-memory sample caching. Defaults to False.
            download (bool, optional): Automatically download dataset archive if missing. Defaults to False.

        Raises:
            ValueError: If resolution or modality is not supported.
            RuntimeError: If dataset files are not found and `download=False`.
        """
        super().__init__()
        if resolution not in (64, 128):
            raise ValueError(f"Unsupported resolution: {resolution}. Supported resolutions are 64 and 128.")
        if modality not in ("normal", "depth", "both"):
            raise ValueError(f"Unsupported modality: '{modality}'. Choose from 'normal', 'depth', 'both'.")

        self.root = os.path.expanduser(root)
        self.train = train
        self.resolution = resolution
        self.modality = modality
        self.all_360 = all_360
        self.return_front_variation = return_front_variation
        self.return_c2w = return_c2w
        self.transform = transform
        self.cached = cached
        self._cache = {}
        self._zip = None

        self.split_name = "train" if train else "test"
        
        # View selection indices
        self.indices_360 = list(range(12)) if all_360 else list(range(4))
        self.indices_front_var = list(range(1, 9))

        # Determine archive and directory candidates
        zip_candidates = [
            os.path.join(self.root, f"mv_{resolution}.zip"),
            os.path.join(self.root, f"digit3dmv_{resolution}.zip"),
            os.path.join(self.root, "data", f"mv_{resolution}.zip") if os.path.isdir(os.path.join(self.root, "data")) else "",
        ]
        dir_candidates = [
            os.path.join(self.root, f"mv_{resolution}"),
            os.path.join(self.root, "data", f"mv_{resolution}"),
            self.root if os.path.basename(self.root) == f"mv_{resolution}" else "",
        ]

        if download:
            self.download()

        # Resolve storage mode: Zip vs Uncompressed Directory
        resolved_zip = next((p for p in zip_candidates if p and os.path.isfile(p)), None)
        resolved_dir = next((p for p in dir_candidates if p and os.path.isdir(p)), None)

        if use_zip is True:
            if not resolved_zip:
                raise RuntimeError(f"Zip archive for resolution {resolution} not found in {self.root}.")
            self.use_zip = True
            self.data_source = resolved_zip
        elif use_zip is False:
            if not resolved_dir:
                raise RuntimeError(f"Uncompressed directory mv_{resolution} not found in {self.root}.")
            self.use_zip = False
            self.data_source = resolved_dir
        else:
            # Auto-detect: prefer uncompressed directory if available, else zip archive
            if resolved_dir and os.path.isdir(os.path.join(resolved_dir, self.split_name)):
                self.use_zip = False
                self.data_source = resolved_dir
            elif resolved_zip:
                self.use_zip = True
                self.data_source = resolved_zip
            elif resolved_dir:
                self.use_zip = False
                self.data_source = resolved_dir
            else:
                raise RuntimeError(
                    f"Digit3DMV (resolution {resolution}) not found at {self.root}. "
                    f"Provide 'mv_{resolution}.zip' or directory 'mv_{resolution}/', or set download=True."
                )

        # Index dataset samples and parse camera metadata
        self._init_samples_and_cameras()

    def _init_samples_and_cameras(self) -> None:
        """Discovers sample identifiers and parses global camera trajectory metadata."""
        if self.use_zip:
            with zipfile.ZipFile(self.data_source, 'r') as z:
                namelist = z.namelist()
                
                # Detect directory prefix inside zip (e.g., 'mv_64/test/' vs 'test/')
                prefix = ""
                for n in namelist:
                    if f"/{self.split_name}/" in n:
                        prefix = n.split(f"/{self.split_name}/")[0] + "/"
                        break
                    elif n.startswith(f"{self.split_name}/"):
                        prefix = ""
                        break
                
                self.zip_prefix = prefix
                split_prefix = f"{prefix}{self.split_name}/"
                
                # Discover unique sample names
                sample_set = set()
                for name in namelist:
                    if name.startswith(split_prefix):
                        rel = name[len(split_prefix):]
                        parts = rel.split('/')
                        if len(parts) > 1 and parts[0]:
                            sample_set.add(parts[0])
                self.samples = sorted(list(sample_set))

                # Load global or first-sample cameras.json
                cam_json_name = f"{prefix}cameras.json"
                if cam_json_name in namelist:
                    cam_bytes = z.read(cam_json_name)
                    self.camera_meta = json.loads(cam_bytes.decode('utf-8'))
                elif self.samples:
                    sample_cam = f"{split_prefix}{self.samples[0]}/cameras.json"
                    cam_bytes = z.read(sample_cam)
                    self.camera_meta = json.loads(cam_bytes.decode('utf-8'))
                else:
                    self.camera_meta = {}
        else:
            split_dir = os.path.join(self.data_source, self.split_name)
            if not os.path.exists(split_dir):
                raise RuntimeError(f"Split directory '{split_dir}' does not exist.")
                
            self.samples = sorted([
                d for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d)) and "_" in d
            ])

            global_cam = os.path.join(self.data_source, "cameras.json")
            if os.path.exists(global_cam):
                with open(global_cam, 'r') as f:
                    self.camera_meta = json.load(f)
            elif self.samples:
                sample_cam = os.path.join(split_dir, self.samples[0], "cameras.json")
                with open(sample_cam, 'r') as f:
                    self.camera_meta = json.load(f)
            else:
                self.camera_meta = {}

        # Precompute PyTorch c2w tensors for quick retrieval
        self._c2w_tensors = {}
        if "views" in self.camera_meta:
            views = self.camera_meta["views"]
            
            # Front canonical view pose
            if "front/view_00" in views:
                self._c2w_tensors["front"] = torch.tensor(
                    views["front/view_00"]["cam2world_matrix"], dtype=torch.float32
                )
                
            # Front variation view poses
            front_vars = []
            for i in self.indices_front_var:
                k = f"front/view_{i:02d}"
                if k in views:
                    front_vars.append(views[k]["cam2world_matrix"])
            if front_vars:
                self._c2w_tensors["front_variation"] = torch.tensor(front_vars, dtype=torch.float32)

            # 360 view poses
            c360_list = []
            for i in self.indices_360:
                k = f"360/view_{i:02d}"
                if k in views:
                    c360_list.append(views[k]["cam2world_matrix"])
            if c360_list:
                self._c2w_tensors["360"] = torch.tensor(c360_list, dtype=torch.float32)

    def download(self) -> None:
        """Downloads the dataset zip archive via Google Drive using gdown."""
        target_zip = os.path.join(self.root, f"mv_{self.resolution}.zip")
        if os.path.exists(target_zip):
            return
        os.makedirs(self.root, exist_ok=True)
        file_id = self.GDRIVE_IDS.get(self.resolution)
        if not file_id or "placeholder" in file_id:
            raise NotImplementedError(
                f"Automatic download for resolution {self.resolution} is not configured with a valid Google Drive ID."
            )
        try:
            import gdown
        except ImportError:
            raise ImportError("gdown is required for dataset download. Install with 'pip install gdown'.")
            
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading Digit3DMV ({self.resolution}x{self.resolution}) to {target_zip}...")
        gdown.download(url, target_zip, quiet=False)

    def __len__(self) -> int:
        """int: Number of mesh samples in the loaded split."""
        return len(self.samples)

    def _read_image(self, rel_path: str, is_depth: bool = False) -> torch.Tensor:
        """Reads and converts an image file to a PyTorch float32 tensor [C, H, W] in [0, 1]."""
        if self.use_zip:
            pid = os.getpid()
            if self._zip is None or getattr(self, "_pid", None) != pid:
                self._zip = zipfile.ZipFile(self.data_source, 'r')
                self._pid = pid
            full_path = f"{self.zip_prefix}{rel_path}"
            img_bytes = self._zip.read(full_path)
            img_pil = Image.open(io.BytesIO(img_bytes))
        else:
            full_path = os.path.join(self.data_source, rel_path)
            img_pil = Image.open(full_path)

        if is_depth:
            img_pil = img_pil.convert('L')
            tensor = TF.to_tensor(img_pil)  # [1, H, W]
        else:
            img_pil = img_pil.convert('RGB')
            tensor = TF.to_tensor(img_pil)  # [3, H, W]

        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads and returns multi-view normal and/or depth images, c2w matrices, and label.

        Args:
            idx (int): Sample index.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'front': [C, H, W] float32 tensor (where C=3 for normal, 1 for depth)
                - '360': [V, C, H, W] float32 tensor (V=4 or V=12)
                - 'front_variation': [8, C, H, W] float32 tensor (if return_front_variation=True)
                - 'c2w_front': [4, 4] float32 tensor (if return_c2w=True)
                - 'c2w_360': [V, 4, 4] float32 tensor (if return_c2w=True)
                - 'c2w_front_variation': [8, 4, 4] float32 tensor (if return_c2w=True and return_front_variation=True)
                - 'label': int ground-truth digit class (0-9)
                - 'sample_name': str identifier (e.g. '0_10')
        """
        if self.cached and idx in self._cache:
            return self._cache[idx]

        sample_name = self.samples[idx]
        label = int(sample_name.split('_')[0])
        sample_rel = f"{self.split_name}/{sample_name}"

        result: Dict[str, Any] = {
            "label": label,
            "sample_name": sample_name,
        }

        # Helper to load views for a modality
        def _load_modality_views(mod_name: str, is_depth: bool) -> Dict[str, torch.Tensor]:
            views_dict = {}
            # 1. Front view
            front_rel = f"{sample_rel}/{mod_name}/front/view_00.jpg"
            views_dict["front"] = self._read_image(front_rel, is_depth=is_depth)

            # 2. 360 views (4 or 12)
            c360_tensors = []
            for i in self.indices_360:
                p = f"{sample_rel}/{mod_name}/360/view_{i:02d}.jpg"
                c360_tensors.append(self._read_image(p, is_depth=is_depth))
            views_dict["360"] = torch.stack(c360_tensors, dim=0)

            # 3. Optional Front variations
            if self.return_front_variation:
                fvar_tensors = []
                for i in self.indices_front_var:
                    p = f"{sample_rel}/{mod_name}/front/view_{i:02d}.jpg"
                    fvar_tensors.append(self._read_image(p, is_depth=is_depth))
                views_dict["front_variation"] = torch.stack(fvar_tensors, dim=0)
                
            return views_dict

        if self.modality in ("normal", "depth"):
            is_depth = (self.modality == "depth")
            views = _load_modality_views(self.modality, is_depth)
            result.update(views)
        elif self.modality == "both":
            norm_views = _load_modality_views("normal", is_depth=False)
            depth_views = _load_modality_views("depth", is_depth=True)
            result["normal_front"] = norm_views["front"]
            result["normal_360"] = norm_views["360"]
            result["depth_front"] = depth_views["front"]
            result["depth_360"] = depth_views["360"]
            if self.return_front_variation:
                result["normal_front_variation"] = norm_views["front_variation"]
                result["depth_front_variation"] = depth_views["front_variation"]

        # Camera-to-World (c2w) matrices
        if self.return_c2w:
            if "front" in self._c2w_tensors:
                result["c2w_front"] = self._c2w_tensors["front"]
            if "360" in self._c2w_tensors:
                result["c2w_360"] = self._c2w_tensors["360"]
            if self.return_front_variation and "front_variation" in self._c2w_tensors:
                result["c2w_front_variation"] = self._c2w_tensors["front_variation"]
            if "fov_deg" in self.camera_meta:
                result["fov_deg"] = self.camera_meta["fov_deg"]

        if self.cached:
            self._cache[idx] = result

        return result
