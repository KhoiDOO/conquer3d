"""Redwood indoor RGB-D scan and camera trajectory dataset loader.

This module provides data loading interfaces for the Redwood RGB-D benchmark dataset,
parsing camera trajectories from `.log` files and loading synchronized depth/color frames.
"""

from typing import Dict, Any, List
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

# Standard Redwood Dataset Constants (PrimeSense Default)
REDWOOD_FX: float = 525.0
REDWOOD_FY: float = 525.0
REDWOOD_CX: float = 319.5
REDWOOD_CY: float = 239.5
REDWOOD_DEPTH_SCALE: float = 1000.0

REDWOOD_INTRINSICS: np.ndarray = np.array([
    [REDWOOD_FX, 0.0, REDWOOD_CX],
    [0.0, REDWOOD_FY, REDWOOD_CY],
    [0.0, 0.0, 1.0]
], dtype=np.float32)


class RedWood(Dataset):
    """PyTorch Dataset for loading Redwood RGB-D indoor sequences.

    Automatically parses trajectory `.log` files and synchronously delivers
    depth maps, color frames, and extrinsic/intrinsic camera matrices.

    Attributes:
        data_dir (str): Root path to sequence directory.
        scene_name (str): Name of scene (e.g. `'apartment'`).
        load_color (bool): Whether RGB frames are loaded.
        depth_max (float): Maximum valid depth threshold in meters.
        convert_rgb_to_intensity (bool): Whether to convert RGB to 1-channel grayscale intensity.
        poses (List[np.ndarray]): List of 4x4 Camera-to-World (C2W) pose matrices.
        image_paths (List[str]): List of sorted RGB image filepaths.
        depth_paths (List[str]): List of sorted depth map filepaths.
    """

    def __init__(
        self,
        data_dir: str,
        scene_name: str = "",
        load_color: bool = True,
        depth_max: float = 3.0,
        convert_rgb_to_intensity: bool = False,
        convert_rgb_to_intensity_type: str = 'weighted'
    ) -> None:
        """Initializes the Redwood RGB-D dataset loader.

        Args:
            data_dir (str): Directory containing the scene folders.
            scene_name (str, optional): Subfolder name of the sequence. Defaults to `""`.
            load_color (bool, optional): Whether to load RGB frames. Defaults to True.
            depth_max (float, optional): Maximum valid depth in meters. Defaults to 3.0.
            convert_rgb_to_intensity (bool, optional): Convert RGB to scalar intensity. Defaults to False.
            convert_rgb_to_intensity_type (str, optional): Intensity conversion type (`'weighted'` or `'equal'`).
                Defaults to `'weighted'`.
        """
        super().__init__()
        self.data_dir = os.path.join(data_dir, scene_name) if scene_name else data_dir
        self.scene_name = scene_name
        self.load_color = load_color
        self.depth_max = depth_max
        self.convert_rgb_to_intensity = convert_rgb_to_intensity
        self.convert_rgb_to_intensity_type = convert_rgb_to_intensity_type
        
        log_name = f"{scene_name}.log" if scene_name else "apartment.log"
        self.log_path = os.path.join(self.data_dir, log_name)
        if not os.path.exists(self.log_path):
            self.log_path = os.path.join(self.data_dir, "apartment.log")
            
        json_name = f"{scene_name}.json" if scene_name else "apartment.json"
        self.json_path = os.path.join(self.data_dir, json_name)
        
        self.poses = self._parse_log_file(self.log_path)
        self.image_paths = sorted(glob.glob(os.path.join(self.data_dir, "image", "*.jpg")))
        self.depth_paths = sorted(glob.glob(os.path.join(self.data_dir, "depth", "*.png")))
        
        if len(self.poses) == 0:
            print(f"Warning: No poses found or missing apartment.log at {self.log_path}")
        if len(self.depth_paths) == 0:
            print(f"Warning: No depth images found in {os.path.join(self.data_dir, 'depth')}")
            
    def _parse_log_file(self, log_path: str) -> List[np.ndarray]:
        poses = []
        if not os.path.exists(log_path):
            return poses
            
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            meta = line.split()
            if len(meta) == 3:
                pose = []
                for j in range(4):
                    row = [float(x) for x in lines[i+1+j].strip().split()]
                    pose.append(row)
                poses.append(np.array(pose, dtype=np.float32))
                i += 5
            else:
                i += 1
                
        return poses

    def __len__(self) -> int:
        """int: Total number of frames in sequence."""
        return len(self.depth_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Loads and returns synchronized frame dictionary `{'depth', 'color', 'w2c', 'c2w', 'intrinsics'}`."""
        depth_path = self.depth_paths[idx]
        depth = np.array(Image.open(depth_path)).astype(np.float32) / REDWOOD_DEPTH_SCALE
        
        depth[depth >= self.depth_max] = 0.0
        
        if self.load_color and idx < len(self.image_paths):
            image_path = self.image_paths[idx]
            color_img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)
            
            if self.convert_rgb_to_intensity:
                if self.convert_rgb_to_intensity_type == 'weighted':
                    color = (0.2990 * color_img[..., 0] + 0.5870 * color_img[..., 1] + 0.1140 * color_img[..., 2]) / 255.0
                elif self.convert_rgb_to_intensity_type == 'equal':
                    color = (color_img[..., 0] + color_img[..., 1] + color_img[..., 2]) / 3.0 / 255.0
                else:
                    raise ValueError(f"Unknown convert_rgb_to_intensity_type: {self.convert_rgb_to_intensity_type}")
                color = np.expand_dims(color, axis=-1).astype(np.float32)
            else:
                color = (color_img / 255.0).astype(np.float32)
        else:
            num_channels = 1 if self.convert_rgb_to_intensity else 3
            color = np.zeros((depth.shape[0], depth.shape[1], num_channels), dtype=np.float32)
            
        c2w = self.poses[idx]
        w2c = np.linalg.inv(c2w).astype(np.float32)
        
        return {
            "depth": torch.from_numpy(depth),
            "color": torch.from_numpy(color),
            "w2c": torch.from_numpy(w2c),
            "c2w": torch.from_numpy(c2w),
            "intrinsics": torch.from_numpy(REDWOOD_INTRINSICS)
        }
