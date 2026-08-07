import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

# Standard Redwood Dataset Constants (PrimeSense Default)
REDWOOD_FX = 525.0
REDWOOD_FY = 525.0
REDWOOD_CX = 319.5
REDWOOD_CY = 239.5
REDWOOD_DEPTH_SCALE = 1000.0

REDWOOD_INTRINSICS = np.array([
    [REDWOOD_FX, 0.0, REDWOOD_CX],
    [0.0, REDWOOD_FY, REDWOOD_CY],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

class RedWood(Dataset):
    """
    PyTorch Dataset for loading the Redwood RGB-D indoor sequences (e.g., apartment).
    Automatically parses the .log trajectory file and synchronously loads the depth/color images.
    """
    def __init__(self, data_dir: str, scene_name: str = "", load_color: bool = True, depth_max: float = 3.0,
                 convert_rgb_to_intensity: bool = False, convert_rgb_to_intensity_type: str = 'weighted'):
        super().__init__()
        self.data_dir = os.path.join(data_dir, scene_name) if scene_name else data_dir
        self.scene_name = scene_name
        self.load_color = load_color
        self.depth_max = depth_max
        self.convert_rgb_to_intensity = convert_rgb_to_intensity
        self.convert_rgb_to_intensity_type = convert_rgb_to_intensity_type
        
        # Determine the log file name (e.g. apartment.log or just a generic trajectory.log)
        # If scene_name is provided, try scene_name.log, otherwise fallback to apartment.log
        log_name = f"{scene_name}.log" if scene_name else "apartment.log"
        self.log_path = os.path.join(self.data_dir, log_name)
        if not os.path.exists(self.log_path):
            self.log_path = os.path.join(self.data_dir, "apartment.log")
            
        json_name = f"{scene_name}.json" if scene_name else "apartment.json"
        self.json_path = os.path.join(self.data_dir, json_name)
        
        # 1. Parse the trajectory log file
        self.poses = self._parse_log_file(self.log_path)
        
        # 2. Collect image and depth paths
        # Adjust folder names if they were extracted differently (e.g. 'image' vs 'images')
        self.image_paths = sorted(glob.glob(os.path.join(self.data_dir, "image", "*.jpg")))
        self.depth_paths = sorted(glob.glob(os.path.join(self.data_dir, "depth", "*.png")))
        
        if len(self.poses) == 0:
            print(f"Warning: No poses found or missing apartment.log at {self.log_path}")
        if len(self.depth_paths) == 0:
            print(f"Warning: No depth images found in {os.path.join(self.data_dir, 'depth')}")
            
    def _parse_log_file(self, log_path: str) -> list:
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
                
            # Redwood metadata line: "frame_i frame_j num_frames"
            meta = line.split()
            if len(meta) == 3:
                # Read the next 4 lines as the 4x4 pose matrix (Camera-to-World)
                pose = []
                for j in range(4):
                    row = [float(x) for x in lines[i+1+j].strip().split()]
                    pose.append(row)
                poses.append(np.array(pose, dtype=np.float32))
                i += 5
            else:
                i += 1
                
        return poses

    def __len__(self):
        # The number of frames is dictated by the number of depth images
        return len(self.depth_paths)

    def __getitem__(self, idx):
        # 1. Load depth image (usually 16-bit PNG for Redwood)
        depth_path = self.depth_paths[idx]
        # Convert depth from millimeters to meters using the standard depth scale
        depth = np.array(Image.open(depth_path)).astype(np.float32) / REDWOOD_DEPTH_SCALE
        
        # Apply depth_max truncation (set distant points to 0.0, matching Open3D logic)
        depth[depth >= self.depth_max] = 0.0
        
        # 2. Load color if requested
        if self.load_color and idx < len(self.image_paths):
            image_path = self.image_paths[idx]
            color_img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)
            
            if self.convert_rgb_to_intensity:
                # Convert to intensity and scale to [0, 1] like Open3D
                if self.convert_rgb_to_intensity_type == 'weighted':
                    # 0.2990 * R + 0.5870 * G + 0.1140 * B
                    color = (0.2990 * color_img[..., 0] + 0.5870 * color_img[..., 1] + 0.1140 * color_img[..., 2]) / 255.0
                elif self.convert_rgb_to_intensity_type == 'equal':
                    # (R + G + B) / 3.0
                    color = (color_img[..., 0] + color_img[..., 1] + color_img[..., 2]) / 3.0 / 255.0
                else:
                    raise ValueError(f"Unknown convert_rgb_to_intensity_type: {self.convert_rgb_to_intensity_type}")
                color = np.expand_dims(color, axis=-1).astype(np.float32) # (H, W, 1)
            else:
                # Open3D's FloatImage format scales standard RGB to [0, 1] as well
                color = (color_img / 255.0).astype(np.float32) # (H, W, 3)
        else:
            # Default to a zero tensor if color loading is disabled or missing
            num_channels = 1 if self.convert_rgb_to_intensity else 3
            color = np.zeros((depth.shape[0], depth.shape[1], num_channels), dtype=np.float32)
            
        # 3. Retrieve Camera-to-World pose
        c2w = self.poses[idx]
        
        # 4. Invert to World-to-Camera (required for the CUDA integration kernel)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        
        return {
            "depth": torch.from_numpy(depth),
            "color": torch.from_numpy(color),
            "w2c": torch.from_numpy(w2c),
            "c2w": torch.from_numpy(c2w),
            "intrinsics": torch.from_numpy(REDWOOD_INTRINSICS)
        }
