import torch
import json
import os
from tqdm import tqdm
from glob import glob

EXCLUDE = ['woody', 'alligator', 'iphiagenia']

def count_grid_vertices(sub_folder, save_path):
    grid_verts = {}

    pts = glob(os.path.join(sub_folder, '*.pt'))

    if len(pts) == 0:
        print(f"No .pt files found in {sub_folder}")
        return

    for pt_path in tqdm(pts, desc="Evaluating .pt files"):
        filename = os.path.basename(pt_path)
        obj_name = filename.split('.')[0]

        if obj_name in EXCLUDE:
            continue

        try:
            # map_location='cpu' prevents unnecessary VRAM usage
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
            grid_verts[obj_name] = int(data['grid_vertices'].shape[0])
            del data
        except Exception as e:
            print(f"Error processing {pt_path}: {e}")
            raise RuntimeError(e)

    if not grid_verts:
        print(f"No successful evaluations for {sub_folder}")
        return

    # Report the results
    with open(save_path, 'w') as f:
        json.dump({'num_grid_vert': grid_verts}, f, indent=4)

if __name__ == "__main__":
    
    wd = os.path.dirname(os.path.abspath(__file__))
    
    for sub_folder in tqdm(os.listdir(wd), desc="Evaluating experiments"):

        full_sub_folder = os.path.join(wd, sub_folder)
        if not os.path.isdir(full_sub_folder):
            continue

        folder_name = os.path.basename(full_sub_folder)
        
        # Only process numeric folders representing resolutions
        if not folder_name.isdigit():
            continue

        print(f"Folder Name: {folder_name}")
        save_path = os.path.join(wd, f'{folder_name}_num_grid_vert.json')
        
        if os.path.exists(save_path):
            print(f"Skipping {save_path}")
            continue
            
        count_grid_vertices(full_sub_folder, save_path)
