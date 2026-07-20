import trimesh
import point_cloud_utils as pcu

import json
import os
from tqdm.notebook import tqdm
from glob import glob
import numpy as np

import conquer3d.data.assets as c3d_assets

EXCLUDE = ['woody', 'alligator', 'iphiagenia']

ASSET_CLASSES = {k.lower(): k for k in dir(c3d_assets) if not k.startswith('_')}

def sub_exp_eval(sub_folder, save_path):

    chamfer = {}
    hausdorff = {}

    objs = glob(os.path.join(sub_folder, '*.obj'))

    if len(objs) == 0:
        print(f"No meshes found in {sub_folder}")
        return

    for obj_path in tqdm(objs, desc="Evaluating meshes"):

        filename = os.path.basename(obj_path)
        obj_name = filename.split('.')[0]

        if obj_name in EXCLUDE:
            continue

        try:
            mesh = trimesh.load(obj_path, force='mesh')

            if obj_name not in ASSET_CLASSES:
                print(f"Asset {obj_name} not found in c3d_assets, skipping...")
                continue
            
            asset_class_name = ASSET_CLASSES[obj_name]
            asset_class = getattr(c3d_assets, asset_class_name)
            asset = asset_class()
            vertices, faces, _ = asset.get()
            
            gt_mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())

            # Normalization
            vmin = gt_mesh.vertices.min(axis=0)
            vmax = gt_mesh.vertices.max(axis=0)
            center = (vmin + vmax) / 2.0
            scale = 1.8 / (vmax - vmin).max()

            gt_mesh.vertices = (gt_mesh.vertices - center) * scale

            # Sample point clouds
            pc_mesh = trimesh.sample.sample_surface(mesh, 16384)[0]
            pc_gt = trimesh.sample.sample_surface(gt_mesh, 16384)[0]

            # Calculate metrics
            chamfer[obj_name] = pcu.chamfer_distance(pc_mesh, pc_gt)
            hausdorff[obj_name] = pcu.hausdorff_distance(pc_mesh, pc_gt)

            # Explicitly delete large objects to free memory
            del mesh, gt_mesh, pc_mesh, pc_gt

        except Exception as e:
            print(f"Error processing {obj_path}: {e}")
            raise RuntimeError(e)

    if not chamfer or not hausdorff:
        print(f"No successful evaluations for {sub_folder}")
        return

    avg_chamfer_distance = np.mean(list(chamfer.values()))
    avg_hausdorff_distance = np.mean(list(hausdorff.values()))

    with open(save_path, 'w') as f:
        json.dump(
            {
                'avg_chamfer': avg_chamfer_distance,
                'avg_hausdorff': avg_hausdorff_distance,
                'chamfer': chamfer,
                'hausdorff': hausdorff
            }, f)

if __name__ == "__main__":
    
    wd = os.path.dirname(os.path.abspath(__file__))
    
    for sub_folder in tqdm(os.listdir(wd), desc="Evaluating experiments"):

        sub_folder = os.path.join(wd, sub_folder)
        if not os.path.isdir(sub_folder):
            continue

        folder_name = os.path.basename(sub_folder)

        print(f"Folder Name: {folder_name}")
        save_path = os.path.join(wd, f'{folder_name}.json')
        if os.path.exists(save_path):
            print(f"Skipping {save_path}")
            continue
        sub_exp_eval(sub_folder, save_path)