import os
import torch
import torchvision
import numpy as np
from scipy.ndimage import distance_transform_edt
import conquer3d as c3d
import argparse
import open3d as o3d
import tqdm
import multiprocessing as mp

def save_obj(filename, vert, face):
    with open(filename, "w") as f:
        for v in vert:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in face:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

def image_to_mesh(image_tensor, args, device="cuda"):
    img_np = image_tensor.squeeze().numpy()
    binary = img_np > 0.0
    
    sdf_2d = distance_transform_edt(~binary) - distance_transform_edt(binary)
    sdf_2d = sdf_2d / 14.0 
    
    if args.gaussian_blur:
        from scipy.ndimage import gaussian_filter
        sdf_2d = gaussian_filter(sdf_2d, sigma=1.0)
    
    grid_res = [32, 32, 32]
    # Z-Upward configuration: Thickness is now along the Y-axis (-0.6 to 0.6)
    grid_vertices, voxels, _ = c3d.data_structure.create_voxel_grid(
        grid_min=[-1.2, -0.6, -1.2], 
        grid_max=[1.2, 0.6, 1.2], 
        res=grid_res, 
        device=device
    )
    
    # Map grid coordinates X (width), Z (height) to [-1, 1] for grid_sample
    x_coords = grid_vertices[:, 0] / 1.2
    y_coords = -grid_vertices[:, 2] / 1.2  # Map Z axis to Image Y (Flip to keep upright)
    
    sample_coords = torch.stack([x_coords, y_coords], dim=-1).view(1, 1, -1, 2).float()
    sdf_2d_tensor = torch.from_numpy(sdf_2d).to(device).float().view(1, 1, 28, 28)
    
    sampled_sdf_2d = torch.nn.functional.grid_sample(
        sdf_2d_tensor, 
        sample_coords, 
        mode='bilinear', 
        padding_mode='border',
        align_corners=True
    ).view(-1)
    
    thickness = args.thickness
    # Extrusion axis is now Y
    extrusion_axis = grid_vertices[:, 1]
    
    if args.spherical_z:
        sdf_3d = sampled_sdf_2d + (extrusion_axis**2) / thickness
    else:
        sdf_3d = torch.max(sampled_sdf_2d, torch.abs(extrusion_axis) - thickness)
    
    mc_vertices, mc_faces, _, _ = c3d.ops.marching_cubes(
        grid_vertices=grid_vertices,
        voxels=voxels,
        voxel_values=sdf_3d.view(-1),
        iso=0.0
    )
    
    return mc_vertices, mc_faces

worker_dataset = None

def init_worker(is_train):
    global worker_dataset
    # Each worker initializes its own dataset instance to avoid shared memory file descriptors
    transform = torchvision.transforms.ToTensor()
    worker_dataset = torchvision.datasets.MNIST(root='./.cache', train=is_train, download=False, transform=transform)

def process_sample(item):
    global worker_dataset
    i, split_name, args, save_dir = item
    
    # Retrieve the image from the worker's local dataset instance
    img, label = worker_dataset[i]
    filename = os.path.join(save_dir, f"{label}_{i}.obj")
    
    vert, face = image_to_mesh(img, args, device="cuda")
    
    if vert is not None and vert.shape[0] > 0:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vert.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(face.cpu().numpy())
        
        if args.debug:
            save_obj(os.path.join("debug_mesh", f"{split_name.lower()}_{label}_{i}_step1_mc.obj"), np.asarray(mesh.vertices), np.asarray(mesh.triangles))
            
        if args.decimate:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.target_triangles)
            if args.debug:
                save_obj(os.path.join("debug_mesh", f"{split_name.lower()}_{label}_{i}_step2_decimated.obj"), np.asarray(mesh.vertices), np.asarray(mesh.triangles))
                
        if args.post_process:
            if args.smooth_type == "taubin":
                mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iter)
            elif args.smooth_type == "laplacian":
                mesh = mesh.filter_smooth_laplacian(number_of_iterations=args.smooth_iter)
            if args.debug:
                save_obj(os.path.join("debug_mesh", f"{split_name.lower()}_{label}_{i}_step3_smoothed.obj"), np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        
        vert = np.asarray(mesh.vertices)
        face = np.asarray(mesh.triangles)
        save_obj(filename, vert, face)
        
    if args.debug:
        debug_img_path = os.path.join("debug", f"{split_name.lower()}_{label}_{i}.png")
        torchvision.utils.save_image(img, debug_img_path)

def process_dataset(num_total_samples, is_train, split_name, args, save_dir):
    print(f"\nProcessing {split_name} set...")
    
    items = []
    num_samples = 1 if args.debug else num_total_samples
    for i in range(num_samples):
        items.append((i, split_name, args, save_dir))
        
    # Using spawn context prevents CUDA initialization conflicts across processes
    ctx = mp.get_context('spawn')
    num_workers = min(8, os.cpu_count() or 1)
    
    # Pass initializer to avoid sending PyTorch tensors through multiprocessing queue
    with ctx.Pool(processes=num_workers, initializer=init_worker, initargs=(is_train,)) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_sample, items), total=len(items)))
        
    if args.debug:
        print(f"Debug mode: Successfully processed first sample.")

def main():
    parser = argparse.ArgumentParser(description="Construct 3D MNIST Dataset")
    parser.add_argument("--debug", action="store_true", help="Run only for the first file in train and test sets")
    parser.add_argument("--post_process", action="store_true", help="Apply mesh smoothing")
    parser.add_argument("--smooth_iter", type=int, default=10, help="Number of smoothing iterations")
    parser.add_argument("--smooth_type", type=str, default="taubin", choices=["taubin", "laplacian"], help="Type of smoothing to apply")
    parser.add_argument("--gaussian_blur", action="store_true", help="Apply Gaussian blur to 2D SDF for organic rounding")
    parser.add_argument("--spherical_z", action="store_true", help="Use a swept-capsule SDF for rounded balloon-like Z-axis edges")
    parser.add_argument("--thickness", type=float, default=0.5, help="Thickness constraint for 3D SDF")
    parser.add_argument("--decimate", action="store_true", help="Apply quadric error decimation to the mesh")
    parser.add_argument("--target_triangles", type=int, default=1000, help="Target number of triangles for decimation")
    args = parser.parse_args()

    os.makedirs("src/train", exist_ok=True)
    os.makedirs("src/test", exist_ok=True)
    if args.debug:
        os.makedirs("debug", exist_ok=True)
        os.makedirs("debug_mesh", exist_ok=True)

    print("Downloading/Loading MNIST metadata...")
    transform = torchvision.transforms.ToTensor()
    # We load once in main to ensure download=True works securely, and get total lengths
    train_dataset = torchvision.datasets.MNIST(root='./.cache', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./.cache', train=False, download=True, transform=transform)

    process_dataset(len(train_dataset), True, "Train", args, "src/train")
    process_dataset(len(test_dataset), False, "Test", args, "src/test")

    print("\nDataset construction complete!")

if __name__ == "__main__":
    main()
