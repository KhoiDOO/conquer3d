import argparse
import sys
import os

if os.path.abspath('../../') not in sys.path:
    sys.path.append(os.path.abspath('../../'))
if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

import torch
import tqdm
import kaolin as kal
import render
import nvdiffrast.torch as dr

from conquer3d.ops.delaunay_triangulation import tetrahedralize, get_edges
from conquer3d.ops.marching_tetrahedra import marching_tetrahedra
from conquer3d.io.obj import write_obj
import conquer3d.data.assets as c3d_assets
from conquer3d.data_structure.grid import create_random_points_ball

def compute_angle(a, b, c):
    """
    Compute the angle between vectors a-b and c-b (in radians)
    """
    ab = a - b
    cb = c - b
    cosine_angle = torch.sum(ab * cb, dim=-1) / ((torch.norm(ab, dim=-1) * torch.norm(cb, dim=-1))+1e-6)
    angle = torch.acos(cosine_angle.clamp(-1, 1))  
    return angle

def triangle_angle_fairness(vertices, faces):
    """
    Compute the fairness loss for a mesh (vertices and faces).
    
    Args:
    - vertices: Tensor of shape (V, 3), where V is the number of vertices in the mesh.
    - faces: Tensor of shape (F, 3), where F is the number of triangular faces.

    Returns:
    - loss: A scalar value representing the fairness loss of the whole mesh.
    """
    tri_verts = vertices[faces]
    angles = torch.zeros((faces.shape[0], 3), device=vertices.device)
    angles[:, 0] = compute_angle(tri_verts[:, 1, :], tri_verts[:, 0, :], tri_verts[:, 2, :])
    angles[:, 1] = compute_angle(tri_verts[:, 0, :], tri_verts[:, 1, :], tri_verts[:, 2, :])
    angles[:, 2] = compute_angle(tri_verts[:, 0, :], tri_verts[:, 2, :], tri_verts[:, 1, :])

    target_angle = torch.pi / 3
    target_angle_tensor = torch.ones_like(angles) * target_angle
    fairness_loss = torch.nn.functional.mse_loss(angles, target_angle_tensor)

    return fairness_loss

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

def main():
    parser = argparse.ArgumentParser(description="Differentiable Marching Tetrahedra Optimization")
    parser.add_argument('--input', type=str, required=True, help='Name of asset from conquer3d/data/assets (e.g., Bunny, Iphigenia)')
    parser.add_argument('--output', type=str, required=True, help='Output obj file path')
    parser.add_argument('--lr_sdf', type=float, default=0.002, help='Learning rate for sdf')
    parser.add_argument('--lr_grid_verts', type=float, default=0.0003, help='Learning rate for grid vertices')
    parser.add_argument('--epoch', type=int, default=8000, help='Number of optimization epochs')
    parser.add_argument('--use_fairness', action='store_true', help='Turn on/off triangle fairness loss')
    parser.add_argument('--weight_fairness', type=float, default=0.35, help='Weight for fairness loss')
    parser.add_argument('--weight_mask', type=float, default=10.0, help='Weight for mask loss')
    parser.add_argument('--weight_depth', type=float, default=250.0, help='Weight for depth loss')
    parser.add_argument('--weight_normal', type=float, default=1.0, help='Weight for normal loss')
    parser.add_argument('--weight_sdf_reg', type=float, default=0.1, help='Weight for sdf regularization loss')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Asset
    if not hasattr(c3d_assets, args.input):
        raise ValueError(f"Asset {args.input} not found in conquer3d.data.assets")
    
    asset_class = getattr(c3d_assets, args.input)
    asset = asset_class()
    vertices, faces, _ = asset.get()
    
    vertices = vertices.float()
    faces = faces.long()
    
    # Normalize vertices to [-0.9, 0.9]
    vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
    scale = 1.8 / torch.max(vmax - vmin).item()
    vertices = vertices - (vmax + vmin) / 2
    vertices = vertices * scale
    
    vertices = vertices.to(device)
    faces = faces.to(device)
    gt_mesh = kal.rep.SurfaceMesh(vertices=vertices, faces=faces)

    glctx = dr.RasterizeCudaContext(device=device)

    # Initialize DMTet grid
    num_sampling_points = 100000
    grid_vertices = create_random_points_ball(num_sampling_points, 1.2, device=device)
    tetras = tetrahedralize(grid_vertices)
    
    sdf = torch.rand_like(grid_vertices[:,0]) - 0.1
    sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
    grid_vertices = torch.nn.Parameter(grid_vertices.clone().detach(), requires_grad=True)
    
    # Optimization Setup
    epoch = args.epoch
    batch = 8
    train_res = [2048, 2048]

    def lr_schedule(ep):
        return max(0.0, 10 ** (-(ep) * 0.0002))

    optimizer = torch.optim.Adam([
        {'params': sdf, 'lr': args.lr_sdf}, 
        {'params': grid_vertices, 'lr': args.lr_grid_verts}
    ])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x))

    print(f"Starting DMTet Optimization for {epoch} epochs...")
    best_loss = float('inf')
    checkpoint_path = os.path.splitext(args.output)[0] + '.pt'

    for it in tqdm.tqdm(range(epoch)): 
        optimizer.zero_grad()
        if it % 10 == 0:
            tetras = tetrahedralize(grid_vertices)

        # sample random camera poses
        cameras = render.get_random_camera_batch(batch, iter_res=train_res, device=device)
        
        # render gt mesh at sampled views
        target = render.render_mesh(glctx, gt_mesh, cameras, train_res, return_types = ["mask", "depth", "normals"])

        # mesh extraction
        mc_vertices, mc_faces = marching_tetrahedra(grid_vertices, tetras, sdf)
        if mc_vertices.shape[0] == 0:
            continue
        
        extracted_mesh = kal.rep.SurfaceMesh(vertices=mc_vertices, faces=mc_faces.int())
        buffers = render.render_mesh(glctx, extracted_mesh, cameras, train_res, return_types = ["mask", "depth", "normals"])

        # reconstruction loss
        mask_loss = (buffers['mask'] - target['mask']).abs().mean()
        depth_loss = (((((buffers['depth'] - target['depth']) * target['mask'])**2).sum(-1)+1e-8)).sqrt().mean()
        normal_loss = (((((buffers['normals'] - target['normals']) * target['mask'])**2).sum(-1)+1e-8)).sqrt().mean()
        
        edges = get_edges(tetras)
        reg_sdf_loss = sdf_reg_loss(sdf, edges.reshape(-1, 2)).mean()
        
        total_loss = (args.weight_mask * mask_loss + 
                      args.weight_depth * depth_loss + 
                      args.weight_normal * normal_loss + 
                      args.weight_sdf_reg * reg_sdf_loss)
                      
        if args.use_fairness:
            fairness_loss = triangle_angle_fairness(mc_vertices, mc_faces.long())
            total_loss += args.weight_fairness * fairness_loss
        else:
            fairness_loss = torch.tensor(0.0, device=device)
            
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Check total loss for best checkpoint save for each 10 iterations
        if (it + 1) % 10 == 0:
            current_loss = total_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                checkpoint = {
                    'sdf': sdf.data.clone(),
                    'grid_vertices': grid_vertices.data.clone(),
                    'tetras': tetras.clone() if isinstance(tetras, torch.Tensor) else tetras,
                    'best_loss': best_loss,
                    'iteration': it + 1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }
                checkpoint_dir = os.path.dirname(checkpoint_path)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                tqdm.tqdm.write(f"Iter {it+1}: New best total loss {best_loss:.4f}, saving checkpoint to {checkpoint_path}")
        
        if (it + 1) % 100 == 0:
            tqdm.tqdm.write(f"Epoch {it+1}/{epoch}, Mask: {mask_loss.item():.4f}, Depth: {depth_loss.item():.4f}, Normal: {normal_loss.item():.4f}, Reg SDF: {reg_sdf_loss.item():.4f}, Fairness: {fairness_loss.item():.4f}, Total: {total_loss.item():.4f}")

    print("Optimization finished.")
    if os.path.exists(checkpoint_path):
        print(f"Loading best checkpoint from {checkpoint_path} (best loss: {best_loss:.4f})")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        with torch.no_grad():
            sdf.data.copy_(checkpoint['sdf'])
            grid_vertices.data.copy_(checkpoint['grid_vertices'])
            tetras = checkpoint['tetras'].to(device)

    with torch.no_grad():
        final_vertices, final_faces = marching_tetrahedra(grid_vertices, tetras, sdf)
    
    print(f"Saving mesh to {args.output}...")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    write_obj(args.output, final_vertices, final_faces)
    print("Done!")

if __name__ == '__main__':
    main()
