import argparse
import os
import torch
import sys

# Append root directory to PYTHONPATH so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.insert(0, "/home/koi/Documents/git/rectified-flow-pytorch")

from experiments.pc.generation.transformer import PointTransformer, ClassConditionedPointTransformer
from rectified_flow_pytorch import RectifiedFlow, MeanFlow
from rectified_flow_pytorch.soflow import SoFlow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0, help='0: RectifiedFlow, 1: MeanFlow, 2: SoFlow')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--steps', type=int, default=64)
    
    # PointTransformer args (must match training!)
    parser.add_argument('--input_channels', type=int, default=6)
    parser.add_argument('--output_channels', type=int, default=6)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--init_scale', type=float, default=0.25)
    parser.add_argument('--time_token_cond', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--class_cond', action='store_true', help="Use class conditioning")
    parser.add_argument('--class_token_cond', action='store_true', help="Pass class as a token")
    parser.add_argument('--exp_name', type=str, default="", help="Custom experiment name to load from")
    args = parser.parse_args()

    if args.class_cond:
        model = ClassConditionedPointTransformer(
            device=torch.device('cuda'),
            dtype=torch.float32,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            n_ctx=args.n_ctx,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            init_scale=args.init_scale,
            time_token_cond=args.time_token_cond,
            use_checkpoint=args.use_checkpoint,
            num_classes=10,
            cond_drop_prob=0.0, # no dropout at inference
            token_cond=args.class_token_cond
        )
    else:
        model = PointTransformer(
            device=torch.device('cuda'),
            dtype=torch.float32,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            n_ctx=args.n_ctx,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            init_scale=args.init_scale,
            time_token_cond=args.time_token_cond,
            use_checkpoint=args.use_checkpoint
        )

    accept_cond = args.class_cond
    if args.mode == 0:
        flow_model = RectifiedFlow(model, time_cond_kwarg='t', predict='flow')
        mode_name = "rectified_flow"
    elif args.mode == 1:
        flow_model = MeanFlow(model)
        mode_name = "mean_flow"
    elif args.mode == 2:
        flow_model = SoFlow(model, accept_cond=accept_cond)
        mode_name = "soflow"
        args.steps = 1  # SoFlow is a 1-step generative model!
    else:
        raise ValueError("Invalid mode")

    flow_model = flow_model.cuda()

    # Resolve paths locally to the runs directory
    exp_suffix = "_class_cond" if args.class_cond else ""
    exp_dir = f"{mode_name}{exp_suffix}" if not args.exp_name else args.exp_name
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir, "model.pt")
    
    print(f"Loading weights from {ckpt_path}...")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Could not find model weights at {ckpt_path}")
        return

    flow_model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
    flow_model.eval()

    print(f"Generating {args.num_samples} samples using {mode_name} with {args.steps} steps...")
    
    with torch.no_grad():
        # The sampler expects the raw spatial dimensions. 
        # Since we permuted input inside training as [B, C, L], our output is generated identically.
        
        if args.class_cond:
            cond = (torch.arange(args.num_samples, device='cuda') % 10).long()
            samples = flow_model.sample(
                batch_size=args.num_samples, 
                data_shape=(args.input_channels, args.n_ctx), 
                steps=args.steps,
                cond=cond
            )
        else:
            samples = flow_model.sample(
                batch_size=args.num_samples, 
                data_shape=(args.input_channels, args.n_ctx), 
                steps=args.steps
            )
    
    # Permute back from [B, C, L] -> [B, L, C] so that points are properly formatted (X, Y, Z, Nx, Ny, Nz)
    samples = samples.permute(0, 2, 1).cpu()

    # 1. Save as raw PyTorch tensor
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir, "samples.pt")
    torch.save(samples, save_path)
    print(f"Saved generated samples to {save_path} (Shape: {samples.shape})")

    # 2. Save as readable .ply point clouds with normals for native macOS visualization
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir, "ply_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    for i in range(args.num_samples):
        if args.class_cond:
            class_label = cond[i].item()
            ply_path = os.path.join(samples_dir, f"sample_{i:03d}_class_{class_label}.ply")
        else:
            ply_path = os.path.join(samples_dir, f"sample_{i:03d}.ply")
            
        with open(ply_path, "w") as f:
            # Write PLY Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(samples[i])}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("end_header\n")
            
            # Write Points and Normals
            for pt in samples[i]:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {pt[3]:.6f} {pt[4]:.6f} {pt[5]:.6f}\n")
                
    print(f"Saved {args.num_samples} individual .ply point cloud files to {samples_dir}")

if __name__ == "__main__":
    main()
