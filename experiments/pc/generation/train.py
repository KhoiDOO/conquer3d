import argparse
import os
import json
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import sys
sys.path.insert(0, "/home/koi/Documents/git/rectified-flow-pytorch")

from conquer3d.data.dataset.digit3d import PointDigit3D
from experiments.pc.generation.transformer import PointTransformer, ClassConditionedPointTransformer
from rectified_flow_pytorch import RectifiedFlow, MeanFlow
from rectified_flow_pytorch.soflow import SoFlow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0, help='0: RectifiedFlow, 1: MeanFlow, 2: SoFlow')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # PointTransformer args
    parser.add_argument('--input_channels', type=int, default=6)
    parser.add_argument('--output_channels', type=int, default=6)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--init_scale', type=float, default=0.25)
    parser.add_argument('--time_token_cond', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--class_cond', action='store_true', help="Use class conditioning")
    parser.add_argument('--class_token_cond', action='store_true', help="Pass class as a token")
    parser.add_argument('--cond_drop_prob', type=float, default=0.15, help="CFG drop probability")
    parser.add_argument('--exp_name', type=str, default="", help="Custom experiment name for the run folder")
    
    args = parser.parse_args()

    print("Initializing Datasets...")
    train_dataset = PointDigit3D(root="~/.conquer3d/", train=True, download=False, cached=True, num_points=args.n_ctx)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )

    print("Initializing Model...")
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
            cond_drop_prob=args.cond_drop_prob,
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
    else:
        raise ValueError("Invalid mode")

    flow_model = flow_model.cuda()

    exp_suffix = "_class_cond" if args.class_cond else ""
    exp_dir = f"{mode_name}{exp_suffix}" if not args.exp_name else args.exp_name
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir)
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    num_epochs = args.epochs
    warmup_epochs = args.warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    print(f"Starting Training Loop for {mode_name}...")
    history = []
    
    for epoch in range(num_epochs):
        flow_model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for points, features, labels in progress_bar:
            # features is [B, 512, 6] -> permute to [B, 6, 512] for transformer
            features = features.cuda(non_blocking=True).permute(0, 2, 1)
            labels = labels.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                if args.class_cond:
                    loss = flow_model(features, cond=labels)
                else:
                    loss = flow_model(features)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        train_loss = total_loss / len(train_loader)
        print(f"==> Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}")
        
        ckpt_path = os.path.join(save_dir, "model.pt")
        torch.save(flow_model.state_dict(), ckpt_path)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
        })
        
        json_path = os.path.join(save_dir, "history.json")
        with open(json_path, "w") as f:
            json.dump(history, f, indent=4)
            
        scheduler.step()
            
    print("Training finished.")

if __name__ == "__main__":
    main()
