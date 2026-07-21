import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Append geocutool path explicitly if needed
sys.path.append("/home/koi/Documents/git/geocutool")

# Append local repos
sys.path.insert(0, "/home/koi/Documents/git/rectified-flow-pytorch")

from conquer3d.data.dataset.digit3d import PointDigit3D
from experiments.pc.generation.transformer import PointTransformer, ClassConditionedPointTransformer
from rectified_flow_pytorch import RectifiedFlow, MeanFlow
from rectified_flow_pytorch.soflow import SoFlow

# Import classification model
from experiments.pc.classification.models import PointTransformerCls

# Import FID function from installed package
from pytorch_fid.fid_score import calculate_frechet_distance

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Evaluation Options")
    parser.add_argument('--mode', type=int, default=0, help="0: RectifiedFlow, 1: MeanFlow, 2: SoFlow")
    parser.add_argument('--class_cond', action='store_true', help="Use class conditioning")
    parser.add_argument('--class_token_cond', action='store_true', help="Pass class as a token")
    parser.add_argument('--exp_name', type=str, default="", help="Custom experiment name to load from")
    parser.add_argument('--batch_size', type=int, default=500, help="Batch size for generating samples")
    args = parser.parse_args(sys.argv[1:])

    # 1. Initialize Generator Model
    if args.class_cond:
        model = ClassConditionedPointTransformer(
            num_classes=10,
            input_channels=6,
            output_channels=6,
            n_ctx=512,
            width=256,
            layers=6,
            heads=8,
            cond_drop_prob=0.0,
            device=torch.device('cuda'),
            dtype=torch.float32
        )
    else:
        model = PointTransformer(
            input_channels=6,
            output_channels=6,
            n_ctx=512,
            width=256,
            layers=6,
            heads=8,
            device=torch.device('cuda'),
            dtype=torch.float32
        )

    accept_cond = args.class_cond
    if args.mode == 0:
        flow_model = RectifiedFlow(model, time_cond_kwarg='t', predict='flow')
        mode_name = "rectified_flow"
    elif args.mode == 1:
        flow_model = MeanFlow(model, accept_cond=accept_cond)
        mode_name = "mean_flow"
    elif args.mode == 2:
        flow_model = SoFlow(model, accept_cond=accept_cond)
        mode_name = "soflow"
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    flow_model = flow_model.cuda()

    # Load generator weights
    exp_suffix = "_class_cond" if args.class_cond else ""
    exp_dir = f"{mode_name}{exp_suffix}" if not args.exp_name else args.exp_name
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir, "model.pt")
    
    print(f"Loading generator weights from {ckpt_path}...")
    if not os.path.exists(ckpt_path):
        print(f"Error: Could not find model weights at {ckpt_path}")
        sys.exit(1)
        
    flow_model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
    flow_model.eval()

    # 2. Initialize Classifier Model
    classifier = PointTransformerCls(
        depth=4, 
        in_channels=6, 
        num_classes=10, 
        dim=128, 
        share_planes=8, 
        patch_size=32
    )
    cls_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "classification", "point_classification.pt")
    print(f"Loading classifier weights from {cls_ckpt}...")
    classifier.load_state_dict(torch.load(cls_ckpt, map_location='cuda'))
    classifier.cuda()
    classifier.eval()

    # 3. Load Data
    test_dataset = PointDigit3D(root="~/.conquer3d/", train=False, download=False, cached=True, num_points=512)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    real_features = []
    gen_features = []
    all_labels = []

    print(f"Generating 10K samples and extracting features in batches of {args.batch_size}...")
    for points, features, labels in tqdm(test_loader):
        features = features.cuda()
        labels = labels.cuda().long()
        batch_size = features.shape[0]

        # Extract real features (B, 512)
        real_feat = classifier(features, return_features=True)
        real_features.append(real_feat.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        # Generate fake samples (B, C, N) -> (B, N, C)
        if args.class_cond:
            if not args.class_token_cond:
                fake_samples = flow_model.sample(batch_size=batch_size, cond=labels, data_shape=(6, 512))
            else:
                fake_samples = flow_model.sample(batch_size=batch_size, cond=labels, class_token_cond=True, data_shape=(6, 512))
        else:
            fake_samples = flow_model.sample(batch_size=batch_size, data_shape=(6, 512))
            
        fake_samples = fake_samples.permute(0, 2, 1)

        # Extract fake features (B, 512)
        fake_feat = classifier(fake_samples, return_features=True)
        gen_features.append(fake_feat.cpu().numpy())

    # 4. Compute FID
    real_features = np.concatenate(real_features, axis=0)
    gen_features = np.concatenate(gen_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("Computing FID scores...")
    results = {}
    
    # Global FID
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    global_fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"Global FID: {global_fid:.4f}")
    results["global_fid"] = global_fid
    
    # Class-wise FID
    results["class_wise_fid"] = {}
    class_fids = []
    for c in range(10):
        mask = (all_labels == c)
        if mask.sum() == 0:
            continue
            
        c_real = real_features[mask]
        c_gen = gen_features[mask]
        
        mu_r = np.mean(c_real, axis=0)
        sigma_r = np.cov(c_real, rowvar=False)
        mu_g = np.mean(c_gen, axis=0)
        sigma_g = np.cov(c_gen, rowvar=False)
        
        fid_c = calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
        results["class_wise_fid"][str(c)] = fid_c
        class_fids.append(fid_c)
        print(f"Class {c} FID: {fid_c:.4f}")
        
    if len(class_fids) > 0:
        avg_class_fid = np.mean(class_fids)
        results["avg_class_fid"] = avg_class_fid
        print(f"Average Class-wise FID: {avg_class_fid:.4f}")
    
    # 5. Save results
    fid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", exp_dir, "fid.json")
    with open(fid_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved FID results to {fid_path}")

if __name__ == "__main__":
    main()
