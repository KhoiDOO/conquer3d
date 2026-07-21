import torch
import torch.nn as nn
from experiments.pc.classification.block import PointTransformerBlock

class PointTransformerCls(nn.Module):
    def __init__(self, depth, in_channels=6, num_classes=10, dim=256, share_planes=8, patch_size=32):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth
        
        # Initial projection from raw features to hidden dimension
        self.proj = nn.Sequential(
            nn.Linear(in_channels, dim, bias=False),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )
        
        # Since we removed TransitionDown (no downsampling), the point cloud size N remains constant.
        # Therefore, we can simply stack 'depth' identical PointTransformerBlocks (like a Vision Transformer).
        layers = []
        for _ in range(depth):
            layers.append(PointTransformerBlock(
                in_planes=dim, 
                planes=dim, 
                share_planes=share_planes, 
                patch_size=patch_size
            ))
            
        self.encoder = nn.Sequential(*layers)
        
        # Classification head (adapted from PTv1, using LayerNorm instead of BatchNorm1d)
        self.cls = nn.Sequential(
            nn.Linear(dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_features=False):
        """
        x: (B, N, in_channels)
        Note: The first 3 channels of x are assumed to be (X, Y, Z) coordinates.
        """
        # Extract coordinates
        xyz = x[..., :3].contiguous()
        
        # Initial feature projection
        features = self.proj(x)
        
        # Pass through the sequence of Point Transformer Blocks
        # Since our block accepts and returns [xyz, features], it perfectly chains in nn.Sequential!
        px = [xyz, features]
        px = self.encoder(px)
        xyz, features = px
        
        # Global Average Pooling over the points (dim=1)
        pooled_features = features.mean(dim=1)
        
        if return_features:
            return pooled_features
            
        # Classification
        res = self.cls(pooled_features)
        
        return res
