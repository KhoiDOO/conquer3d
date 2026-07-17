import torch
import torch.nn as nn
from experiments.pc.classification.attn import PointTransformerLayer

class PointTransformerBlock(nn.Module):
    def __init__(self, in_planes, planes, share_planes=8, patch_size=32):
        super().__init__()
        
        # Linear + LayerNorm + ReLU (Pre-processing)
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.norm1 = nn.LayerNorm(planes)
        
        # Point Transformer Layer (Our optimized Z-curve version)
        self.transformer = PointTransformerLayer(
            in_planes=planes, 
            out_planes=planes, 
            share_planes=share_planes, 
            patch_size=patch_size
        )
        self.norm2 = nn.LayerNorm(planes)
        
        # Linear + LayerNorm (Post-processing)
        self.linear3 = nn.Linear(planes, planes, bias=False)
        self.norm3 = nn.LayerNorm(planes)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Optional shortcut if in_planes != planes (prevents shape mismatch in residual connection)
        if in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Linear(in_planes, planes, bias=False),
                nn.LayerNorm(planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, px):
        """
        Input:
            px: list [xyz, features]
                xyz: (B, N, 3)
                features: (B, N, in_planes)
        Output:
            [xyz, new_features]
        """
        xyz, features = px  
        
        identity = self.shortcut(features)
        
        # Pre-process
        out = self.linear1(features)
        out = self.norm1(out)
        out = self.relu(out)
        
        # Z-curve Point Transformer Attention
        out = self.transformer(xyz, out)
        out = self.norm2(out)
        out = self.relu(out)
        
        # Post-process
        out = self.linear3(out)
        out = self.norm3(out)
        
        # Residual connection
        out += identity
        out = self.relu(out)
        
        return [xyz, out]
