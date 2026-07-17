import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from conquer3d.data_structure import z_curve_sort

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, patch_size=32):
        super().__init__()
        self.mid_planes = mid_planes = out_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
        assert out_planes % share_planes == 0, "out_planes must be divisible by share_planes"
        self.attn_planes = attn_planes = out_planes // share_planes
        self.patch_size = patch_size

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        # Using LayerNorm instead of BatchNorm since we are working with dense arbitrary shapes (B, P, K, K, C)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, 3),
            nn.LayerNorm(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes)
        )

        self.fc_gamma = nn.Sequential(
            nn.LayerNorm(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, attn_planes),
            nn.LayerNorm(attn_planes),
            nn.ReLU(inplace=True),
            nn.Linear(attn_planes, attn_planes)
        )

    def forward(self, xyz, features):
        """
        Input:
            xyz: (B, N, 3)
            features: (B, N, in_planes)
        Output:
            res: (B, N, out_planes)
        """
        B, N, _ = xyz.shape
        K = self.patch_size
        
        assert N % K == 0, f"Number of points N ({N}) must be divisible by patch_size K ({K})"

        # 1. Sort the entire point cloud using Z-curve
        sorted_xyz, sort_idx, inv_idx = z_curve_sort(xyz)
        
        # Gather sorted features
        batch_indices = torch.arange(B, device=xyz.device).unsqueeze(-1)
        sorted_feat = features[batch_indices, sort_idx]

        # 2. Reshape into Patches (removing the need for KNN)
        # (B, N, C) -> (B, P, K, C)
        xyz_patch = sorted_xyz.view(B, -1, K, 3) 
        feat_patch = sorted_feat.view(B, -1, K, features.shape[-1])

        # 3. Compute Pairwise RPE inside the Patch
        # delta_xyz: (B, P, K, K, 3)
        delta_xyz = xyz_patch.unsqueeze(3) - xyz_patch.unsqueeze(2)
        pos_enc = self.fc_delta(delta_xyz)  # (B, P, K, K, out_planes)

        # 4. Patch-based Attention
        q = self.linear_q(feat_patch) # (B, P, K, mid_planes)
        k = self.linear_k(feat_patch) # (B, P, K, mid_planes)
        v = self.linear_v(feat_patch) # (B, P, K, out_planes)

        # Q - K + pos_enc
        # Q: (B, P, K, 1, mid_planes)
        # K: (B, P, 1, K, mid_planes)
        attn_input = q.unsqueeze(3) - k.unsqueeze(2) + pos_enc
        
        attn = self.fc_gamma(attn_input) # (B, P, K, K, attn_planes)
        attn = F.softmax(attn, dim=-2) # Softmax over the K neighbors in the patch

        # Multiply V and attn
        # We need to add pos_enc: (v.unsqueeze(2) + pos_enc) -> (B, P, K, K, out_planes)
        v_pos = v.unsqueeze(2) + pos_enc
        
        # Rearrange to separate share_planes and attn_planes
        v_pos = einops.rearrange(v_pos, "b p i j (s a) -> b p i j s a", s=self.share_planes)
        
        # Einsum: multiply values by attention probabilities and sum over neighborhood
        res = torch.einsum("b p i j s a, b p i j a -> b p i s a", v_pos, attn)
        res = einops.rearrange(res, "b p i s a -> b p i (s a)") # (B, P, K, out_planes)

        # 5. Flatten Patches and Un-sort
        res = res.view(B, N, self.out_planes)
        res = res[batch_indices, inv_idx]
        
        return res
