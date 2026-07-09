import torch
import torch.nn as nn
from typing import Union, List

import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torchsparse.backbones.modules.blocks import SparseResBlock

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: Union[torch.Tensor, List[torch.Tensor]], deterministic=False, feat_dim=-1):
        self.feat_dim = feat_dim
        self.parameters = parameters
        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, dims=-1):
        if self.deterministic:
            return torch.Tensor([0.]).to(self.mean.device)
        return 0.5 * torch.mean(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
            dim=dims
        )


class SimpleSparseVAE(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=32, latent_channels=16, out_channels=8, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # Encoder (Coarse-to-Fine Downsampling)
        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, hidden_channels, kernel_size=3, stride=1),
            spnn.BatchNorm(hidden_channels),
            spnn.ReLU(True),
            SparseResBlock(hidden_channels, hidden_channels, 3)
        )
        
        self.enc_layers = nn.ModuleList()
        current_channels = hidden_channels
        
        for i in range(num_layers - 1):
            self.enc_layers.append(nn.Sequential(
                spnn.Conv3d(current_channels, current_channels * 2, kernel_size=3, stride=2), # Downsample
                spnn.BatchNorm(current_channels * 2),
                spnn.ReLU(True),
                SparseResBlock(current_channels * 2, current_channels * 2, 3)
            ))
            current_channels *= 2
            
        self.enc_out = spnn.Conv3d(current_channels, 2 * latent_channels, kernel_size=3, stride=1)
        
        # Decoder (Fine-to-Coarse Upsampling)
        self.dec_in = nn.Sequential(
            spnn.Conv3d(latent_channels, current_channels, kernel_size=3, stride=1),
            spnn.BatchNorm(current_channels),
            spnn.ReLU(True),
            SparseResBlock(current_channels, current_channels, 3)
        )
        
        self.dec_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.dec_layers.append(nn.Sequential(
                spnn.Conv3d(current_channels, current_channels // 2, kernel_size=3, stride=2, transposed=True), # Upsample
                spnn.BatchNorm(current_channels // 2),
                spnn.ReLU(True),
                SparseResBlock(current_channels // 2, current_channels // 2, 3)
            ))
            current_channels //= 2
            
        self.dec_out = spnn.Conv3d(current_channels, out_channels, kernel_size=3, stride=1)

    def get_latent_resolution(self, input_resolution):
        """
        Calculate the spatial resolution of the latent grid given the input resolution.
        """
        stride = 2 ** (self.num_layers - 1)
        if isinstance(input_resolution, (list, tuple)):
            return [res // stride for res in input_resolution]
        return input_resolution // stride

    def forward(self, x: SparseTensor):
        # MUST set default coordinate map for transposed convolutions to work!
        x._caches.cmaps.setdefault(x.stride, x.coords)
        
        # 1. Encode
        h = self.stem(x)
        for layer in self.enc_layers:
            h = layer(h)
            
        enc_out = self.enc_out(h)
        
        # 2. VAE Reparameterization
        posterior = DiagonalGaussianDistribution(enc_out.feats, feat_dim=-1)
        z_feats = posterior.sample()
        
        # 3. Create Latent Sparse Tensor (Inherit topology cache for transposed convs)
        z = SparseTensor(coords=enc_out.coords, feats=z_feats, stride=enc_out.stride)
        z._caches = x._caches
        
        # 4. Decode
        h = self.dec_in(z)
        for layer in self.dec_layers:
            h = layer(h)
            
        dec_out = self.dec_out(h)
        
        return dec_out.feats, posterior
