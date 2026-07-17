import torch
from torch.utils.data import DataLoader
from torchsparse import SparseTensor
from conquer3d.data.dataset.digit3d import SparseDigit3D
from conquer3d.data.collate.sparse_tensor import sparse_collate_fn
from sparse_vae import SimpleSparseVAE

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Data
dataset = SparseDigit3D(root="~/.conquer3d/", train=False, download=False, cached=True)
loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=sparse_collate_fn)

batch = next(iter(loader))
batched_coords, batched_sdf, _ = batch
batched_coords = batched_coords.to(device)
batched_sdf = batched_sdf.to(device)

x = SparseTensor(coords=batched_coords, feats=batched_sdf)
print(f"Input Coords Min: {x.coords[:, 1:].min(dim=0)[0].tolist()}")
print(f"Input Coords Max: {x.coords[:, 1:].max(dim=0)[0].tolist()}")
print(f"Input Coords Shape: {x.coords.shape}")

# 2. Load Model
model = SimpleSparseVAE(in_channels=8, hidden_channels=32, latent_channels=16, out_channels=8, num_layers=3).to(device)
model.load_state_dict(torch.load("sparse_reconstruction.pt", map_location=device))
model.eval()
print("Model loaded successfully.")

# 3. Layer-by-Layer Forward Pass (Encoder)
print("=== ENCODER ===")
x._caches.cmaps.setdefault(x.stride, x.coords)

with torch.no_grad():
    h = model.stem(x)
    print(f"[Stem] Coords Min: {h.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {h.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {h.stride}")
    
    for i, layer in enumerate(model.enc_layers):
        h = layer(h)
        print(f"[Enc Layer {i+1} (Downsample)] Coords Min: {h.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {h.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {h.stride}")
        
    enc_out = model.enc_out(h)
    print(f"[Enc Out (Latent)] Coords Min: {enc_out.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {enc_out.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {enc_out.stride}")

# 4. Layer-by-Layer Forward Pass (Decoder)
print("\n=== DECODER ===")
with torch.no_grad():
    from sparse_vae import DiagonalGaussianDistribution
    posterior = DiagonalGaussianDistribution(enc_out.feats, feat_dim=-1)
    z_feats = posterior.mean
    
    h = SparseTensor(coords=enc_out.coords, feats=z_feats, stride=enc_out.stride)
    h._caches = x._caches  # Crucial for decoder to find upsampling maps!
    
    h = model.dec_in(h)
    print(f"[Dec In] Coords Min: {h.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {h.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {h.stride}")
    
    for i, layer in enumerate(model.dec_layers):
        h = layer(h)
        print(f"[Dec Layer {i+1} (Upsample)] Coords Min: {h.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {h.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {h.stride}")
        
    out = model.dec_out(h)
    print(f"[Dec Out] Coords Min: {out.coords[:, 1:].min(dim=0)[0].tolist()}, Max: {out.coords[:, 1:].max(dim=0)[0].tolist()}, Stride: {out.stride}")
