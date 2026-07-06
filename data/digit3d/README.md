# Digit3D (3D MNIST) Dataset Generation

This folder contains the complete pipeline to generate the **Digit3D** dataset, which transforms the classic 2D MNIST image dataset into a suite of lightweight 3D `.obj` meshes and offline-precomputed Sparse Signed Distance Fields (SDF).

## Requirements
Ensure you have installed the required dependencies from the root of the `conquer3d` project, especially `conquer3d` (for GPU-accelerated BVH tree querying and Voxelization) and standard scientific libraries (`numpy`, `torch`, `scipy`, `skimage`).

## Generating the Dataset
To run the complete pipeline end-to-end, simply execute the `run.sh` script:

```bash
cd data/digit3d/
bash run.sh
```

### Pipeline Architecture:
1. **Mesh Construction (`construct.py`)**: Downloads the MNIST dataset via `torchvision`, applies distance transforms, scales to a parabolic spherical thickness along the Z-axis, extracts isosurfaces via Marching Cubes, applies Taubin smoothing, and decimates the geometry to a lightweight target of 500 triangles per mesh. The resulting 70,000 `.obj` files are saved into `src/`.
2. **Dense Archiving**: Zips `src/` into `digit3d.zip`.
3. **Offline Sparse Voxelization (`compute.py`)**: Pushes all 70,000 `.obj` meshes through `conquer3d`'s lightning-fast C++ BVH engine on the GPU to compute Signed Distance Fields. It extracts only the sparse surface geometry narrow-band coordinates (`idx_grids`) and features (`sdf`), compressing them into 70,000 lightweight `.npz` archives inside the `sdf/` folder.
4. **Sparse Archiving**: Zips `sdf/` into `digit3d_sdf.zip`.

## Dataset Usage in PyTorch
Once generated, you can upload `digit3d.zip` and `digit3d_sdf.zip` to your Google Drive to distribute the dataset without relying on Hugging Face.

In the `conquer3d.data.dataset.digit3d` module, there are two PyTorch `Dataset` classes ready to consume these zips natively without requiring manual disk extraction (which prevents file descriptor leaks and inode exhaustion):
- `Digit3D`: Downloads `digit3d.zip` via `gdown` and streams dense `vertices` and `faces`.
- `SparseDigit3D`: Downloads `digit3d_sdf.zip` via `gdown` and streams precomputed sparse `idx_grids` and `sdf` directly into multiprocessing-safe CPU tensors.

## Visualization
To visualize the generated assets interactively, open the `visualization.ipynb` notebook and run the cells. It uses Plotly to render the original meshes from `src/` alongside meshes dynamically reconstructed from the `.npz` sparse voxel archives via GPU marching cubes.
