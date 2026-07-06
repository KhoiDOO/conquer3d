#!/bin/bash

set -e

echo "==========================================="
echo "   Digit3D Dataset Generation Pipeline     "
echo "==========================================="

echo "[1/2] Constructing 3D Meshes from MNIST..."
# This generates the .obj files into src/train and src/test
python construct.py --spherical_z --post_process --decimate --target_triangles 500

echo "[2/2] Precomputing Sparse Voxel SDFs..."
# This processes .obj files and creates .npz files into sdf/train and sdf/test
python compute.py

echo "==========================================="
echo "Pipeline Completed Successfully!"
