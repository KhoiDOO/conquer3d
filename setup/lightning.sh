#!/bin/bash
set -e

echo "Setting up Lightning Studio default conda environment for geocutool..."

# 1. Install correct C++ compilers, CUDA toolkit, and Google sparsehash headers
echo "Installing compilers, sparsehash, and CUDA toolkit..."
conda install -c conda-forge gxx_linux-64=13 gcc_linux-64=13 sparsehash -y
conda install nvidia::cuda-toolkit==12.8.2 -y

# 2. Overwrite PyTorch with the specific required version
echo "Installing PyTorch 2.8.0 cu128..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Install build dependencies
echo "Installing build dependencies..."
pip install setuptools wheel ninja pybind11-stubgen

# 4. Install heavy 3D and graphics dependencies
echo "Installing torchsparse, kaolin, and nvdiffrast..."
pip install git+https://github.com/mit-han-lab/torchsparse.git --no-build-isolation
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
pip install git+https://github.com/NVlabs/nvdiffrast.git

# 5. Install visualization and meshing utilities
echo "Installing visualization and meshing utilities..."
pip install plotly open3d jupyter trimesh point-cloud-utils pymeshlab kiui einops
pip install rectified-flow-pytorch

# 6. Install geocutool itself
echo "Installing geocutool..."
export FORCE_CUDA=1
pip install git+https://github.com/KhoiDOO/geocutool.git --no-build-isolation

echo "Lightning Studio setup complete! Environment is ready."
