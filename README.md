# Conquer3D

# Installation

## 1. Install from PyPI
```bash
pip install -U conquer3d
```

## 2. Install via Docker (Recommended for complete 3D & CUDA setup)
You can directly pull and run the pre-built Docker image with full GPU and CUDA toolchain support:
```bash
docker pull kohido/conquer3d:latest
docker run --rm --gpus all -it kohido/conquer3d:latest bash
```
Or build the image locally from source:
```bash
docker build -t conquer3d:latest .
docker run --rm --gpus all -it conquer3d:latest bash
```

## 3. Build from Source
To build from source, ensure you have a compatible CUDA toolchain (e.g., CUDA Toolkit 12.8) and PyTorch installed:

```bash
# Optional: Create a dedicated Conda environment with modern C++ compilers and CUDA
conda create -c conda-forge -n geocutool python=3.10 gxx_linux-64=13 gcc_linux-64=13 -y
conda activate geocutool
conda install -c conda-forge sparsehash -y
conda install nvidia::cuda-toolkit==12.8.2 -y

# Install PyTorch and binding generators
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pybind11-stubgen
```

Install directly from GitHub without build isolation:
```bash
pip install git+https://github.com/KhoiDOO/geocutool.git --no-build-isolation
```

Or clone the repository for local development in editable mode:
```bash
git clone https://github.com/KhoiDOO/geocutool.git
cd geocutool
pip install -e . --no-build-isolation
```

# Acknowledgements & References
For further theoretical background, GPU collision detection guides, and related open-source projects, please refer to:
- **[Research Papers](acknowledgement/REFERENCE.md)**: Key computational geometry, differential topology, and acceleration structure literature.
- **[Blog Posts](acknowledgement/BLOG_POST.md)**: Articles and guides on NVIDIA GPU spatial traversal and parallel construction.
- **[Related Repositories](acknowledgement/REPOSITORY.md)**: Open-source libraries and frameworks supporting geometric deep learning and processing.