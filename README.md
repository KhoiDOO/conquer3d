# Conquer3D

# Setup

## Build from source
```bash
git clone https://github.com/KhoiDOO/geocutool.git
pip install pybind11-stubgen 

# then
cd geocutool
pip install -e . --no-build-isolation

# or 
pip install pybind11-stubgen
pip install git+https://github.com/KhoiDOO/geocutool.git --no-build-isolation
```

## To run notebooks in examples
```bash
conda create -c conda-forge -n geocutool python=3.10 gxx_linux-64=13 gcc_linux-64=13 -y
conda activate geocutool

conda install nvidia::cuda-toolkit==12.8.2 -y

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install pybind11-stubgen
pip install git+https://github.com/KhoiDOO/geocutool.git --no-build-isolation

pip install plotly open3d jupyter trimesh
```

# Development
```bash
pip install build twine
rm -rf dist
python -m build --sdist
twine upload dist/* --verbose
```

# Reference

## Research Paper
```bibtex
@inproceedings{2383795.2383801,
author = {Karras, Tero},
title = {Maximizing parallelism in the construction of BVHs, octrees, and k-d trees},
year = {2012},
isbn = {9783905674415},
publisher = {Eurographics Association},
address = {Goslar, DEU},
booktitle = {Proceedings of the Fourth ACM SIGGRAPH / Eurographics Conference on High-Performance Graphics},
pages = {33–37},
numpages = {5},
location = {Paris, France},
series = {EGGH-HPG'12}
}
```

## Blog Post
- [Thinking Parallel, Part I: Collision Detection on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/)
- [Thinking Parallel, Part II: Tree Traversal on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)
- [Thinking Parallel, Part III: Tree Construction on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)

## Repository
- [cuBQL](https://github.com/NVIDIA/cuBQL)
- [cudaKDTree](https://github.com/ingowald/cudaKDTree)
- [Kaolin](https://github.com/NVIDIAGameWorks/kaolin)
- [Pytorch3D](https://github.com/facebookresearch/pytorch3d)
- [Open3D](https://www.open3d.org/)
- [trimesh](https://github.com/mikedh/trimesh)