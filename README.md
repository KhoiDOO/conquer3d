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

pip install plotly open3d jupyter trimesh point-cloud-utils
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

```bibtex
@article{10.1080/10867651.1997.10487472,
    author = {M\"{o}ller, Tomas},
    title = {A fast triangle-triangle intersection test},
    year = {1997},
    issue_date = {1997},
    publisher = {A. K. Peters, Ltd.},
    address = {USA},
    volume = {2},
    number = {2},
    issn = {1086-7651},
    url = {https://doi.org/10.1080/10867651.1997.10487472},
    doi = {10.1080/10867651.1997.10487472},
    journal = {J. Graph. Tools},
    month = nov,
    pages = {25–30},
    numpages = {6}
}
```

```bibtex
@inproceedings{10.1145/1198555.1198746,
    author = {M\"{o}ller, Tomas and Trumbore, Ben},
    title = {Fast, minimum storage ray/triangle intersection},
    year = {2005},
    isbn = {9781450378338},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/1198555.1198746},
    doi = {10.1145/1198555.1198746},
    pages = {7–es},
    keywords = {base transformation, intersection, ray tracing, ray/triangle-intersection},
    location = {Los Angeles, California},
    series = {SIGGRAPH '05}
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