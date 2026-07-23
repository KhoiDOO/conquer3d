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
conda install -c conda-forge sparsehash -y

conda install nvidia::cuda-toolkit==12.8.2 -y

pip install setuptools wheel ninja
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install git+https://github.com/mit-han-lab/torchsparse.git

pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

pip install pybind11-stubgen
pip install git+https://github.com/KhoiDOO/geocutool.git --no-build-isolation

pip install plotly open3d jupyter trimesh point-cloud-utils meshlib

pip install rectified-flow-pytorch
```

# Development
```bash
pip install build twine
rm -rf dist
python -m build --sdist
twine upload dist/* --verbose
```

# Reference

## Book


## Research Paper
```bibtex
@inproceedings{2383795.2383801,
    author = {Karras, Tero},
    title = {Maximizing parallelism in the construction of BVHs, octrees, and k-d trees},
    year = {2012},
    booktitle = {Proceedings of the Fourth ACM SIGGRAPH / Eurographics Conference on High-Performance Graphics},
}
```

```bibtex
@article{10.1080/10867651.1997.10487472,
    author = {M\"{o}ller, Tomas},
    title = {A fast triangle-triangle intersection test},
    year = {1997},
    journal = {J. Graph. Tools},
}
```

```bibtex
@inproceedings{10.1145/1198555.1198746,
    author = {M\"{o}ller, Tomas and Trumbore, Ben},
    title = {Fast, minimum storage ray/triangle intersection},
    year = {2005},
    booktitle = {ACM SIGGRAPH 2005 Courses},
}
```

```bibtex
@inproceedings{10.1007/978-3-662-05105-4_2,
    author="Meyer, Mark and Desbrun, Mathieu and Schr{\"o}der, Peter and Barr, Alan H.",
    title="Discrete Differential-Geometry Operators for Triangulated 2-Manifolds",
    booktitle="Visualization and Mathematics III",
    year="2003"
}
```

```bibtex
@article{9167456,
    author={Khan, Dawar and Plopski, Alexander and Fujimoto, Yuichiro and Kanbara, Masayuki and Jabeen, Gul and Zhang, Yongjie Jessica and Zhang, Xiaopeng and Kato, Hirokazu},
    journal={IEEE Transactions on Visualization and Computer Graphics}, 
    title={Surface Remeshing: A Systematic Literature Review of Methods and Research Directions}, 
    year={2022}
}
```

```bibtex
@article{4487066,
    author={Dietrich, Carlos A. and Scheidegger, Carlos E. and Schreiner, John and Comba, João L.D. and Nedel, Luciana P. and Silva, Cláudio T.},
    journal={IEEE Transactions on Visualization and Computer Graphics}, 
    title={Edge Transformations for Improving Mesh Quality of Marching Cubes}, 
    year={2009}}
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
- [TetWeave](https://github.com/AlexandreBinninger/TetWeave/tree/main)