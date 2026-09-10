<div align="center">

# ⚡ Conquer3D

### *High-Performance GPU-Accelerated Differentiable Geometry, Spatial Computing & Neural Rendering Toolbox*

[![Documentation](https://img.shields.io/badge/📖_Documentation-conquer3d-76B900?style=for-the-badge)](https://khoidoo.github.io/conquer3d/)
[![PyPI Version](https://img.shields.io/pypi/v/conquer3d.svg?color=blue&style=for-the-badge)](https://pypi.org/project/conquer3d/)
[![Docker Image](https://img.shields.io/badge/Docker-kohido%2Fconquer3d-2496ED?logo=docker&logoColor=white&style=for-the-badge)](https://hub.docker.com/r/kohido/conquer3d)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia&logoColor=white&style=for-the-badge)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)](https://pytorch.org/)
[![API Coverage](https://img.shields.io/badge/API_docs-100%25_documented-76B900?style=for-the-badge)](https://khoidoo.github.io/conquer3d/api/index.html)

<p align="center">
  <b><a href="https://khoidoo.github.io/conquer3d/">🌐 Website</a></b> •
  <a href="https://khoidoo.github.io/conquer3d/documentation.html">Guide</a> •
  <a href="https://khoidoo.github.io/conquer3d/api/index.html">API Reference</a> •
  <a href="https://khoidoo.github.io/conquer3d/benchmarks.html">Benchmarks</a> •
  <a href="#-qualitative-results">Results</a> •
  <a href="#-installation">Installation</a>
</p>

</div>

---

> [!NOTE]
> The API documentation and the [documentation website](https://khoidoo.github.io/conquer3d/)
> were written with [Claude](https://claude.ai/code). The library itself — every CUDA kernel,
> data structure and operator — is the author's own work.

---

## 🌟 Overview

**Conquer3D** is an ultra-fast, GPU-native computational geometry and differentiable spatial computing library engineered in **PyTorch and CUDA**. Designed from the ground up for 3D computer vision, generative AI, neural surface reconstruction, and differentiable rendering, **Conquer3D** delivers up to **~1.3 Billion faces/second** isosurface extraction, exact CAD sharp crease preservation, and memory-efficient spatial acceleration structures.

Every operator consumes and produces PyTorch tensors **in place** — no host round-trip, no format conversion — so meshing a field is an operation *inside* a training step rather than a preprocessing stage around it.

```bash
pip install -U conquer3d
```

```python
import torch
from conquer3d.data_structure import create_voxel_grid
from conquer3d.ops import dmc

grid_vertices, voxels, _ = create_voxel_grid(
    grid_min=[-1.0] * 3, grid_max=[1.0] * 3, res=[64, 64, 64], device="cuda"
)

sdf = (torch.norm(grid_vertices, dim=-1) - 0.6).requires_grad_(True)
verts, faces = dmc(grid_vertices, voxels, sdf, iso=0.0)

verts.sum().backward()          # gradients flow back into the field
```

---

## 🔬 Qualitative Results

### Isosurface extraction

<img src="docs/assets/img/fig-algorithms.webp" alt="Isosurface extraction" width="100%">

One signed distance field meshed by four different extractors.

### Sharp features

<img src="docs/assets/img/fig-hermite.webp" alt="Sharp features" width="100%">

Exact Hermite data lets the dual methods reconstruct a crease instead of rounding it.

### Grid resolution

<img src="docs/assets/img/fig-resolution.webp" alt="Grid resolution" width="100%">

The same model extracted from 64³ up to 2048³, with the error measured at each step.

### Extraction pipeline

<img src="docs/assets/img/fig-pipeline.webp" alt="Extraction pipeline" width="100%">

Every stage of one extraction, from input mesh to extracted surface.

### Sign modes

<img src="docs/assets/img/fig-sign-modes.webp" alt="Sign modes" width="100%">

One slice through a mesh, signed by each of the six ways of deciding inside.

### Ray queries

<img src="docs/assets/img/fig-meshbvh.webp" alt="Ray queries" width="100%">

Which triangles and which voxels a ray hits, found through the BVH.

---

## ⚡ Benchmarks

*RTX 4090, torch 2.8.0+cu128, CUDA 12.8. Fandisk at $1024^3$ (5.15M active cells).
CUDA events around the operator alone, median of 7 runs after 2 warm-ups.*

| Algorithm | Output | Vertices | Faces | Latency | Throughput |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Dual Marching Cubes** | Triangles | 1,716,384 | 3,432,764 | **2.62 ms** | **1,311M faces/s** |
| DMC (pure quads) | Quads | 1,716,384 | 1,716,382 | 2.51 ms | 684M quads/s |
| MC Asymptotic | Triangles | 1,716,382 | 3,432,760 | 2.71 ms | 1,267M faces/s |
| Dual Contouring | Triangles | 1,716,384 | 3,432,764 | 3.88 ms | 884M faces/s |
| Marching Cubes | Triangles | 1,716,382 | 3,432,760 | 7.90 ms | 434M faces/s |
| Marching Tetrahedra | Triangles | 6,113,918 | 12,227,832 | 44.90 ms | 272M faces/s |

Sign-mode costs, distance-operator throughput, pipeline breakdown and memory scaling are
on the **[benchmarks page](https://khoidoo.github.io/conquer3d/benchmarks.html)**.

---

## 📦 Installation

```bash
pip install -U conquer3d
```

Building from source, the full feature list, worked pipelines and the complete API
reference are on the **[documentation site](https://khoidoo.github.io/conquer3d/documentation.html)**.

---

## 📄 License

Conquer3D is licensed under the [MIT License](LICENSE).

<div align="center">
<sub>Built with CUDA and PyTorch · <a href="https://khoidoo.github.io/conquer3d/">khoidoo.github.io/conquer3d</a></sub>
</div>
