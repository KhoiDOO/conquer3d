"""Differentiable Poisson Surface Reconstruction (DPSR).

This module implements GPU-accelerated Differentiable Poisson Surface Reconstruction.
It solves the continuous Poisson indicator equation via the Spectral Fourier Method
using PyTorch's native cuFFT backend, enabling direct gradient backpropagation from
implicit volume grids and surface meshes to input points and normals.

Adapted from Shape As Points (Peng et al., NeurIPS 2021):
https://github.com/autonomousvision/shape_as_points/tree/main

Example:
    >>> import torch
    >>> from conquer3d.ops import dpsr, DPSR
    >>> # Compute indicator / pseudo-SDF field
    >>> phi = dpsr(points, normals, res=128, sig=10.0)
"""

from typing import Tuple, Union, Optional, List
import math
import torch
import torch.nn as nn


def _compute_trilinear_corners(
    points: torch.Tensor,
    res: Tuple[int, int, int]
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Calculates corner coordinate indices and trilinear interpolation weights."""
    rx, ry, rz = res
    device = points.device
    grid_size = torch.tensor([rx, ry, rz], dtype=torch.float32, device=device)
    clamped_pts = torch.clamp(points, 0.0, 0.999999) * grid_size

    idx0 = torch.floor(clamped_pts).long()
    u = clamped_pts - idx0.float()

    i0 = idx0 % grid_size.long()
    i1 = (idx0 + 1) % grid_size.long()

    w0_x, w1_x = 1.0 - u[..., 0:1], u[..., 0:1]
    w0_y, w1_y = 1.0 - u[..., 1:2], u[..., 1:2]
    w0_z, w1_z = 1.0 - u[..., 2:3], u[..., 2:3]

    return [
        (i0[..., 0], i0[..., 1], i0[..., 2], w0_x * w0_y * w0_z),
        (i1[..., 0], i0[..., 1], i0[..., 2], w1_x * w0_y * w0_z),
        (i0[..., 0], i1[..., 1], i0[..., 2], w0_x * w1_y * w0_z),
        (i1[..., 0], i1[..., 1], i0[..., 2], w1_x * w1_y * w0_z),
        (i0[..., 0], i0[..., 1], i1[..., 2], w0_x * w0_y * w1_z),
        (i1[..., 0], i0[..., 1], i1[..., 2], w1_x * w0_y * w1_z),
        (i0[..., 0], i1[..., 1], i1[..., 2], w0_x * w1_y * w1_z),
        (i1[..., 0], i1[..., 1], i1[..., 2], w1_x * w1_y * w1_z),
    ]


def point_rasterize(
    points: torch.Tensor,
    values: torch.Tensor,
    res: Tuple[int, int, int]
) -> torch.Tensor:
    """Trilinearly rasterizes point features onto a regular 3D grid.

    Args:
        points (torch.Tensor): Coordinates tensor of shape `(B, N, 3)` in normalized range `[0, 1]^3`.
        values (torch.Tensor): Feature values tensor of shape `(B, N, C)` (e.g. point normals).
        res (Tuple[int, int, int]): 3D grid resolution `(rx, ry, rz)`.

    Returns:
        torch.Tensor: Rasterized grid tensor of shape `(B, C, rx, ry, rz)`.
    """
    B, N, _ = points.shape
    C = values.shape[-1]
    rx, ry, rz = res
    device = points.device

    grid = torch.zeros((B, C, rx, ry, rz), dtype=values.dtype, device=device)
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)

    for cx, cy, cz, cw in _compute_trilinear_corners(points, res):
        weighted_vals = values * cw  # (B, N, C)
        for ch in range(C):
            grid[:, ch].index_put_((batch_idx, cx, cy, cz), weighted_vals[..., ch], accumulate=True)

    return grid


def grid_interp(
    grid: torch.Tensor,
    points: torch.Tensor
) -> torch.Tensor:
    """Trilinearly interpolates grid values at arbitrary point coordinates.

    Args:
        grid (torch.Tensor): 3D grid tensor of shape `(B, rx, ry, rz)` or `(B, C, rx, ry, rz)`.
        points (torch.Tensor): Query point coordinates of shape `(B, N, 3)` in `[0, 1]^3`.

    Returns:
        torch.Tensor: Interpolated values of shape `(B, N)` (scalar) or `(B, N, C)` (multichannel).
    """
    if grid.ndim == 4:
        grid = grid.unsqueeze(1)  # (B, 1, rx, ry, rz)
    B, C, rx, ry, rz = grid.shape
    device = points.device

    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, points.shape[1])
    out = torch.zeros((B, points.shape[1], C), dtype=grid.dtype, device=device)

    for cx, cy, cz, cw in _compute_trilinear_corners(points, (rx, ry, rz)):
        for ch in range(C):
            corner_val = grid[batch_idx, ch, cx, cy, cz]
            out[..., ch] += corner_val * cw.squeeze(-1)

    return out.squeeze(-1) if out.shape[-1] == 1 else out


def _build_frequencies_and_filter(
    res: Tuple[int, int, int],
    sig: float,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes frequency coordinate grid and spectral Gaussian filter buffer."""
    rx, ry, rz = res

    fx = torch.fft.fftfreq(rx, d=1.0 / rx, device=device, dtype=torch.float32)
    fy = torch.fft.fftfreq(ry, d=1.0 / ry, device=device, dtype=torch.float32)
    fz = torch.fft.rfftfreq(rz, d=1.0 / rz, device=device, dtype=torch.float32)

    wx, wy, wz = torch.meshgrid(fx, fy, fz, indexing="ij")
    omega = torch.stack([wx, wy, wz], dim=0)  # (3, rx, ry, rz//2 + 1)

    dis = torch.sqrt(torch.sum(omega ** 2, dim=0, keepdim=True))  # (1, rx, ry, rz//2 + 1)
    G = torch.exp(-0.5 * ((sig * 2.0 * dis / rx) ** 2))  # (1, rx, ry, rz//2 + 1)

    return omega, G


def _solve_spectral_poisson(
    ras_p: torch.Tensor,
    omega: torch.Tensor,
    G: torch.Tensor,
    pts: torch.Tensor,
    res: Tuple[int, int, int],
    shift: bool = True,
    scale: bool = True
) -> torch.Tensor:
    """Solves continuous Poisson indicator equation in the spectral Fourier domain."""
    # 1. Real-to-Complex 3D FFT
    ras_s = torch.fft.rfftn(ras_p, dim=(-3, -2, -1))

    # 2. Spectral Filter & Divergence
    omega_dev = omega.to(ras_p.device)
    G_dev = G.to(ras_p.device)

    N_filtered = ras_s * G_dev.unsqueeze(0)
    omega_2pi = omega_dev * (2.0 * math.pi)

    DivN = -1j * torch.sum(omega_2pi.unsqueeze(0) * N_filtered, dim=1)
    Lap = -torch.sum(omega_2pi ** 2, dim=0, keepdim=True)

    Phi = DivN / (Lap + 1e-6)
    Phi[:, 0, 0, 0] = 0.0

    # 3. Complex-to-Real Inverse 3D FFT
    phi = torch.fft.irfftn(Phi, s=res, dim=(-3, -2, -1))

    # 4. Zero-Level Calibration
    if shift or scale:
        fv = grid_interp(phi, pts)
        if shift:
            offset = torch.mean(fv, dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            phi = phi - offset
        if scale:
            fv0 = phi[:, 0, 0, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            phi = -phi / (torch.abs(fv0) + 1e-7) * 0.5

    return phi


class DPSR(nn.Module):
    """Differentiable Poisson Surface Reconstruction PyTorch Module.

    Caches Fourier frequency grids and Gaussian low-pass filters in memory/buffers
    for high-speed repeated forward passes and backpropagation.
    """

    def __init__(
        self,
        res: Union[int, Tuple[int, int, int], List[int]] = 128,
        sig: float = 10.0,
        shift: bool = True,
        scale: bool = True
    ) -> None:
        """Initializes the DPSR module.

        Args:
            res (Union[int, Tuple[int, int, int], List[int]], optional): 3D grid resolution.
                Defaults to 128.
            sig (float, optional): Gaussian smoothing filter degree. Defaults to 10.0.
            shift (bool, optional): If True, shifts field to zero at point coordinates.
                Defaults to True.
            scale (bool, optional): If True, normalizes field amplitude. Defaults to True.
        """
        super().__init__()
        if isinstance(res, int):
            self.res = (res, res, res)
        else:
            self.res = tuple(res)
        self.sig = float(sig)
        self.shift = shift
        self.scale = scale

        omega, G = _build_frequencies_and_filter(self.res, self.sig, torch.device("cpu"))
        self.register_buffer("omega", omega)
        self.register_buffer("G", G)

    def forward(
        self,
        points: torch.Tensor,
        normals: torch.Tensor
    ) -> torch.Tensor:
        """Computes the continuous indicator field from points and normals.

        Args:
            points (torch.Tensor): Coordinates tensor of shape `(..., N, 3)` in range `[0, 1]^3`.
            normals (torch.Tensor): Normal vectors tensor of shape `(..., N, 3)`.

        Returns:
            torch.Tensor: Continuous indicator scalar field of shape `(..., rx, ry, rz)`.
        """
        orig_shape = points.shape[:-2]
        pts = points.unsqueeze(0) if points.ndim == 2 else points.view(-1, points.shape[-2], 3)
        nrms = normals.unsqueeze(0) if normals.ndim == 2 else normals.view(-1, normals.shape[-2], 3)

        ras_p = point_rasterize(pts, nrms, self.res)
        phi = _solve_spectral_poisson(ras_p, self.omega, self.G, pts, self.res, self.shift, self.scale)

        if len(orig_shape) == 0:
            return phi.squeeze(0)
        return phi.view(*orig_shape, *self.res)


def dpsr(
    points: torch.Tensor,
    normals: torch.Tensor,
    res: Union[int, Tuple[int, int, int], List[int]] = 128,
    sig: float = 10.0,
    shift: bool = True,
    scale: bool = True,
    grid_min: Optional[Union[List[float], Tuple[float, float, float], torch.Tensor]] = None,
    grid_max: Optional[Union[List[float], Tuple[float, float, float], torch.Tensor]] = None
) -> torch.Tensor:
    """Computes a continuous indicator / pseudo-SDF field via Differentiable Poisson Surface Reconstruction.

    Args:
        points (torch.Tensor): Coordinates tensor of shape `(..., N, 3)` (float32, CUDA/CPU).
        normals (torch.Tensor): Outward normal vectors of shape `(..., N, 3)`.
        res (Union[int, Tuple[int, int, int], List[int]], optional): 3D grid resolution `(rx, ry, rz)`.
            Defaults to 128.
        sig (float, optional): Degree of spectral Gaussian smoothing filter. Defaults to 10.0.
        shift (bool, optional): If True, shifts field to zero at point coordinates. Defaults to True.
        scale (bool, optional): If True, normalizes field amplitude. Defaults to True.
        grid_min (Union[List[float], Tuple[float, float, float], torch.Tensor], optional): Minimum
            bounding box coordinates. If provided, points are normalized from `[grid_min, grid_max]`
            into `[0, 1]^3`. Defaults to None.
        grid_max (Union[List[float], Tuple[float, float, float], torch.Tensor], optional): Maximum
            bounding box coordinates. Defaults to None.

    Returns:
        torch.Tensor: Indicator scalar field of shape `(..., rx, ry, rz)`.

    Example:
        >>> import torch
        >>> from conquer3d.ops import dpsr
        >>> pts = torch.rand(1000, 3, device='cuda')
        >>> nrms = torch.randn(1000, 3, device='cuda')
        >>> phi = dpsr(pts, nrms, res=64)
    """
    if isinstance(res, int):
        res_tuple = (res, res, res)
    else:
        res_tuple = tuple(res)

    pts = points
    if grid_min is not None and grid_max is not None:
        g_min = torch.as_tensor(grid_min, dtype=pts.dtype, device=pts.device)
        g_max = torch.as_tensor(grid_max, dtype=pts.dtype, device=pts.device)
        extent = torch.clamp(g_max - g_min, min=1e-7)
        pts = (pts - g_min) / extent

    orig_shape = pts.shape[:-2]
    pts_b = pts.unsqueeze(0) if pts.ndim == 2 else pts.view(-1, pts.shape[-2], 3)
    nrms_b = normals.unsqueeze(0) if normals.ndim == 2 else normals.view(-1, normals.shape[-2], 3)

    omega, G = _build_frequencies_and_filter(res_tuple, sig, pts_b.device)
    ras_p = point_rasterize(pts_b, nrms_b, res_tuple)
    phi = _solve_spectral_poisson(ras_p, omega, G, pts_b, res_tuple, shift, scale)

    if len(orig_shape) == 0:
        return phi.squeeze(0)
    return phi.view(*orig_shape, *res_tuple)
