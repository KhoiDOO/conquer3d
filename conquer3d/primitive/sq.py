"""Learnable superquadric primitive sets.

Barr's (1981) superquadric — an implicit solid set by three semi-axes and two shape
exponents — as a differentiable set of $K$ primitives whose parameters are
`torch.nn.Parameter`, fittable by gradient descent. The parameterisation follows the direct
per-object path of SuperFlex (Tavernini et al.) **without its tapering and bending
deformations**. Parameters are stored unconstrained and mapped through smooth activations,
so the semi-axes stay positive and the exponents stay in their stable range with no
projection step.

Note:
    The field is the *radial* inside-outside function, not a metric SDF
    ($\\|\\nabla f\\| \\neq 1$ away from the isotropic case): it suits occupancy and level-set
    work but is not safe input to a sphere tracer. Accuracy floors at $10^{-2}$ at unit
    scale, because :func:`_safe_pow` clamps its result to $[10^{-3}, 5 \\times 10^{2}]$ —
    kept deliberately, since it matches the reference implementation and is what keeps the
    exponentiations stable.

Example:
    >>> import torch
    >>> from conquer3d.primitive import SuperQuadrics
    >>> sq = SuperQuadrics(num_quadrics=8)
    >>> points = torch.rand(1000, 3) - 0.5
    >>> field = sq(points)          # (8, 1000), one row per primitive
    >>> sq.save('shape.pt')
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._C import compute_superquadric_mesh_func

__all__ = ['SuperQuadrics', 'compute_sq_sdf', 'compute_sq_union']

#: Lower bound added to the exponentiated raw semi-axes, keeping every scale strictly positive.
MIN_SCALE = 0.001
#: Lower bound of the shape exponent range; below this the inside-outside function stiffens sharply.
MIN_EXPONENT = 0.1
#: Upper bound of the shape exponent range; above this the surface becomes pinched and unstable.
MAX_EXPONENT = 1.9
#: Default smoothing width of the softmin union; smaller is a sharper, less differentiable union.
DEFAULT_UNION_TAU = 0.01


def _safe_pow(x: torch.Tensor, y: Union[float, torch.Tensor], eps: float = 1e-3) -> torch.Tensor:
    """Raises a tensor to a power with the base and result clamped away from 0 and infinity."""
    return x.abs().clamp(min=eps, max=5e2).pow(y).clamp(min=eps, max=5e2)


def _safe_mul(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Multiplies two tensors with both magnitudes clamped, biasing the result away from zero."""
    x = torch.sign(x) * x.abs().clamp(min=eps, max=1e6)
    y = torch.sign(y) * y.abs().clamp(min=eps, max=1e6)
    return x * y + eps


def _quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """Converts unit quaternions `[w, x, y, z]` into local-to-world rotation matrices."""
    w, x, y, z = quaternions.unbind(-1)
    row0 = torch.stack([1 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)], -1)
    row1 = torch.stack([2.0 * (x * y + w * z), 1 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)], -1)
    row2 = torch.stack([2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1 - 2.0 * (x * x + y * y)], -1)
    return torch.stack([row0, row1, row2], dim=-2)


def compute_sq_sdf(
    points: torch.Tensor,
    scales: torch.Tensor,
    exponents: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    floor: bool = True
) -> torch.Tensor:
    """Evaluates the superquadric inside-outside function of a primitive set at query points.

    Each primitive is expressed in its own frame by $X = R^{T} (p - t)$, and the implicit
    function of Barr's superquadric is

    $$F(X) = \\left[ \\left(\\frac{x}{a_1}\\right)^{2/\\epsilon_2} + \\left(\\frac{y}{a_2}\\right)^{2/\\epsilon_2} \\right]^{\\epsilon_2/\\epsilon_1} + \\left(\\frac{z}{a_3}\\right)^{2/\\epsilon_1}$$

    which equals 1 exactly on the surface. The returned field is the radial form
    $r_0 (1 - F^{-\\epsilon_1/2})$ with $r_0 = \\|X\\|$, negative inside and positive outside.

    Args:
        points (torch.Tensor): Query coordinates of shape `(M, 3)`.
        scales (torch.Tensor): Semi-axes $(a_1, a_2, a_3)$ of shape `(K, 3)`, strictly positive.
        exponents (torch.Tensor): Shape exponents $(\\epsilon_1, \\epsilon_2)$ of shape `(K, 2)`.
        rotations (torch.Tensor): Local-to-world rotations, either unit quaternions
            `[w, x, y, z]` of shape `(K, 4)` or matrices of shape `(K, 3, 3)`.
        translations (torch.Tensor): Primitive centres of shape `(K, 3)`.
        floor (bool, optional): If True, lower-bounds the field by $r_0 - \\|a\\|$ so it keeps
            growing outside the primitive instead of flattening. Defaults to True.

    Returns:
        torch.Tensor: Inside-outside values of shape `(K, M)`, one row per primitive.

    Note:
        Accurate to roughly $10^{-2}$ at unit scale; the numerical clamps described in the
        module docstring set that floor.

    Raises:
        ValueError: If `points` is not `(M, 3)`, if the parameter tensors disagree on `K`,
            or if `rotations` is neither `(K, 4)` nor `(K, 3, 3)`.

    Example:
        >>> import torch
        >>> from conquer3d.primitive import compute_sq_sdf
        >>> scales = torch.tensor([[1.0, 1.0, 1.0]])
        >>> exponents = torch.tensor([[1.0, 1.0]])
        >>> rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        >>> translations = torch.zeros(1, 3)
        >>> pts = torch.tensor([[2.0, 0.0, 0.0]])
        >>> compute_sq_sdf(pts, scales, exponents, rotations, translations).round()
        tensor([[1.]])
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (M, 3), got {tuple(points.shape)}")

    num_quadrics = scales.shape[0]
    for name, tensor, width in (
        ("scales", scales, 3), ("exponents", exponents, 2), ("translations", translations, 3)
    ):
        if tensor.ndim != 2 or tensor.shape[0] != num_quadrics or tensor.shape[1] != width:
            raise ValueError(
                f"{name} must have shape ({num_quadrics}, {width}), got {tuple(tensor.shape)}"
            )

    if rotations.ndim == 2 and rotations.shape[-1] == 4:
        rotmat = _quat_to_mat(F.normalize(rotations, dim=-1, eps=1e-8))
    elif rotations.ndim == 3 and rotations.shape[-2:] == (3, 3):
        rotmat = rotations
    else:
        raise ValueError(
            f"rotations must have shape (K, 4) or (K, 3, 3), got {tuple(rotations.shape)}"
        )
    if rotmat.shape[0] != num_quadrics:
        raise ValueError("rotations must have one entry per primitive")

    points = points.to(scales.dtype)

    # World -> local: X = R^T (p - t), giving (K, M, 3).
    diff = points.unsqueeze(0) - translations.unsqueeze(1)
    local = torch.einsum('kji,kmj->kmi', rotmat, diff)

    # Sign-preserving guard so no coordinate is exactly zero before the powers below.
    local = torch.sign(local) * local.abs().clamp(min=1e-6)

    r0 = local.norm(dim=-1)
    normalised = local.abs() / scales.unsqueeze(1)

    e1 = exponents[:, 0].unsqueeze(-1)
    e2 = exponents[:, 1].unsqueeze(-1)

    term_x = _safe_pow(normalised[..., 0], 2.0 / e2)
    term_y = _safe_pow(normalised[..., 1], 2.0 / e2)
    term_z = _safe_pow(normalised[..., 2], 2.0 / e1)
    implicit = _safe_pow(term_x + term_y, e2 / e1) + term_z

    sdf = _safe_mul(r0, 1.0 - _safe_pow(implicit, -e1 / 2.0))

    if floor:
        radius = scales.norm(dim=-1).unsqueeze(-1)
        sdf = torch.maximum(sdf, r0 - radius)

    return sdf


def compute_sq_union(
    fields: torch.Tensor,
    tau: float = DEFAULT_UNION_TAU,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Combines the per-primitive fields of a superquadric set into a single solid.

    The union of solids is the pointwise minimum of their inside-outside functions. A hard
    minimum routes gradient to one primitive per point, which stalls a fit while primitives
    still overlap heavily, so this uses the smooth minimum

    $$\\mathrm{softmin}_\\tau(f) = -\\tau \\log \\sum_k \\exp(-f_k / \\tau)$$

    which spreads gradient over every primitive that is near-closest, in proportion to how
    close it is. As $\\tau \\to 0$ it converges to the hard minimum.

    Args:
        fields (torch.Tensor): Per-primitive inside-outside values of shape `(K, M)`, as
            returned by :func:`compute_sq_sdf` or :meth:`SuperQuadrics.forward`.
        tau (float, optional): Smoothing width. Pass `0.0` for the exact, non-smooth minimum.
            Defaults to `DEFAULT_UNION_TAU`.
        mask (torch.Tensor, optional): Bool tensor of shape `(K,)` selecting the primitives that
            take part. Excluded primitives contribute nothing and receive no gradient. Defaults
            to None, meaning every primitive participates.

    Returns:
        torch.Tensor: Union field of shape `(M,)`, negative inside the set and positive outside.

    Raises:
        ValueError: If `fields` is not 2-D, if `tau` is negative, or if `mask` has the wrong
            shape or selects no primitive at all.

    Note:
        The softmin is a *lower* bound on the true minimum, short of it by at most
        $\\tau \\log K$ — $0.021$ at the default $\\tau$ with $K = 8$, not negligible against
        unit-cube geometry — so the union surface is slightly inflated. Reduce `tau` if that
        bias matters more than the smoother gradient.

    Example:
        >>> import torch
        >>> from conquer3d.primitive import compute_sq_union
        >>> fields = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        >>> compute_sq_union(fields, tau=0.0)
        tensor([-1., -4.])
    """
    if fields.ndim != 2:
        raise ValueError(f"fields must have shape (K, M), got {tuple(fields.shape)}")
    if tau < 0.0:
        raise ValueError(f"tau must be non-negative, got {tau}")

    if mask is not None:
        if mask.shape != (fields.shape[0],):
            raise ValueError(
                f"mask must have shape ({fields.shape[0]},), got {tuple(mask.shape)}"
            )
        if not bool(mask.any()):
            raise ValueError("mask selects no primitive, so the union is undefined")
        # +inf never wins a minimum and receives zero gradient, which drops the primitive
        # without disturbing the ones that remain.
        fields = fields.masked_fill(~mask.unsqueeze(-1), float('inf'))

    if tau == 0.0:
        return fields.amin(dim=0)
    return -tau * torch.logsumexp(-fields / tau, dim=0)


class SuperQuadrics(nn.Module):
    """A learnable set of $K$ superquadrics in the original, undeformed Barr formulation.

    Parameters are held unconstrained as `raw_*` tensors and mapped to valid geometry by
    smooth activations exposed as properties, so an optimiser can move them freely without
    producing a negative semi-axis or a degenerate exponent. :meth:`forward` reports every
    primitive separately; `return_union=True` also returns the :func:`compute_sq_union`
    softmin of width `union_tau`, which is the field a fitting loss attaches to, while the
    per-primitive rows feed the overlap and extent regularisers.

    Note:
        `existences` gates the union discretely through the bool :attr:`mask` and receives no
        gradient — a primitive is either in the union or absent, never faded. Nothing drives it
        down, so a set fitted without a parsimony term or a pruning step keeps all $K$ alive.

    Attributes:
        raw_scales (torch.nn.Parameter): Unconstrained semi-axes of shape `(K, 3)`.
        raw_exponents (torch.nn.Parameter): Unconstrained shape exponents of shape `(K, 2)`.
        raw_rotations (torch.nn.Parameter): Unnormalised quaternions of shape `(K, 4)`.
        raw_translations (torch.nn.Parameter): Primitive centres of shape `(K, 3)`.
        raw_existences (torch.nn.Parameter): Unconstrained existence logits of shape `(K, 1)`.
    """

    def __init__(
        self,
        num_quadrics: int = 1,
        learnable: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        init_scale: float = 0.2,
        init_exponent: float = 1.0,
        init_existence: float = 0.9,
        init_spread: float = 0.1,
        floor: bool = True,
        existence_threshold: float = 0.5,
        union_tau: float = DEFAULT_UNION_TAU
    ) -> None:
        """Initializes a set of superquadrics scattered around the origin.

        Args:
            num_quadrics (int, optional): Number of primitives $K$ in the set. Defaults to 1.
            learnable (bool, optional): If True, every parameter requires gradient.
                Defaults to True.
            device (Union[str, torch.device], optional): Device to allocate parameters on.
                `.to(device)` works as well. Defaults to None, meaning CPU.
            dtype (torch.dtype, optional): Parameter dtype. Defaults to `torch.float32`.
            init_scale (float, optional): Initial semi-axis length for every primitive.
                Defaults to 0.2.
            init_exponent (float, optional): Initial value of both shape exponents; 1.0 is an
                ellipsoid. Defaults to 1.0.
            init_existence (float, optional): Initial existence probability. Defaults to 0.9.
            init_spread (float, optional): Standard deviation of the random initial centres.
                Defaults to 0.1.
            floor (bool, optional): If True, the field keeps growing outside a primitive rather
                than flattening. Defaults to True.
            existence_threshold (float, optional): Probability above which a primitive counts as
                present. Defaults to 0.5.
            union_tau (float, optional): Smoothing width used by `forward(..., return_union=True)`.
                Pass 0.0 for the exact minimum. Defaults to `DEFAULT_UNION_TAU`.

        Raises:
            ValueError: If `num_quadrics` is not positive, if `init_scale` or `init_exponent`
                falls outside the representable range, or if `union_tau` is negative.
        """
        super().__init__()

        if num_quadrics < 1:
            raise ValueError(f"num_quadrics must be >= 1, got {num_quadrics}")
        if init_scale <= MIN_SCALE:
            raise ValueError(f"init_scale must exceed {MIN_SCALE}, got {init_scale}")
        if not MIN_EXPONENT < init_exponent < MAX_EXPONENT:
            raise ValueError(
                f"init_exponent must lie in ({MIN_EXPONENT}, {MAX_EXPONENT}), got {init_exponent}"
            )
        if not 0.0 < init_existence < 1.0:
            raise ValueError(f"init_existence must lie in (0, 1), got {init_existence}")
        if union_tau < 0.0:
            raise ValueError(f"union_tau must be non-negative, got {union_tau}")

        self.floor = bool(floor)
        self.existence_threshold = float(existence_threshold)
        self.union_tau = float(union_tau)

        factory: Dict[str, Any] = {"device": device, "dtype": dtype}

        raw_scale = torch.log(torch.tensor(init_scale - MIN_SCALE, **factory))
        raw_exponent = torch.logit(
            torch.tensor((init_exponent - MIN_EXPONENT) / (MAX_EXPONENT - MIN_EXPONENT), **factory)
        )
        raw_existence = torch.logit(torch.tensor(init_existence, **factory))

        # Identity quaternion, not zeros: a near-zero quaternion has an arbitrary normalised
        # direction and a gradient scaled by its reciprocal norm.
        quaternions = torch.zeros(num_quadrics, 4, **factory)
        quaternions[:, 0] = 1.0

        self.raw_scales = nn.Parameter(
            raw_scale.expand(num_quadrics, 3).clone(), requires_grad=learnable
        )
        self.raw_exponents = nn.Parameter(
            raw_exponent.expand(num_quadrics, 2).clone(), requires_grad=learnable
        )
        self.raw_rotations = nn.Parameter(quaternions, requires_grad=learnable)
        self.raw_translations = nn.Parameter(
            torch.randn(num_quadrics, 3, **factory) * init_spread, requires_grad=learnable
        )
        self.raw_existences = nn.Parameter(
            raw_existence.expand(num_quadrics, 1).clone(), requires_grad=learnable
        )

    @property
    def num_quadrics(self) -> int:
        """int: Number of primitives $K$ in the set."""
        return self.raw_scales.shape[0]

    @property
    def device(self) -> torch.device:
        """torch.device: Device the parameters currently live on, derived rather than stored."""
        return self.raw_scales.device

    @property
    def scales(self) -> torch.Tensor:
        """torch.Tensor: Semi-axes of shape `(K, 3)`, strictly greater than `MIN_SCALE`."""
        return torch.exp(self.raw_scales) + MIN_SCALE

    @property
    def exponents(self) -> torch.Tensor:
        """torch.Tensor: Shape exponents of shape `(K, 2)` inside `(MIN_EXPONENT, MAX_EXPONENT)`."""
        span = MAX_EXPONENT - MIN_EXPONENT
        return MIN_EXPONENT + span * torch.sigmoid(self.raw_exponents)

    @property
    def quaternions(self) -> torch.Tensor:
        """torch.Tensor: Unit quaternions `[w, x, y, z]` of shape `(K, 4)`."""
        return F.normalize(self.raw_rotations, dim=-1, eps=1e-8)

    @property
    def rotations(self) -> torch.Tensor:
        """torch.Tensor: Local-to-world rotation matrices of shape `(K, 3, 3)`."""
        return _quat_to_mat(self.quaternions)

    @property
    def translations(self) -> torch.Tensor:
        """torch.Tensor: Primitive centres of shape `(K, 3)`."""
        return self.raw_translations

    @property
    def existences(self) -> torch.Tensor:
        """torch.Tensor: Existence probabilities of shape `(K, 1)` in `(0, 1)`.

        Not used by :meth:`forward`, which reports every primitive separately; existence
        applies when a caller combines them.
        """
        return torch.sigmoid(self.raw_existences)

    @property
    def mask(self) -> torch.Tensor:
        """torch.Tensor: Bool tensor of shape `(K,)`, True where existence exceeds the threshold."""
        return self.existences.squeeze(-1) > self.existence_threshold

    def forward(
        self,
        points: torch.Tensor,
        return_union: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Evaluates every primitive's inside-outside function at the query points.

        Args:
            points (torch.Tensor): Query coordinates of shape `(M, 3)`.
            return_union (bool, optional): If True, additionally returns the set combined into a
                single solid by :func:`compute_sq_union`, smoothed with `union_tau` and
                restricted to the primitives selected by :attr:`mask`. This is the field a
                fitting loss attaches to. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - fields (torch.Tensor): Inside-outside values of shape `(K, M)`, negative
                  inside a primitive and positive outside it.
                - [union] (torch.Tensor, optional): Combined field of shape `(M,)`, returned
                  only when `return_union=True`.

        Raises:
            ValueError: If `points` does not have shape `(M, 3)`, or if `return_union` is True
                while :attr:`mask` selects no primitive.

        Note:
            The union applies the existence mask; the per-primitive rows do not. A primitive
            below `existence_threshold` still has a row and still receives gradient from any
            per-primitive term, but contributes nothing to the union.
        """
        fields = compute_sq_sdf(
            points,
            self.scales,
            self.exponents,
            self.quaternions,
            self.translations,
            floor=self.floor
        )
        if return_union:
            return fields, compute_sq_union(fields, tau=self.union_tau, mask=self.mask)
        return fields

    @torch.no_grad()
    def get_mesh(
        self,
        resolution: int = 30,
        return_labels: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Tessellates every present primitive and concatenates the result into one mesh.

        Each primitive is meshed analytically from Barr's parametric form on an
        `(resolution, resolution)` grid of angles spaced by approximately equal arc length,
        placed into world space, then packed into one vertex and triangle array.

        Args:
            resolution (int, optional): Number of angular samples along each axis. Each
                primitive contributes `resolution * (resolution - 2) + 2` vertices and
                `2 * resolution * (resolution - 2)` triangles. Defaults to 30.
            return_labels (bool, optional): If True, additionally returns the index into the
                original `K` primitives that each vertex came from, which survives a change of
                the existence mask. Defaults to False.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                - vertices (torch.Tensor): Float32 coordinates of shape `(V, 3)`.
                - triangles (torch.Tensor): Int32 vertex indices of shape `(F, 3)`.
                - [labels] (torch.Tensor, optional): Int32 originating primitive index of shape
                  `(V,)`, returned only when `return_labels=True`.

        Raises:
            ValueError: If `resolution` is below 3, if :attr:`mask` selects no primitive, or if
                the set is not on a CUDA device.

        Note:
            The result is a **concatenation, not a union**: one closed surface per present
            primitive, self-intersecting wherever two overlap, with Euler characteristic $2K$
            rather than 2. For the boundary of the union solid, evaluate
            :func:`compute_sq_union` on a grid and run an isosurface extractor over it.

        Example:
            >>> import torch
            >>> from conquer3d.primitive import SuperQuadrics
            >>> sq = SuperQuadrics(num_quadrics=2).cuda()
            >>> vertices, triangles = sq.get_mesh(resolution=16)
            >>> vertices.shape[0] == 2 * (16 * 14 + 2)
            True
            >>> triangles.shape[0] == 2 * (2 * 16 * 14)
            True
        """
        if resolution < 3:
            raise ValueError(f"resolution must be at least 3, got {resolution}")

        present = self.mask
        if not bool(present.any()):
            raise ValueError("mask selects no primitive, so the set has no surface to tessellate")
        if not self.raw_scales.is_cuda:
            raise ValueError("get_mesh runs on CUDA; move the set with .to('cuda') first")

        # The native layer knows nothing about existence: it tessellates every primitive it is
        # handed, so the absent ones are dropped here and the labels mapped back afterwards.
        index = present.nonzero().flatten()
        vertices, triangles, labels = compute_superquadric_mesh_func(
            self.scales[index].contiguous().to(torch.float32),
            self.exponents[index].contiguous().to(torch.float32),
            self.rotations[index].contiguous().to(torch.float32),
            self.translations[index].contiguous().to(torch.float32),
            int(resolution),
            bool(return_labels)
        )

        if return_labels:
            return vertices, triangles, index.to(torch.int32)[labels.long()]
        return vertices, triangles

    @torch.no_grad()
    def set_values(
        self,
        scales: Optional[torch.Tensor] = None,
        exponents: Optional[torch.Tensor] = None,
        quaternions: Optional[torch.Tensor] = None,
        translations: Optional[torch.Tensor] = None,
        existences: Optional[torch.Tensor] = None
    ) -> 'SuperQuadrics':
        """Overwrites parameters in place from constrained geometric values.

        Applies the inverse of each activation, so the properties read back the values given
        here. Values sitting exactly on a bound are nudged inside it, because the inverse maps
        diverge there.

        Args:
            scales (torch.Tensor, optional): Semi-axes of shape `(K, 3)`. Defaults to None.
            exponents (torch.Tensor, optional): Shape exponents of shape `(K, 2)`. Defaults to None.
            quaternions (torch.Tensor, optional): Quaternions `[w, x, y, z]` of shape `(K, 4)`.
                Defaults to None.
            translations (torch.Tensor, optional): Centres of shape `(K, 3)`. Defaults to None.
            existences (torch.Tensor, optional): Probabilities of shape `(K, 1)` or `(K,)`.
                Defaults to None.

        Returns:
            SuperQuadrics: This module, to allow chaining.

        Raises:
            ValueError: If any supplied tensor disagrees with the set size `K`.
        """
        span = MAX_EXPONENT - MIN_EXPONENT
        margin = 1e-4

        if scales is not None:
            scales = scales.to(self.raw_scales)
            self._check_shape("scales", scales, (self.num_quadrics, 3))
            self.raw_scales.copy_(torch.log((scales - MIN_SCALE).clamp(min=1e-6)))
        if exponents is not None:
            exponents = exponents.to(self.raw_exponents)
            self._check_shape("exponents", exponents, (self.num_quadrics, 2))
            bounded = exponents.clamp(MIN_EXPONENT + margin, MAX_EXPONENT - margin)
            self.raw_exponents.copy_(torch.logit((bounded - MIN_EXPONENT) / span))
        if quaternions is not None:
            quaternions = quaternions.to(self.raw_rotations)
            self._check_shape("quaternions", quaternions, (self.num_quadrics, 4))
            self.raw_rotations.copy_(F.normalize(quaternions, dim=-1, eps=1e-8))
        if translations is not None:
            translations = translations.to(self.raw_translations)
            self._check_shape("translations", translations, (self.num_quadrics, 3))
            self.raw_translations.copy_(translations)
        if existences is not None:
            existences = existences.to(self.raw_existences).reshape(-1, 1)
            self._check_shape("existences", existences, (self.num_quadrics, 1))
            self.raw_existences.copy_(torch.logit(existences.clamp(margin, 1.0 - margin)))
        return self

    @classmethod
    def from_values(
        cls,
        scales: torch.Tensor,
        exponents: torch.Tensor,
        quaternions: torch.Tensor,
        translations: torch.Tensor,
        existences: Optional[torch.Tensor] = None,
        learnable: bool = True,
        **kwargs: Any
    ) -> 'SuperQuadrics':
        """Builds a set directly from known geometric values.

        Args:
            scales (torch.Tensor): Semi-axes of shape `(K, 3)`.
            exponents (torch.Tensor): Shape exponents of shape `(K, 2)`.
            quaternions (torch.Tensor): Quaternions `[w, x, y, z]` of shape `(K, 4)`.
            translations (torch.Tensor): Centres of shape `(K, 3)`.
            existences (torch.Tensor, optional): Probabilities of shape `(K, 1)` or `(K,)`.
                Defaults to None, leaving the constructor's initial value.
            learnable (bool, optional): If True, parameters require gradient. Defaults to True.
            **kwargs (Any): Further keyword arguments forwarded to the constructor.

        Returns:
            SuperQuadrics: A set reproducing the supplied values.
        """
        module = cls(
            num_quadrics=scales.shape[0],
            learnable=learnable,
            device=scales.device,
            dtype=scales.dtype,
            **kwargs
        )
        return module.set_values(scales, exponents, quaternions, translations, existences)

    def _check_shape(self, name: str, tensor: torch.Tensor, expected: Any) -> None:
        """Raises if a tensor does not match the expected shape."""
        if tuple(tensor.shape) != tuple(expected):
            raise ValueError(
                f"{name} must have shape {tuple(expected)}, got {tuple(tensor.shape)}"
            )

    def save(self, filepath: str) -> None:
        """Writes the set to a PyTorch archive.

        The unconstrained parameters are what is stored, so a reloaded set is bit-identical.
        Constrained values are written alongside for inspection and ignored when loading.

        Args:
            filepath (str): Destination path, conventionally ending in `.pt`.
        """
        payload = {
            'format': 'conquer3d.primitive.SuperQuadrics',
            'version': 1,
            'num_quadrics': int(self.num_quadrics),
            'learnable': bool(self.raw_scales.requires_grad),
            'floor': bool(self.floor),
            'existence_threshold': float(self.existence_threshold),
            'union_tau': float(self.union_tau),
            'state_dict': {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
            'values': {
                'scales': self.scales.detach().cpu(),
                'exponents': self.exponents.detach().cpu(),
                'quaternions': self.quaternions.detach().cpu(),
                'translations': self.translations.detach().cpu(),
                'existences': self.existences.detach().cpu(),
            },
        }
        torch.save(payload, filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        device: Optional[Union[str, torch.device]] = None,
        learnable: Optional[bool] = None
    ) -> 'SuperQuadrics':
        """Reconstructs a set written by :meth:`save`.

        Args:
            filepath (str): Path to a `.pt` archive produced by :meth:`save`.
            device (Union[str, torch.device], optional): Device to place the set on. Defaults to
                None, meaning CPU. The device the file was written from is not used.
            learnable (bool, optional): Overrides whether parameters require gradient. Defaults
                to None, keeping whatever was saved.

        Returns:
            SuperQuadrics: The reconstructed set, with `K` inferred from the stored tensors.

        Raises:
            ValueError: If the archive is not a `SuperQuadrics` payload, if its version is
                unsupported, or if the stored tensors are missing or inconsistent.
        """
        location = device if device is not None else 'cpu'
        payload = torch.load(filepath, map_location=location, weights_only=True)

        if not isinstance(payload, dict) or payload.get('format') != 'conquer3d.primitive.SuperQuadrics':
            raise ValueError(f"{filepath} is not a SuperQuadrics archive")
        if payload.get('version', 0) > 1:
            raise ValueError(
                f"unsupported SuperQuadrics archive version {payload.get('version')}, expected <= 1"
            )

        state = payload.get('state_dict')
        required = ('raw_scales', 'raw_exponents', 'raw_rotations',
                    'raw_translations', 'raw_existences')
        if not isinstance(state, dict) or any(k not in state for k in required):
            raise ValueError(f"{filepath} is missing one or more superquadric parameter tensors")

        num_quadrics = state['raw_scales'].shape[0]
        if any(state[k].shape[0] != num_quadrics for k in required):
            raise ValueError(f"{filepath} stores parameter tensors of inconsistent length")
        if payload.get('num_quadrics', num_quadrics) != num_quadrics:
            raise ValueError(f"{filepath} declares a primitive count its tensors do not match")

        module = cls(
            num_quadrics=num_quadrics,
            learnable=payload.get('learnable', True) if learnable is None else learnable,
            device=device,
            dtype=state['raw_scales'].dtype,
            floor=payload.get('floor', True),
            existence_threshold=payload.get('existence_threshold', 0.5),
            union_tau=payload.get('union_tau', DEFAULT_UNION_TAU)
        )
        module.load_state_dict(state)
        return module

    def extra_repr(self) -> str:
        """Returns the set size and field options shown in the module representation."""
        return (f"num_quadrics={self.num_quadrics}, floor={self.floor}, "
                f"existence_threshold={self.existence_threshold}, "
                f"union_tau={self.union_tau}")
