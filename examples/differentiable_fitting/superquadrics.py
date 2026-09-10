"""Differentiable superquadric abstraction of a surface mesh, following SuperFlex.

Fits a set of $K$ learnable superquadrics to the Stanford Armadillo by gradient descent,
then tessellates the result and writes both the fitted abstraction and the target mesh to
Stanford PLY files.

The objective mirrors the per-object optimisation path of SuperFlex
(`superflex/optimization.py::SuperQOptimization`), minus the tapering and bending
deformations that `conquer3d.primitive.SuperQuadrics` deliberately omits:

$$\\mathcal{L} = -\\log \\mathrm{IoU} + w_{sdf} L_{sdf} + w_{bbox} L_{bbox} + w_{ov} L_{ov}$$

- **IoU** is a soft Jaccard index between the occupancy predicted from the union field and
  ground-truth occupancy sampled from the mesh with `query_points(..., return_occ=True)`.
- **$L_{sdf}$** pulls the union field to zero at points sampled on the mesh surface, through a
  truncated `tanh` and a leaky rectifier that penalises surface points left outside the
  abstraction more heavily than those left inside.
- **$L_{bbox}$** keeps the six axis extremes ("poles") of every primitive inside the target's
  bounding box, which stops a primitive escaping to infinity early in the fit.
- **$L_{ov}$** charges for every point covered by more than one primitive, discouraging
  primitives from stacking on the same piece of geometry.

The $K$ primitives are combined by the softmin union of :func:`compute_sq_union`, and
existence gates that union discretely, exactly as in the reference: it carries no gradient
and changes only through the optional pruning step.

Two things are not taken from SuperFlex, because its optimiser starts from a trained
network's prediction and this script starts from nothing: primitive centres are initialised
by k-means over surface samples, and the target is normalised here rather than in the
optimiser. Both are noted where they occur.

Example:
    $ python superquadrics.py --num_quadrics 24 --steps 800
"""

import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from conquer3d.data.assets import Armadillo
from conquer3d.data_structure import TriangleMesh
from conquer3d.io import write_ply
from conquer3d.ops import chamfer_distance
from conquer3d.primitive import SuperQuadrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Differentiable superquadric fitting to a mesh, following SuperFlex."
    )
    parser.add_argument("--num_quadrics", type=int, default=24,
                        help="Number of superquadrics K in the set (default: 24).")
    parser.add_argument("--steps", type=int, default=800,
                        help="Number of optimization steps (default: 800).")
    parser.add_argument("--n_points_iou", type=int, default=100000,
                        help="Volume points carrying occupancy labels (default: 100000).")
    parser.add_argument("--n_points_surf", type=int, default=4096,
                        help="Points sampled on the target surface (default: 4096).")
    parser.add_argument("--resolution", type=int, default=40,
                        help="Angular resolution of the extracted mesh (default: 40).")
    parser.add_argument("--sign_mode", type=int, default=1,
                        help="Occupancy sign mode; 1 is Fast Winding Number, robust to "
                             "the Armadillo's small defects (default: 1).")
    parser.add_argument("--temperature", type=float, default=1e-3,
                        help="Sharpness of the field-to-occupancy sigmoid (default: 1e-3).")
    parser.add_argument("--truncation", type=float, default=0.05,
                        help="Truncation band of the surface SDF term (default: 0.05).")
    parser.add_argument("--union_tau", type=float, default=0.01,
                        help="Smoothing width of the softmin union (default: 0.01).")
    parser.add_argument("--w_sdf", type=float, default=1.0,
                        help="Weight of the surface SDF term (default: 1.0).")
    parser.add_argument("--w_bbox", type=float, default=1.0,
                        help="Weight of the bounding-box containment term (default: 1.0).")
    parser.add_argument("--w_overlap", type=float, default=1.0,
                        help="Weight of the overlap term (default: 1.0).")
    parser.add_argument("--prune_every", type=int, default=0,
                        help="Prune primitives winning too few surface points every N steps. "
                             "0 disables it, matching the reference default (default: 0).")
    parser.add_argument("--prune_min_points", type=int, default=5,
                        help="A primitive winning at most this many surface points is pruned "
                             "(default: 5).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda when available).")
    return parser.parse_args()


def normalize(vertices: torch.Tensor,
              surface_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centres and rescales geometry the way SuperFlex's `normalize` does.

    The reference normalises inside the optimiser and rewrites the primitive parameters to
    match. Here nothing is fitted yet, so the target is normalised up front instead and the
    primitives are born in the normalised frame. The transform is returned so the fitted
    abstraction can be mapped back.

    Args:
        vertices (torch.Tensor): Mesh vertices of shape `(V, 3)`.
        surface_points (torch.Tensor): Surface samples of shape `(S, 3)` defining the frame.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Normalised vertices, the centre that
        was subtracted, and the scalar scale that was divided out.
    """
    centre = surface_points.mean(dim=0)
    scale = 2.0 * (surface_points - centre).abs().max()
    return (vertices - centre) / scale, centre, scale


def kmeans(points: torch.Tensor, k: int, iters: int = 30) -> torch.Tensor:
    """Runs Lloyd's algorithm to place `k` centres over a point cloud.

    SuperFlex has no equivalent, because its optimiser refines a network's prediction.
    Starting from nothing, superquadric fitting is very sensitive to where the primitives
    begin, and scattering them over the surface is far better than scattering them about the
    origin.

    Args:
        points (torch.Tensor): Point cloud of shape `(N, 3)`.
        k (int): Number of centres.
        iters (int, optional): Lloyd iterations. Defaults to 30.

    Returns:
        torch.Tensor: Centres of shape `(k, 3)`.
    """
    centres = points[torch.randperm(points.shape[0], device=points.device)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(points, centres).argmin(dim=1)
        for j in range(k):
            member = points[assign == j]
            if member.numel():
                centres[j] = member.mean(dim=0)
    return centres


def compute_poles(sq: SuperQuadrics) -> torch.Tensor:
    """Returns the six axis extremes of every primitive in world space.

    These are the points $(\\pm a_1, 0, 0)$, $(0, \\pm a_2, 0)$ and $(0, 0, \\pm a_3)$ of each
    primitive, mapped through its pose. SuperFlex uses them as cheap proxies for a
    primitive's extent when constraining it to the target's bounding box.

    Args:
        sq (SuperQuadrics): The primitive set.

    Returns:
        torch.Tensor: Pole coordinates of shape `(K, 6, 3)`.
    """
    scales = sq.scales
    k = scales.shape[0]
    local = torch.zeros(k, 6, 3, device=scales.device, dtype=scales.dtype)
    for axis in range(3):
        local[:, 2 * axis + 0, axis] = scales[:, axis]
        local[:, 2 * axis + 1, axis] = -scales[:, axis]
    return torch.einsum('kij,kpj->kpi', sq.rotations, local) + sq.translations.unsqueeze(1)


def compute_losses(sq, points, num_iou, gt_occ, bbox_min, bbox_max,
                   args) -> Tuple[torch.Tensor, dict]:
    """Evaluates SuperFlex's per-object objective on the current primitive set.

    Args:
        sq (SuperQuadrics): The primitive set being fitted.
        points (torch.Tensor): Volume points followed by surface points, shape `(M, 3)`.
        num_iou (int): Number of leading volume points in `points`.
        gt_occ (torch.Tensor): Ground-truth occupancy of the volume points, shape `(num_iou,)`.
        bbox_min (torch.Tensor): Lower corner of the target's bounding box, shape `(3,)`.
        bbox_max (torch.Tensor): Upper corner of the target's bounding box, shape `(3,)`.
        args (argparse.Namespace): Parsed configuration supplying weights and temperatures.

    Returns:
        Tuple[torch.Tensor, dict]: The scalar loss, and a dictionary of its reported terms.
    """
    fields, union = sq(points, return_union=True)          # (K, M) and (M,)
    present = sq.mask.unsqueeze(-1)

    # --- soft IoU against ground-truth occupancy -------------------------------------
    pred_occ = torch.sigmoid(-union[:num_iou] / args.temperature)
    target = gt_occ.to(pred_occ.dtype)
    intersection = (pred_occ * target).sum()
    union_area = (pred_occ + target - pred_occ * target).sum()
    iou = (intersection + 1e-6) / (union_area.clamp(min=1.0) + 1e-6)
    loss_iou = -torch.log(iou)

    # --- the union's zero level set should pass through the target surface ----------
    truncated = args.truncation * torch.tanh(union[num_iou:] / args.truncation)
    loss_sdf = 16.0 * F.leaky_relu(truncated).abs().mean()

    # --- every primitive's extent stays inside the target's bounding box -------------
    poles = compute_poles(sq)
    violation = torch.relu(bbox_min - poles) + torch.relu(poles - bbox_max)
    loss_bbox = (violation.norm(dim=-1) * present).sum(dim=1).mean()

    # --- charge for points covered by more than one primitive ------------------------
    masked = fields.masked_fill(~present, float('inf'))
    indicator = torch.sigmoid(-(masked + args.truncation) / args.temperature)
    loss_overlap = 1e-1 * torch.relu(indicator.sum(dim=0) - 1.0).mean()

    loss = (loss_iou + args.w_sdf * loss_sdf
            + args.w_bbox * loss_bbox + args.w_overlap * loss_overlap)
    return loss, {"iou": iou.item(), "sdf": loss_sdf.item(),
                  "bbox": loss_bbox.item(), "overlap": loss_overlap.item()}


@torch.no_grad()
def prune(sq: SuperQuadrics, surface_points: torch.Tensor, min_points: int) -> int:
    """Removes primitives that own too little of the target surface.

    Each surface point is assigned to the primitive whose field is lowest there; a primitive
    winning at most `min_points` of them is contributing nothing and is switched off. This
    is how SuperFlex changes existence: discretely, outside the gradient path.

    Args:
        sq (SuperQuadrics): The primitive set to prune in place.
        surface_points (torch.Tensor): Points on the target surface, shape `(S, 3)`.
        min_points (int): Threshold at or below which a primitive is removed.

    Returns:
        int: Number of primitives switched off by this call.
    """
    present = sq.mask
    fields = sq(surface_points).masked_fill(~present.unsqueeze(-1), float('inf'))
    owner = fields.argmin(dim=0)
    won = torch.bincount(owner, minlength=sq.num_quadrics)
    doomed = present & (won <= min_points)
    if bool(doomed.any()):
        sq.raw_existences[doomed] = -20.0
    return int(doomed.sum())


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print("=== Conquer3D Differentiable Superquadric Fitting ===")
    print(f"Device:             {device}")
    print(f"Primitives (K):     {args.num_quadrics}")
    print(f"Optimization Steps: {args.steps}")
    print(f"Occupancy points:   {args.n_points_iou}")
    print(f"Surface points:     {args.n_points_surf}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_ply_path = os.path.join(script_dir, "sq_pred.ply")
    gt_ply_path = os.path.join(script_dir, "sq_gt.ply")

    # 1. Load and normalise the target -------------------------------------------------
    print("\n[1/5] Loading Armadillo asset...")
    verts, faces, _ = Armadillo().get()
    verts = verts.to(device).float()
    faces = faces.to(device).int()

    raw_mesh = TriangleMesh(verts, faces)
    probe, _, _, _ = raw_mesh.sample_points(args.n_points_surf)
    verts_norm, centre, scale = normalize(verts, probe)
    mesh = TriangleMesh(verts_norm.contiguous(), faces)
    print(f"      {verts.shape[0]} vertices, {faces.shape[0]} faces; "
          f"normalised by centre={centre.tolist()} scale={float(scale):.4f}")

    # 2. Build the supervision --------------------------------------------------------
    print(f"[2/5] Sampling supervision (sign_mode={args.sign_mode})...")
    surface_points, _, _, _ = mesh.sample_points(args.n_points_surf)
    surface_points = surface_points.contiguous()

    bbox_min = verts_norm.min(dim=0).values
    bbox_max = verts_norm.max(dim=0).values
    pad = 0.05 * (bbox_max - bbox_min)
    lo, hi = bbox_min - pad, bbox_max + pad
    points_iou = lo + (hi - lo) * torch.rand(args.n_points_iou, 3, device=device)

    # This is what `return_occ` exists for: occupancy is defined once, in the native
    # layer, as `signed distance < 0`, instead of every caller re-deriving the convention.
    *_, gt_occ = mesh.query_points(points_iou.contiguous(), return_sdf=True,
                                   return_prj_pts=False, sign_mode=args.sign_mode,
                                   return_occ=True)
    occupied = float(gt_occ.float().mean())
    print(f"      {args.n_points_iou} volume points, {occupied * 100:.2f}% occupied; "
          f"{args.n_points_surf} surface points")
    if not 0.0 < occupied < 1.0:
        raise RuntimeError(f"degenerate occupancy ({occupied:.4f}); check sign_mode")

    # 3. Initialise the primitives ----------------------------------------------------
    print(f"[3/5] Initialising {args.num_quadrics} superquadrics by k-means...")
    centres = kmeans(surface_points, args.num_quadrics)
    spread = float(torch.cdist(centres, centres).max()) / (2.0 * args.num_quadrics ** (1 / 3))

    sq = SuperQuadrics(num_quadrics=args.num_quadrics, device=device,
                       init_scale=max(spread, 0.02), init_exponent=1.0,
                       union_tau=args.union_tau)
    with torch.no_grad():
        sq.raw_translations.copy_(centres)

    # Per-parameter learning rates, as in SuperFlex's `get_param_groups`. `raw_existences`
    # is deliberately absent: existence gates the union discretely and takes no gradient.
    lrs = {"raw_scales": 2e-2, "raw_exponents": 1e-2}
    groups = [{"params": [p], "lr": lrs.get(n, 2e-3)}
              for n, p in sq.named_parameters() if n != "raw_existences"]
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    points = torch.cat([points_iou, surface_points], dim=0).contiguous()

    # 4. Fit ---------------------------------------------------------------------------
    print("\n[4/5] Fitting...")
    start = time.time()
    pbar = tqdm(range(args.steps), desc="Optimizing", unit="step")
    terms = {}
    pruned_total = 0
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        loss, terms = compute_losses(sq, points, args.n_points_iou, gt_occ,
                                     lo, hi, args)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if args.prune_every and step and step % args.prune_every == 0:
            pruned_total += prune(sq, surface_points, args.prune_min_points)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "IoU": f"{terms['iou']:.4f}",
                          "sdf": f"{terms['sdf']:.4f}", "ov": f"{terms['overlap']:.4f}"})

    elapsed = time.time() - start
    alive = int(sq.mask.sum())
    print(f"\nFitting completed in {elapsed:.2f}s ({args.steps / elapsed:.1f} steps/s)")
    print(f"Final soft IoU: {terms['iou']:.4f} | surface SDF: {terms['sdf']:.4f} | "
          f"bbox: {terms['bbox']:.4f} | overlap: {terms['overlap']:.4f}")
    print(f"Primitives remaining: {alive}/{args.num_quadrics}"
          + (f" ({pruned_total} pruned)" if pruned_total else ""))

    # 5. Extract, evaluate, export -----------------------------------------------------
    print(f"\n[5/5] Extracting mesh at resolution {args.resolution}...")
    pred_v, pred_f, pred_labels = sq.get_mesh(resolution=args.resolution, return_labels=True)

    palette = torch.rand(args.num_quadrics, 3, device=device) * 0.6 + 0.3
    colors = palette[pred_labels.long()]

    pred_mesh = TriangleMesh(pred_v.contiguous(), pred_f.contiguous())
    pred_samples, _, _, _ = pred_mesh.sample_points(args.n_points_surf)
    cd = chamfer_distance(pred_samples.contiguous(), surface_points, squared=False)
    extent = float((bbox_max - bbox_min).max())
    print(f"      {pred_v.shape[0]} vertices, {pred_f.shape[0]} triangles")
    print(f"      Chamfer to target surface: {float(cd):.6f} "
          f"({100 * float(cd) / extent:.2f}% of extent)")

    write_ply(pred_ply_path, pred_v, pred_f, colors)
    write_ply(gt_ply_path, verts_norm, faces)
    print(f"\nSaved results (both in the normalised frame, so they overlay directly):")
    print(f"  - Fitted abstraction: {pred_ply_path}")
    print(f"  - Target mesh:        {gt_ply_path}")


if __name__ == "__main__":
    main()
