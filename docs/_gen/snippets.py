"""The API sketch shown beside each showcase figure in the lightbox.

Hand-written, not extracted. The builders in ``docs/_figures/make_figures.py``
are 40-100 lines each, and the two or three calls that actually produce what a
figure shows are buried in rendering and compositing. What a reader wants on
expanding a figure is the shape of the call, not a runnable script, so each
entry is an import plus the calls that matter.

Keyed by figure name. ``content._check_snippets()`` fails the build if these
keys and ``FIGURES`` ever disagree, and ``content.check_snippet_imports()``
fails it if a snippet imports a name the library no longer has.

Every snippet assumes ``mesh`` is a ``TriangleMesh`` on CUDA, and where a field
is needed, that ``grid_vertices``, ``voxels``, ``sdf`` and ``normals`` come from
the grid construction shown in the extraction pipeline figure.
"""

FIGURE_CODE = {
    "fig-pipeline": """from conquer3d.data_structure import (
    create_voxel_grid_from_tmesh,
)
from conquer3d.ops import dmc

lo, hi, res = [-1.0] * 3, [1.0] * 3, [256] * 3

# A narrow band around the surface, not a dense lattice.
gv, vox, nrm = create_voxel_grid_from_tmesh(
    grid_min=lo, grid_max=hi, res=res,
    tmesh=mesh, pad=1, return_normals=True,
)

mesh.build_flood_fill_data(lo, hi, res)
sdf = mesh.query_points(gv, return_sdf=True, sign_mode=3)[-1]

verts, faces = dmc(gv, vox, sdf.contiguous(), iso=0.0)[:2]
""",

    "fig-algorithms": """from conquer3d.ops import (
    compute_hermite_from_mesh, dc, dmc, marching_cubes, mca,
)

# Exact edge crossings, so the dual methods can put a vertex
# on the crease instead of rounding it off.
ep, en = compute_hermite_from_mesh(mesh, gv, vox, sdf)

mc = marching_cubes(gv, vox, sdf, iso=0.0)[:2]
asym = mca(gv, vox, sdf, iso=0.0)[:2]
dual = dc(gv, vox, sdf, grid_normals=nrm, iso=0.0,
          edge_points=ep, edge_normals=en)[:2]
dual_mc = dmc(gv, vox, sdf, iso=0.0,
              edge_points=ep, edge_normals=en)[:2]
""",

    "fig-hermite": """from conquer3d.ops import compute_hermite_from_mesh, dc

mesh.compute_triangle_normals()

# Where each grid edge crosses the surface and the normal
# there, read off the mesh rather than off the field.
ep, en = compute_hermite_from_mesh(mesh, gv, vox, sdf)

rounded = dc(gv, vox, sdf, grid_normals=nrm, iso=0.0)[:2]
sharp = dc(gv, vox, sdf, grid_normals=nrm, iso=0.0,
           edge_points=ep, edge_normals=en)[:2]
""",

    "fig-resolution": """from conquer3d.data_structure import (
    create_voxel_grid_from_tmesh,
)
from conquer3d.ops import chamfer_distance, dmc

reference = mesh.sample_points(200_000)[0].contiguous()

for res in (64, 128, 256, 512, 1024, 2048):
    gv, vox, _ = create_voxel_grid_from_tmesh(
        grid_min=[-1.0] * 3, grid_max=[1.0] * 3,
        res=[res] * 3, tmesh=mesh, pad=1,
    )
    sdf = mesh.query_points(gv, return_sdf=True,
                            sign_mode=3)[-1]
    verts, faces = dmc(gv, vox, sdf.contiguous(),
                       iso=0.0)[:2]
    error = chamfer_distance(samples, reference,
                             squared=False)
""",

    "fig-sign-modes": """# 0 ray parity      1 winding number    2 pseudonormal
# 3 flood fill      4 hybrid consensus  5 coarse-fine
# 6 band fill
lo, hi, res = [-1.0] * 3, [1.0] * 3, [512] * 3

# Each lattice mode needs its own structure built first.
mesh.build_flood_fill_data(lo, hi, res)       # mode 3
mesh.build_flood_fill_cf_data(lo, hi, res)    # mode 5
mesh.build_flood_fill_band_data(              # mode 6
    lo, hi, res, dilation_radius=2, cavity_max_voxels=125)

for mode in range(7):
    sdf = mesh.query_points(points, return_sdf=True,
                            sign_mode=mode)[-1]
""",

    "fig-meshbvh": """from conquer3d._C import MeshBVH

# The BVH indexes triangle boxes, so build it from those.
tv = verts[faces.long()]
bvh = MeshBVH(tv.min(1).values.contiguous(),
              tv.max(1).values.contiguous())

_, hit_tris, hit_points, hit_dist = bvh.get_ray_intersection(
    ray_origins, ray_dirs, verts, faces, True)
""",

    "fig-normals": """from conquer3d.ops import (
    dc, dmc, marching_cubes, marching_tetrahedra_grid, mca,
)

mesh.fix_normals()            # consistent winding first
mesh.compute_triangle_normals()

extracted = {
    "Marching Cubes": marching_cubes(gv, vox, sdf, iso=0.0),
    "MC Asymptotic": mca(gv, vox, sdf, iso=0.0),
    "Dual Contouring": dc(gv, vox, sdf, grid_normals=nrm,
                          iso=0.0),
    "Dual Marching Cubes": dmc(gv, vox, sdf, iso=0.0),
}
""",

    "fig-zcurve": """from conquer3d.data_structure import z_curve_sort

points = mesh.sample_points(100_000)[0].contiguous()

# Morton order: points close in space end up close in
# memory, which is what makes later tree builds coherent.
ordered = z_curve_sort(points)
""",

    "fig-curvature": """# Mean curvature in each of its three modes.
for mode in (0, 1, 2):
    H = mesh.get_mean_curvature(mode)

K = mesh.get_gaussian_curvature()
k1, k2 = mesh.get_principal_curvatures()
""",

    "fig-normal-modes": """from conquer3d.data_structure import (
    create_voxel_grid_from_tmesh,
)
from conquer3d.ops import dc

# 0 area-weighted  1 angle-weighted  2 from the field
for normal_mode in (0, 1, 2):
    gv, vox, nrm = create_voxel_grid_from_tmesh(
        grid_min=[-1.0] * 3, grid_max=[1.0] * 3,
        res=[128] * 3, tmesh=mesh, pad=1,
        return_normals=True, normal_mode=normal_mode,
    )
    sdf = mesh.query_points(gv, return_sdf=True,
                            sign_mode=3)[-1]
    verts, faces = dc(gv, vox, sdf.contiguous(),
                      grid_normals=nrm, iso=0.0)[:2]
""",

    "fig-sdf-slices": """import torch

# A plane of query points cut through the model.
res, z = 512, -0.08
xs = torch.linspace(-1.0, 1.0, res, device="cuda")
xx, yy = torch.meshgrid(xs, xs, indexing="xy")
points = torch.stack([
    xx.reshape(-1), yy.reshape(-1),
    torch.full((res * res,), z, device="cuda"),
], dim=-1).contiguous()

mesh.build_flood_fill_data([-1.0] * 3, [1.0] * 3, [512] * 3)
sdf = mesh.query_points(points, return_sdf=True,
                        sign_mode=3)[-1]
""",

    "fig-smoothing": """from conquer3d.data_structure import TriangleMesh

base_v = mesh.vertices.clone()
faces = mesh.triangles.int().clone()

for iterations in (10, 50, 100):
    # smooth() is in place, so each level starts fresh.
    level = TriangleMesh(base_v.clone().contiguous(),
                         faces.clone().contiguous())
    level.smooth(iterations=iterations, damping=0.5,
                 mode=1)          # 1 = cotangent weights

    H = level.get_mean_curvature(0)
""",

    "fig-quality": """# The maps the metrics are computed from.
mesh.compute_triangle_areas()
mesh.compute_vertices_to_triangle_map()
mesh.compute_edges_to_triangle_map()

aspect = mesh.get_aspect_ratio(0)
radii = mesh.get_radii_ratio()
radius_edge = mesh.get_radius_edge_ratio()
regularity = mesh.get_triangle_regularity()
deviation = mesh.get_angle_deviation()

q_min, q_mean = mesh.get_quality()
""",

    "fig-kdtree": """from conquer3d._C import KDTree

# The tree indexes points, never the triangles.
points = mesh.sample_points(24_000)[0].contiguous()
tree = KDTree(points)

# exclude_self drops the query point itself when the
# queries come from the cloud being searched.
distances, indices = tree.query(queries, k=12,
                                exclude_self=True)
""",

    "fig-fix-normals": """from conquer3d.data_structure import TriangleMesh

# Half the windings reversed at random.
broken = faces.clone()
flip = torch.rand(broken.shape[0], device=DEV) < 0.5
broken[flip] = broken[flip][:, [0, 2, 1]]

damaged = TriangleMesh(verts, broken.int().contiguous())
damaged.fix_normals()      # recovers the orientation
""",

    "fig-superquadrics": """import torch
from conquer3d.primitive import SuperQuadrics
from conquer3d.primitive.sq import (
    MAX_EXPONENT, MIN_EXPONENT,
)

# e -> 0 is a box, e = 1 an ellipsoid, e -> 2 a star.
steps = 8
axis = torch.linspace(MIN_EXPONENT, MAX_EXPONENT, steps,
                      device="cuda")
count = steps * steps

sq = SuperQuadrics.from_values(
    scales=torch.ones(count, 3, device="cuda"),
    exponents=torch.stack([axis.repeat_interleave(steps),
                           axis.repeat(steps)], dim=-1),
    quaternions=torch.eye(4, device="cuda")[0].repeat(
        count, 1),
    translations=torch.zeros(count, 3, device="cuda"),
    learnable=False,
)
verts, faces = sq.get_mesh(resolution=64)
""",

    "fig-sqfit": """from conquer3d.data_structure import create_voxel_grid
from conquer3d.ops import marching_cubes
from conquer3d.primitive import (
    SuperQuadrics, compute_sq_union,
)

sq = SuperQuadrics(num_quadrics=2000, device="cuda")

for step in range(600):
    optimizer.zero_grad(set_to_none=True)
    fields, union = sq(points, return_union=True)
    union.abs().mean().backward()
    optimizer.step()

# get_mesh() keeps surface buried inside overlapping
# primitives. The union boundary is the honest one, so
# extract it from the union field instead.
gv, vox, _ = create_voxel_grid(
    grid_min=[-0.62] * 3, grid_max=[0.62] * 3,
    res=[512] * 3, device="cuda")
field = compute_sq_union(sq(gv), tau=sq.union_tau,
                         mask=sq.mask)
verts, faces = marching_cubes(gv, vox, field, iso=0.0)[:2]
""",

    "fig-diffrender": """import torch
from conquer3d.data_structure import create_voxel_grid
from conquer3d.ops import diff_marching_cubes

gv, vox, _ = create_voxel_grid(
    grid_min=[-1.0] * 3, grid_max=[1.0] * 3,
    res=[128] * 3, device="cuda")

# Start from noise; only the field is optimised.
sdf = torch.nn.Parameter(torch.rand_like(gv[:, 0]) - 0.1)
optimizer = torch.optim.Adam([sdf], lr=1e-2)

for step in range(1000):
    optimizer.zero_grad()
    # Differentiable, so mask and depth losses on the
    # render reach the field through the extractor.
    verts, faces = diff_marching_cubes(gv, vox, sdf,
                                       iso=0.0)[:2]
    render_and_compare(verts, faces).backward()
    optimizer.step()
""",
}
