#include <torch/extension.h>
#include "../../data_structure/triangle_mesh.h"
#include "../../ops/flood_fill.h"
#include "../../ops/flood_fill_cf.h"
#include "../../check.h"
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <vector>

namespace py = pybind11;

#include <optional>

TriangleMesh::TriangleMesh(
    const torch::Tensor &in_vertices,
    const torch::Tensor &in_triangles,
    std::optional<torch::Tensor> in_vertex_normals,
    std::optional<torch::Tensor> in_vertex_colors)
{
    CHECK_INPUT(in_vertices);
    CHECK_INPUT(in_triangles);
    TORCH_CHECK(in_vertices.scalar_type() == torch::kFloat32, "vertices must be float32");
    TORCH_CHECK(in_triangles.scalar_type() == torch::kInt32, "triangles must be int32");
    TORCH_CHECK(in_vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(in_triangles.size(1) == 3, "triangles must have shape (M, 3)");

    this->num_triangles = static_cast<uint32_t>(in_triangles.size(0));

    this->vertices = in_vertices;
    this->triangles = in_triangles;

    if (in_vertex_normals.has_value() && in_vertex_normals->defined())
    {
        CHECK_INPUT(*in_vertex_normals);
        TORCH_CHECK(in_vertex_normals->scalar_type() == torch::kFloat32, "vertex_normals must be float32");
        TORCH_CHECK(in_vertex_normals->size(1) == 3, "vertex_normals must have shape (N, 3)");
        this->vertex_normals = in_vertex_normals->clone();
        this->vertex_normals_mode = 0;
    }

    if (in_vertex_colors.has_value() && in_vertex_colors->defined())
    {
        CHECK_INPUT(*in_vertex_colors);
        TORCH_CHECK(in_vertex_colors->scalar_type() == torch::kFloat32, "vertex_colors must be float32");
        TORCH_CHECK(in_vertex_colors->size(1) == 3, "vertex_colors must have shape (N, 3)");
        this->vertex_colors = in_vertex_colors->clone();
    }
}

void TriangleMesh::compute_triangle_normals()
{
    at::cuda::CUDAGuard device_guard(this->vertices.device());
    this->triangle_normals = torch::empty({static_cast<int64_t>(this->num_triangles), 3}, torch::dtype(torch::kFloat32).device(this->vertices.device()));
    triangle_mesh::compute_triangle_normals(
        this->num_triangles,
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float3 *>(this->triangle_normals.data_ptr<float>()));
}

void TriangleMesh::compute_vertex_normals(int mode)
{
    at::cuda::CUDAGuard device_guard(this->vertices.device());
    uint32_t num_vertices = this->vertices.size(0);
    this->vertex_normals = torch::zeros({static_cast<int64_t>(num_vertices), 3}, torch::dtype(torch::kFloat32).device(this->vertices.device()));

    triangle_mesh::compute_vertex_normals(
        num_vertices,
        this->num_triangles,
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<const float3 *>(this->get_triangle_normals().data_ptr<float>()),
        reinterpret_cast<float3 *>(this->vertex_normals.data_ptr<float>()),
        mode);
    
    this->vertex_normals_mode = mode;
}

void TriangleMesh::compute_edge_normals()
{
    at::cuda::CUDAGuard device_guard(this->triangles.device());
    this->edge_normals = torch::empty({static_cast<int64_t>(this->num_triangles) * 3, 3}, torch::dtype(torch::kFloat32).device(this->triangles.device()));
    triangle_mesh::compute_edge_normals(
        this->num_triangles,
        this->triangles,
        reinterpret_cast<const float3 *>(this->get_triangle_normals().data_ptr<float>()),
        reinterpret_cast<float3 *>(this->edge_normals.data_ptr<float>()));
}

torch::Tensor TriangleMesh::get_vertex_normals(int mode)
{
    if (!this->vertex_normals.defined() || !this->vertex_normals_mode.has_value() || this->vertex_normals_mode.value() != mode)
    {
        this->compute_vertex_normals(mode);
    }
    return this->vertex_normals;
}

void TriangleMesh::compute_vertex_degrees()
{
    if (this->vertex_degrees.defined()) return;
    at::cuda::CUDAGuard device_guard(this->vertices.device());

    if (!this->edges.defined()) {
        this->compute_edges_to_triangle_map();
    }

    uint32_t num_vertices = this->vertices.size(0);
    this->vertex_degrees = torch::zeros({static_cast<int64_t>(num_vertices)}, torch::dtype(torch::kInt32).device(this->vertices.device()));

    uint32_t num_edges = this->edges.size(0);
    triangle_mesh::compute_vertex_degree(
        num_edges,
        reinterpret_cast<const int *>(this->edges.data_ptr<int>()),
        reinterpret_cast<int *>(this->vertex_degrees.data_ptr<int>()));
}

torch::Tensor TriangleMesh::get_vertex_degrees()
{
    if (!this->vertex_degrees.defined())
    {
        this->compute_vertex_degrees();
    }
    return this->vertex_degrees;
}

float TriangleMesh::get_valence_567_percentage()
{
    torch::Tensor degrees = this->get_vertex_degrees();
    if (degrees.numel() == 0) return 0.0f;

    torch::Tensor mask = (degrees == 5).logical_or(degrees == 6).logical_or(degrees == 7);
    float count = mask.sum().item<float>();
    return (count / degrees.numel()) * 100.0f;
}

void TriangleMesh::compute_vertex_lb_uniform()
{
    if (this->vertex_lb_uniform.defined()) return;
    at::cuda::CUDAGuard device_guard(this->vertices.device());

    if (!this->edges.defined()) {
        this->compute_edges_to_triangle_map();
    }
    if (!this->vertex_degrees.defined()) {
        this->compute_vertex_degrees();
    }

    uint32_t num_vertices = this->vertices.size(0);
    this->vertex_lb_uniform = torch::zeros({static_cast<int64_t>(num_vertices), 3}, torch::dtype(torch::kFloat32).device(this->vertices.device()));

    uint32_t num_edges = this->edges.size(0);
    triangle_mesh::compute_uniform_laplacian(
        num_vertices,
        num_edges,
        reinterpret_cast<const int *>(this->edges.data_ptr<int>()),
        reinterpret_cast<const int *>(this->vertex_degrees.data_ptr<int>()),
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<float3 *>(this->vertex_lb_uniform.data_ptr<float>()));
}

torch::Tensor TriangleMesh::get_vertex_lb_uniform()
{
    if (!this->vertex_lb_uniform.defined())
    {
        this->compute_vertex_lb_uniform();
    }
    return this->vertex_lb_uniform;
}

torch::Tensor TriangleMesh::compute_laplacian(int mode) {
    at::cuda::CUDAGuard device_guard(this->vertices.device());
    if (mode == 0) {
        return this->get_vertex_lb_uniform();
    } else if (mode == 1) {
        return this->get_vertex_lb_cotangent();
    } else {
        throw std::runtime_error("Unsupported laplacian mode. Use 0 for Uniform, 1 for Cotangent.");
    }
}

torch::Tensor TriangleMesh::get_mean_curvature(bool signed_curvature) {
    // mode 1 is cotangent laplacian
    torch::Tensor lb_cotangent = this->compute_laplacian(1); 
    
    if (!signed_curvature) {
        // Absolute Mean Curvature: || 2 * H * n || / 2.0 = |H|
        return torch::norm(lb_cotangent, 2, 1) / 2.0f;
    } else {
        // Signed Mean Curvature: dot(2 * H * n, n) / 2.0 = H
        torch::Tensor v_normals = this->get_vertex_normals();
        return torch::sum(lb_cotangent * v_normals, 1) / 2.0f;
    }
}

torch::Tensor TriangleMesh::get_principal_curvatures(bool signed_curvature) {
    torch::Tensor H = this->get_mean_curvature(signed_curvature);
    torch::Tensor K = this->get_gaussian_curvature();
    
    // In discrete settings, floating point inaccuracies can rarely cause H^2 < K. 
    // We use torch::relu to clamp negative values to 0 before sqrt to prevent NaNs.
    torch::Tensor delta = torch::relu(H * H - K);
    torch::Tensor sqrt_delta = torch::sqrt(delta);
    
    torch::Tensor k1 = H + sqrt_delta;
    torch::Tensor k2 = H - sqrt_delta;
    
    return torch::stack({k1, k2}, 1);
}

void TriangleMesh::compute_voronoi_areas()
{
    if (this->voronoi_areas.defined()) return;
    at::cuda::CUDAGuard device_guard(this->vertices.device());
    
    uint32_t num_vertices = this->vertices.size(0);
    this->voronoi_areas = torch::zeros({static_cast<int64_t>(num_vertices)}, torch::dtype(torch::kFloat32).device(this->vertices.device()));
    uint32_t num_triangles = this->triangles.size(0);
    
    triangle_mesh::compute_voronoi_areas(
        num_triangles,
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<float *>(this->voronoi_areas.data_ptr<float>()));
}

void TriangleMesh::compute_gaussian_curvature()
{
    if (this->gaussian_curvature.defined()) return;
    at::cuda::CUDAGuard device_guard(this->vertices.device());
    
    if (!this->voronoi_areas.defined()) {
        this->compute_voronoi_areas();
    }
    
    uint32_t num_vertices = this->vertices.size(0);
    this->gaussian_curvature = torch::zeros({static_cast<int64_t>(num_vertices)}, torch::dtype(torch::kFloat32).device(this->vertices.device()));
    
    uint32_t num_triangles = this->triangles.size(0);
    auto vertex_angle_sum = torch::zeros({static_cast<int64_t>(num_vertices)}, torch::dtype(torch::kFloat32).device(this->vertices.device()));
    
    triangle_mesh::compute_gaussian_curvature(
        num_vertices,
        num_triangles,
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const float *>(this->voronoi_areas.data_ptr<float>()),
        reinterpret_cast<float *>(vertex_angle_sum.data_ptr<float>()),
        reinterpret_cast<float *>(this->gaussian_curvature.data_ptr<float>()));
}

torch::Tensor TriangleMesh::get_gaussian_curvature()
{
    if (!this->gaussian_curvature.defined())
    {
        this->compute_gaussian_curvature();
    }
    return this->gaussian_curvature;
}
void TriangleMesh::compute_vertex_lb_cotangent()
{
    if (this->vertex_lb_cotangent.defined()) return;
    at::cuda::CUDAGuard device_guard(this->vertices.device());

    if (!this->voronoi_areas.defined()) {
        this->compute_voronoi_areas();
    }

    uint32_t num_vertices = this->vertices.size(0);
    this->vertex_lb_cotangent = torch::zeros({static_cast<int64_t>(num_vertices), 3}, torch::dtype(torch::kFloat32).device(this->vertices.device()));

    uint32_t num_triangles = this->triangles.size(0);
    triangle_mesh::compute_cotangent_laplacian(
        num_vertices,
        num_triangles,
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<float *>(this->voronoi_areas.data_ptr<float>()),
        reinterpret_cast<float3 *>(this->vertex_lb_cotangent.data_ptr<float>()));
}

torch::Tensor TriangleMesh::get_vertex_lb_cotangent()
{
    if (!this->vertex_lb_cotangent.defined())
    {
        this->compute_vertex_lb_cotangent();
    }
    return this->vertex_lb_cotangent;
}

torch::Tensor TriangleMesh::get_voronoi_areas()
{
    if (!this->voronoi_areas.defined())
    {
        this->compute_voronoi_areas();
    }
    return this->voronoi_areas;
}

torch::Tensor TriangleMesh::get_isolated_vertices() {
    torch::Tensor degrees = this->get_vertex_degrees();
    return torch::nonzero(degrees == 0).squeeze(1);
}

int32_t TriangleMesh::get_num_isolated_vertices() {
    return this->get_isolated_vertices().size(0);
}

void TriangleMesh::remove_isolated_vertices() {
    torch::Tensor degrees = this->get_vertex_degrees();
    torch::Tensor keep_mask = degrees > 0;
    
    int64_t num_kept = keep_mask.sum().item<int64_t>();
    if (num_kept == this->vertices.size(0)) {
        return;
    }
    
    // Create old to new mapping. Unused vertices map to -1.
    torch::Tensor cumsum = torch::cumsum(keep_mask.to(torch::kInt32), 0);
    torch::Tensor old_to_new = (cumsum - 1).masked_fill_(~keep_mask, -1);
    
    // Filter vertices
    this->vertices = this->vertices.index({keep_mask});
    
    // Update triangles
    torch::Tensor flat_tris = this->triangles.to(torch::kInt64).view({-1});
    this->triangles = old_to_new.index_select(0, flat_tris).view({-1, 3}).to(torch::kInt32);
    
    // Invalidate all caches
    this->triangle_areas = torch::Tensor();
    this->triangle_normals = torch::Tensor();
    this->surface_area = torch::Tensor();
    this->bvh.reset();
    this->opt_edge_manifold = std::nullopt;
    this->opt_edge_manifold_w_boundary = std::nullopt;
    this->opt_vertex_manifold = std::nullopt;
    this->opt_self_intersected = std::nullopt;
    this->edges = torch::Tensor();
    this->edge_to_triangle_offsets = torch::Tensor();
    this->edge_to_triangle_counts = torch::Tensor();
    this->edge_to_triangle_indices = torch::Tensor();
    this->vertex_to_triangle_offsets = torch::Tensor();
    this->vertex_to_triangle_counts = torch::Tensor();
    this->vertex_to_triangle_indices = torch::Tensor();
    this->vertex_degrees = torch::Tensor();
    this->vertex_lb_uniform = torch::Tensor();
    this->vertex_lb_cotangent = torch::Tensor();
    this->voronoi_areas = torch::Tensor();
}

void TriangleMesh::compute_triangle_areas()
{
    this->triangle_areas = torch::empty({static_cast<int64_t>(this->num_triangles)}, torch::dtype(torch::kFloat32).device(this->vertices.device()));
    triangle_mesh::compute_triangle_areas(
        this->num_triangles,
        reinterpret_cast<float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float *>(this->triangle_areas.data_ptr<float>()));
}

MeshBVH TriangleMesh::build_bvh()
{
    if (!this->bvh.has_value())
    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(this->vertices.device());
        torch::Tensor aabb_mins = torch::empty({static_cast<int64_t>(this->num_triangles), 3}, options);
        torch::Tensor aabb_maxs = torch::empty({static_cast<int64_t>(this->num_triangles), 3}, options);

        triangle_mesh::compute_triangle_aabbs(
            this->num_triangles,
            reinterpret_cast<float3 *>(this->vertices.data_ptr<float>()),
            reinterpret_cast<int3 *>(this->triangles.data_ptr<int>()),
            reinterpret_cast<float3 *>(aabb_mins.data_ptr<float>()),
            reinterpret_cast<float3 *>(aabb_maxs.data_ptr<float>()));

        this->bvh = MeshBVH(aabb_mins, aabb_maxs);
    }
    return this->bvh.value();
}

void TriangleMesh::build_flood_fill_data(
    std::optional<std::vector<float>> grid_min,
    std::optional<std::vector<float>> grid_max,
    std::optional<std::vector<int64_t>> res,
    int connectivity)
{
    this->build_bvh();
    std::vector<float> min_vals;
    std::vector<float> max_vals;
    if (grid_min.has_value()) {
        min_vals = grid_min.value();
    } else {
        auto v_min = std::get<0>(torch::min(this->vertices, 0));
        v_min = v_min - 0.05f;
        min_vals = {v_min[0].item<float>(), v_min[1].item<float>(), v_min[2].item<float>()};
    }
    if (grid_max.has_value()) {
        max_vals = grid_max.value();
    } else {
        auto v_max = std::get<0>(torch::max(this->vertices, 0));
        v_max = v_max + 0.05f;
        max_vals = {v_max[0].item<float>(), v_max[1].item<float>(), v_max[2].item<float>()};
    }
    std::vector<int64_t> res_vals;
    if (res.has_value()) {
        res_vals = res.value();
    } else {
        res_vals = {128, 128, 128};
    }
    this->flood_fill_mask = ops::compute_flood_fill(
        this->vertices,
        this->triangles,
        this->bvh.value().aabb_mins,
        this->bvh.value().aabb_maxs,
        this->bvh.value().bvh_children,
        this->bvh.value().object_ids,
        min_vals,
        max_vals,
        res_vals,
        connectivity
    );
    this->flood_grid_min = min_vals;
    this->flood_grid_max = max_vals;
    this->flood_grid_res = res_vals;
}

torch::Tensor TriangleMesh::get_flood_fill_mask()
{
    if (!this->flood_fill_mask.has_value()) {
        this->build_flood_fill_data();
    }
    return this->flood_fill_mask.value();
}

std::vector<float> TriangleMesh::get_flood_grid_min()
{
    if (!this->flood_grid_min.has_value()) {
        this->build_flood_fill_data();
    }
    return this->flood_grid_min.value();
}

std::vector<float> TriangleMesh::get_flood_grid_max()
{
    if (!this->flood_grid_max.has_value()) {
        this->build_flood_fill_data();
    }
    return this->flood_grid_max.value();
}

std::vector<int64_t> TriangleMesh::get_flood_grid_res()
{
    if (!this->flood_grid_res.has_value()) {
        this->build_flood_fill_data();
    }
    return this->flood_grid_res.value();
}

void TriangleMesh::build_flood_fill_cf_data(
    std::optional<std::vector<float>> grid_min,
    std::optional<std::vector<float>> grid_max,
    std::optional<std::vector<int64_t>> res,
    std::optional<std::vector<int64_t>> block_size,
    int connectivity)
{
    this->build_bvh();
    std::vector<float> min_vals;
    std::vector<float> max_vals;
    if (grid_min.has_value()) {
        min_vals = grid_min.value();
    } else {
        auto v_min = std::get<0>(torch::min(this->vertices, 0));
        v_min = v_min - 0.05f;
        min_vals = {v_min[0].item<float>(), v_min[1].item<float>(), v_min[2].item<float>()};
    }
    if (grid_max.has_value()) {
        max_vals = grid_max.value();
    } else {
        auto v_max = std::get<0>(torch::max(this->vertices, 0));
        v_max = v_max + 0.05f;
        max_vals = {v_max[0].item<float>(), v_max[1].item<float>(), v_max[2].item<float>()};
    }
    std::vector<int64_t> res_vals;
    if (res.has_value()) {
        res_vals = res.value();
    } else {
        res_vals = {128, 128, 128};
    }
    std::vector<int64_t> bs_vals;
    if (block_size.has_value()) {
        bs_vals = block_size.value();
    }

    auto cf_res = ops::compute_flood_fill_cf(
        this->vertices,
        this->triangles,
        this->bvh.value().aabb_mins,
        this->bvh.value().aabb_maxs,
        this->bvh.value().bvh_children,
        this->bvh.value().object_ids,
        min_vals,
        max_vals,
        res_vals,
        bs_vals,
        connectivity
    );

    this->cf_coarse_mask = cf_res.coarse_mask;
    this->cf_boundary_lookup = cf_res.boundary_block_lookup;
    this->cf_fine_masks = cf_res.fine_boundary_masks;
    this->cf_block_size = cf_res.block_size;
    this->cf_coarse_res = cf_res.coarse_res;
    this->flood_grid_min = min_vals;
    this->flood_grid_max = max_vals;
    this->flood_grid_res = res_vals;
}

torch::Tensor TriangleMesh::get_cf_coarse_mask()
{
    if (!this->cf_coarse_mask.has_value()) {
        this->build_flood_fill_cf_data();
    }
    return this->cf_coarse_mask.value();
}

torch::Tensor TriangleMesh::get_cf_boundary_lookup()
{
    if (!this->cf_boundary_lookup.has_value()) {
        this->build_flood_fill_cf_data();
    }
    return this->cf_boundary_lookup.value();
}

torch::Tensor TriangleMesh::get_cf_fine_masks()
{
    if (!this->cf_fine_masks.has_value()) {
        this->build_flood_fill_cf_data();
    }
    return this->cf_fine_masks.value();
}

std::vector<int64_t> TriangleMesh::get_cf_block_size()
{
    if (!this->cf_block_size.has_value()) {
        this->build_flood_fill_cf_data();
    }
    return this->cf_block_size.value();
}

std::vector<int64_t> TriangleMesh::get_cf_coarse_res()
{
    if (!this->cf_coarse_res.has_value()) {
        this->build_flood_fill_cf_data();
    }
    return this->cf_coarse_res.value();
}

torch::Tensor TriangleMesh::get_self_intersection()
{
    this->build_bvh();
    return this->bvh.value().get_self_intersection(this->vertices, this->triangles);
}

bool TriangleMesh::is_self_intersection()
{
    if (this->opt_self_intersected.has_value()) {
        return this->opt_self_intersected.value();
    }
    this->build_bvh();
    bool self_int = this->bvh.value().is_self_intersection(this->vertices, this->triangles);
    this->opt_self_intersected = self_int;
    return self_int;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TriangleMesh::query_points(
    const torch::Tensor &query_pts,
    bool return_sdf,
    bool return_prj_pts,
    int sign_mode,
    int distance_mode)
{
    if (distance_mode == 0)
    {
        this->build_bvh();
        if (sign_mode == 3 && !this->flood_fill_mask.has_value()) {
            this->build_flood_fill_data();
        } else if (sign_mode == 5 && !this->cf_coarse_mask.has_value()) {
            this->build_flood_fill_cf_data();
        }
        return this->bvh.value().query_point(
            query_pts,
            this->vertices,
            this->triangles,
            return_sdf,
            return_prj_pts,
            sign_mode,
            this->get_triangle_normals(),
            (sign_mode == 2 || sign_mode == 4) ? std::optional<torch::Tensor>(this->get_vertex_normals(1)) : std::nullopt,
            (sign_mode == 2 || sign_mode == 4) ? std::optional<torch::Tensor>(this->get_edge_normals()) : std::nullopt,
            (sign_mode == 3) ? this->flood_fill_mask : std::nullopt,
            (sign_mode == 3 || sign_mode == 5) ? this->flood_grid_min : std::nullopt,
            (sign_mode == 3 || sign_mode == 5) ? this->flood_grid_max : std::nullopt,
            (sign_mode == 3 || sign_mode == 5) ? this->flood_grid_res : std::nullopt,
            (sign_mode == 5) ? this->cf_coarse_mask : std::nullopt,
            (sign_mode == 5) ? this->cf_boundary_lookup : std::nullopt,
            (sign_mode == 5) ? this->cf_fine_masks : std::nullopt,
            (sign_mode == 5) ? this->cf_block_size : std::nullopt,
            (sign_mode == 5) ? this->cf_coarse_res : std::nullopt);
    }
    else
    {
        throw std::runtime_error("distance_mode != 0 not implemented yet");
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TriangleMesh::get_ray_intersection(
    const torch::Tensor &ray_origins,
    const torch::Tensor &ray_dirs,
    bool return_distance)
{
    this->build_bvh();
    return this->bvh.value().get_ray_intersection(ray_origins, ray_dirs, this->vertices, this->triangles, return_distance);
}

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> TriangleMesh::sample_points(
    int num_points, bool uniform, bool return_normals, bool return_colors, bool use_triangle_normal)
{
    if (this->num_triangles == 0)
    {
        throw std::runtime_error("Cannot sample points from an empty mesh.");
    }

    torch::Tensor tri_indices;
    if (!uniform)
    {
        tri_indices = torch::randint(0, this->num_triangles, {num_points}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kInt64));
    }
    else
    {
        torch::Tensor areas = this->get_triangle_areas();
        tri_indices = torch::multinomial(areas, num_points, true);
    }

    torch::Tensor r1_r2 = torch::rand({num_points, 2}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));
    torch::Tensor out_points = torch::empty({num_points, 3}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));

    torch::Tensor out_normals, out_colors;
    const float3 *d_vertex_normals = nullptr;
    const float3 *d_triangle_normals = nullptr;
    const float3 *d_vertex_colors = nullptr;
    float3 *d_out_normals = nullptr;
    float3 *d_out_colors = nullptr;

    if (return_normals)
    {
        out_normals = torch::empty({num_points, 3}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));
        d_out_normals = reinterpret_cast<float3 *>(out_normals.data_ptr<float>());
        if (use_triangle_normal)
        {
            d_triangle_normals = reinterpret_cast<const float3 *>(this->get_triangle_normals().data_ptr<float>());
        }
        else
        {
            d_vertex_normals = reinterpret_cast<const float3 *>(this->get_vertex_normals().data_ptr<float>());
        }
    }

    if (return_colors)
    {
        if (!this->vertex_colors.defined())
        {
            throw std::runtime_error("Cannot sample colors because vertex_colors is not defined.");
        }
        out_colors = torch::empty({num_points, 3}, torch::TensorOptions().device(this->vertices.device()).dtype(torch::kFloat32));
        d_out_colors = reinterpret_cast<float3 *>(out_colors.data_ptr<float>());
        d_vertex_colors = reinterpret_cast<const float3 *>(this->vertex_colors.data_ptr<float>());
    }

    triangle_mesh::sample_points_triangle_mesh(
        num_points,
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<const int64_t *>(tri_indices.data_ptr<int64_t>()),
        reinterpret_cast<const float2 *>(r1_r2.data_ptr<float>()),
        d_vertex_normals,
        d_triangle_normals,
        d_vertex_colors,
        reinterpret_cast<float3 *>(out_points.data_ptr<float>()),
        d_out_normals,
        d_out_colors);

    std::optional<torch::Tensor> opt_normals = return_normals ? std::make_optional(out_normals) : std::nullopt;
    std::optional<torch::Tensor> opt_colors = return_colors ? std::make_optional(out_colors) : std::nullopt;

    return std::make_tuple(out_points, tri_indices, opt_normals, opt_colors);
}

torch::Tensor TriangleMesh::get_triangle_areas()
{
    if (!this->triangle_areas.defined())
    {
        this->compute_triangle_areas();
    }
    return this->triangle_areas;
}

torch::Tensor TriangleMesh::get_triangle_normals()
{
    if (!this->triangle_normals.defined())
    {
        this->compute_triangle_normals();
    }
    return this->triangle_normals;
}

torch::Tensor TriangleMesh::get_edge_normals()
{
    if (!this->edge_normals.defined())
    {
        this->compute_edge_normals();
    }
    return this->edge_normals;
}

torch::Tensor TriangleMesh::get_surface_area()
{
    if (!this->surface_area.defined())
    {
        this->surface_area = this->get_triangle_areas().sum();
    }
    return this->surface_area;
}

std::tuple<float, float> TriangleMesh::get_quality() {
    if (this->num_triangles == 0) return std::make_tuple(0.0f, 0.0f);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(this->vertices.device());
    torch::Tensor qualities = torch::empty({static_cast<int64_t>(this->num_triangles)}, options);

    triangle_mesh::compute_quality(
        this->num_triangles,
        reinterpret_cast<const float3*>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3*>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float*>(qualities.data_ptr<float>())
    );
    
    float min_q = qualities.min().item<float>();
    float avg_q = qualities.mean().item<float>();
    
    return std::make_tuple(min_q, avg_q);
}

torch::Tensor TriangleMesh::get_aspect_ratio(int mode) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(this->vertices.device());
    torch::Tensor ar = torch::empty({static_cast<int64_t>(this->num_triangles)}, options);
    
    if (this->num_triangles == 0) return ar;

    triangle_mesh::compute_aspect_ratio(
        this->num_triangles,
        reinterpret_cast<const float3*>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3*>(this->triangles.data_ptr<int>()),
        mode,
        reinterpret_cast<float*>(ar.data_ptr<float>())
    );
    
    return ar;
}

torch::Tensor TriangleMesh::get_radii_ratio() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(this->vertices.device());
    torch::Tensor ratios = torch::empty({static_cast<int64_t>(this->num_triangles)}, options);
    
    if (this->num_triangles == 0) return ratios;

    triangle_mesh::compute_radii_ratio(
        this->num_triangles,
        reinterpret_cast<const float3*>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3*>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float*>(ratios.data_ptr<float>())
    );
    
    return ratios;
}

torch::Tensor TriangleMesh::get_triangle_regularity() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(this->vertices.device());
    torch::Tensor regularities = torch::empty({static_cast<int64_t>(this->num_triangles)}, options);
    
    if (this->num_triangles == 0) return regularities;

    triangle_mesh::compute_triangle_regularity(
        this->num_triangles,
        reinterpret_cast<const float3*>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3*>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float*>(regularities.data_ptr<float>())
    );
    
    return regularities;
}

torch::Tensor TriangleMesh::get_radius_edge_ratio() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(this->vertices.device());
    torch::Tensor ratios = torch::empty({static_cast<int64_t>(this->num_triangles)}, options);
    
    if (this->num_triangles == 0) return ratios;

    triangle_mesh::compute_radius_edge_ratio(
        this->num_triangles,
        reinterpret_cast<const float3*>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3*>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float*>(ratios.data_ptr<float>())
    );
    
    return ratios;
}

torch::Tensor TriangleMesh::get_angle_deviation() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(this->vertices.device());
    torch::Tensor deviations = torch::empty({static_cast<int64_t>(this->num_triangles)}, options);
    
    if (this->num_triangles == 0) return deviations;

    triangle_mesh::compute_angle_deviation(
        this->num_triangles,
        reinterpret_cast<const float3*>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3*>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float*>(deviations.data_ptr<float>())
    );
    
    return deviations;
}

void TriangleMesh::compute_edges_to_triangle_map()
{
    at::cuda::CUDAGuard device_guard(this->triangles.device());

    if (this->num_triangles == 0)
    {
        auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(this->triangles.device());
        this->edges = torch::empty({0, 2}, options_i32);
        this->edge_to_triangle_offsets = torch::empty({0}, options_i32);
        this->edge_to_triangle_counts = torch::empty({0}, options_i32);
        this->edge_to_triangle_indices = torch::empty({0}, options_i32);
        return;
    }

    triangle_mesh::compute_edges_to_triangle_map(
        this->num_triangles,
        this->triangles,
        this->edges,
        this->edge_to_triangle_offsets,
        this->edge_to_triangle_counts,
        this->edge_to_triangle_indices);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TriangleMesh::get_edges_to_triangle_map()
{
    if (!this->edges.defined())
    {
        this->compute_edges_to_triangle_map();
    }
    return std::make_tuple(
        this->edges,
        this->edge_to_triangle_offsets,
        this->edge_to_triangle_counts,
        this->edge_to_triangle_indices);
}

torch::Tensor TriangleMesh::get_edges()
{
    if (!this->edges.defined())
        this->compute_edges_to_triangle_map();
    return this->edges;
}
torch::Tensor TriangleMesh::get_edge_to_triangle_offsets()
{
    if (!this->edge_to_triangle_offsets.defined())
        this->compute_edges_to_triangle_map();
    return this->edge_to_triangle_offsets;
}
torch::Tensor TriangleMesh::get_edge_to_triangle_counts()
{
    if (!this->edge_to_triangle_counts.defined())
        this->compute_edges_to_triangle_map();
    return this->edge_to_triangle_counts;
}
torch::Tensor TriangleMesh::get_edge_to_triangle_indices()
{
    if (!this->edge_to_triangle_indices.defined())
        this->compute_edges_to_triangle_map();
    return this->edge_to_triangle_indices;
}

bool TriangleMesh::is_edge_manifold(bool allow_boundary_edge)
{
    if (this->num_triangles == 0)
        return true;

    if (allow_boundary_edge && this->opt_edge_manifold_w_boundary.has_value()) {
        return this->opt_edge_manifold_w_boundary.value();
    } else if (!allow_boundary_edge && this->opt_edge_manifold.has_value()) {
        return this->opt_edge_manifold.value();
    }

    torch::Tensor counts = this->get_edge_to_triangle_counts();
    bool is_manifold;
    if (allow_boundary_edge)
    {
        is_manifold = (counts <= 2).all().item<bool>();
        this->opt_edge_manifold_w_boundary = is_manifold;
    }
    else
    {
        is_manifold = (counts == 2).all().item<bool>();
        this->opt_edge_manifold = is_manifold;
    }
    return is_manifold;
}

void TriangleMesh::remove_triangles_by_mask(const torch::Tensor &keep_mask)
{
    CHECK_INPUT(keep_mask);
    TORCH_CHECK(keep_mask.scalar_type() == torch::kBool, "keep_mask must be a boolean tensor");
    TORCH_CHECK(keep_mask.dim() == 1 && keep_mask.size(0) == this->num_triangles, "keep_mask must have shape (num_triangles,)");

    this->triangles = this->triangles.index({keep_mask});
    this->num_triangles = this->triangles.size(0);

    // Invalidate all caches
    this->triangle_areas = torch::Tensor();
    this->triangle_normals = torch::Tensor();
    this->surface_area = torch::Tensor();
    this->bvh.reset();
    this->opt_edge_manifold = std::nullopt;
    this->opt_edge_manifold_w_boundary = std::nullopt;
    this->opt_vertex_manifold = std::nullopt;
    this->opt_self_intersected = std::nullopt;
    this->edges = torch::Tensor();
    this->edge_to_triangle_offsets = torch::Tensor();
    this->edge_to_triangle_counts = torch::Tensor();
    this->edge_to_triangle_indices = torch::Tensor();

    this->vertex_to_triangle_offsets = torch::Tensor();
    this->vertex_to_triangle_counts = torch::Tensor();
    this->vertex_to_triangle_indices = torch::Tensor();
    
    this->vertex_degrees = torch::Tensor();
    this->vertex_lb_uniform = torch::Tensor();
    this->vertex_lb_cotangent = torch::Tensor();
    this->voronoi_areas = torch::Tensor();
}

void TriangleMesh::fix_normals()
{
    if (this->num_triangles == 0) return;
    
    if (!this->vertex_to_triangle_offsets.defined()) {
        this->compute_vertices_to_triangle_map();
    }
    
    triangle_mesh::fix_normals(
        this->num_triangles,
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        this->vertex_to_triangle_offsets,
        this->vertex_to_triangle_counts,
        this->vertex_to_triangle_indices,
        reinterpret_cast<int3 *>(this->triangles.data_ptr<int>()));
        
    // Invalidate caches
    this->triangle_areas = torch::Tensor();
    this->triangle_normals = torch::Tensor();
    this->vertex_normals = torch::Tensor();
    this->bvh.reset();
    this->opt_edge_manifold = std::nullopt;
    this->opt_edge_manifold_w_boundary = std::nullopt;
    this->opt_vertex_manifold = std::nullopt;
    this->opt_self_intersected = std::nullopt;
}

int32_t TriangleMesh::get_euler_characteristic()
{
    int32_t V = this->vertices.size(0);
    int32_t E = this->get_edges().size(0);
    int32_t F = this->num_triangles;
    return V - E + F;
}

int32_t TriangleMesh::get_genus()
{
    // For a single closed connected component, chi = 2 - 2g => g = (2 - chi) / 2
    int32_t chi = this->get_euler_characteristic();
    return (2 - chi) / 2;
}

void TriangleMesh::compute_vertices_to_triangle_map()
{
    if (this->vertex_to_triangle_offsets.defined())
        return;
    at::cuda::CUDAGuard device_guard(this->triangles.device());

    uint32_t num_vertices = this->vertices.size(0);
    triangle_mesh::build_vertices_to_triangle_map(
        num_vertices,
        this->num_triangles,
        this->triangles,
        this->vertex_to_triangle_counts,
        this->vertex_to_triangle_offsets,
        this->vertex_to_triangle_indices);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> TriangleMesh::get_vertices_to_triangle_map()
{
    this->compute_vertices_to_triangle_map();
    return {this->vertex_to_triangle_offsets, this->vertex_to_triangle_counts, this->vertex_to_triangle_indices};
}

torch::Tensor TriangleMesh::get_vertex_to_triangle_offsets()
{
    this->compute_vertices_to_triangle_map();
    return this->vertex_to_triangle_offsets;
}

torch::Tensor TriangleMesh::get_vertex_to_triangle_counts()
{
    this->compute_vertices_to_triangle_map();
    return this->vertex_to_triangle_counts;
}

torch::Tensor TriangleMesh::get_vertex_to_triangle_indices()
{
    this->compute_vertices_to_triangle_map();
    return this->vertex_to_triangle_indices;
}

torch::Tensor TriangleMesh::get_non_manifold_vertices()
{
    this->compute_vertices_to_triangle_map();
    return triangle_mesh::get_non_manifold_vertices(
        this->vertices.size(0),
        this->triangles,
        this->vertex_to_triangle_offsets,
        this->vertex_to_triangle_counts,
        this->vertex_to_triangle_indices);
}

bool TriangleMesh::is_vertex_manifold()
{
    if (this->opt_vertex_manifold.has_value()) {
        return this->opt_vertex_manifold.value();
    }
    torch::Tensor nm_vertices = this->get_non_manifold_vertices();
    bool is_manifold = (nm_vertices.size(0) == 0);
    this->opt_vertex_manifold = is_manifold;
    return is_manifold;
}

bool TriangleMesh::is_manifold(bool allow_boundary_edge)
{
    return this->is_edge_manifold(allow_boundary_edge) && this->is_vertex_manifold() && !this->is_self_intersection();
}

void bind_ds_triangle_mesh(py::module_ &m) {
    py::class_<TriangleMesh>(m, "TriangleMesh", R"pbdoc(
        GPU-accelerated Triangle Mesh data structure with differential geometry and spatial query primitives.

        Example:
            >>> import torch
            >>> from conquer3d._C import TriangleMesh
            >>> mesh = TriangleMesh(vertices, triangles)
            >>> is_m = mesh.is_manifold()
        )pbdoc")
        .def(py::init<const torch::Tensor &, const torch::Tensor &, std::optional<torch::Tensor>, std::optional<torch::Tensor>>(),
             py::arg("in_vertices"), py::arg("in_triangles"),
             py::arg("in_vertex_normals") = std::nullopt, py::arg("in_vertex_colors") = std::nullopt,
             R"pbdoc(
             Constructs a GPU TriangleMesh instance.

             Args:
                 in_vertices (torch.Tensor): (N, 3) float32 coordinates on CUDA.
                 in_triangles (torch.Tensor): (M, 3) int32 triangle vertex indices on CUDA.
                 in_vertex_normals (torch.Tensor, optional): (N, 3) float32 vertex normals on CUDA.
                 in_vertex_colors (torch.Tensor, optional): (N, 3) float32 RGB vertex colors on CUDA.

             Example:
                 >>> mesh = TriangleMesh(verts, tris)
             )pbdoc")
        .def_property_readonly("num_triangles", &TriangleMesh::get_num_triangles, "Total number of triangles in mesh (int).")
        .def_property_readonly("vertices", &TriangleMesh::get_vertices, "Mesh vertex coordinates (N, 3) float32.")
        .def_property_readonly("vertex_normals", [](TriangleMesh &self) { return self.get_vertex_normals(0); },
                               "Vertex normals (N, 3) float32 (unweighted average mode 0).")
        .def("get_vertex_normals", &TriangleMesh::get_vertex_normals, py::arg("mode") = 0,
             R"pbdoc(
             Retrieves or lazily computes vertex normals on GPU.

             Args:
                 mode (int, optional): Normal mode (0: unweighted face average, 1: incident angle-weighted pseudonormals). Defaults to 0.

             Returns:
                 torch.Tensor: (N, 3) float32 unit vertex normals.

             Example:
                 >>> v_normals = mesh.get_vertex_normals(mode=1)
             )pbdoc")
        .def("compute_vertex_normals", &TriangleMesh::compute_vertex_normals, py::arg("mode") = 0,
             R"pbdoc(
             Computes vertex normals on GPU.

             Args:
                 mode (int, optional): Normal mode (0: unweighted, 1: angle-weighted). Defaults to 0.

             Example:
                 >>> mesh.compute_vertex_normals(mode=1)
             )pbdoc")
        .def_property_readonly("vertex_colors", &TriangleMesh::get_vertex_colors, "Vertex colors (N, 3) float32.")
        .def_property_readonly("vertex_degrees", &TriangleMesh::get_vertex_degrees, "Vertex degrees (number of incident edges) (N,) int32.")
        .def_property_readonly("valence_567_percentage", &TriangleMesh::get_valence_567_percentage, "Percentage of vertices with valence 5, 6, or 7.")
        .def_property_readonly("vertex_lb_uniform", &TriangleMesh::get_vertex_lb_uniform, "Uniform Laplace-Beltrami operator (N, 3) float32.")
        .def_property_readonly("triangles", &TriangleMesh::get_triangles, "Mesh triangles (M, 3) int32.")
        .def_property_readonly("triangle_areas", &TriangleMesh::get_triangle_areas, "Areas of each triangle (M,) float32.")
        .def_property_readonly("triangle_normals", &TriangleMesh::get_triangle_normals, "Normals of each triangle (M, 3) float32.")
        .def_property_readonly("edge_normals", &TriangleMesh::get_edge_normals, "Directed edge normals (3*M, 3) float32 for sign queries.")
        .def_property_readonly("edge_normal", &TriangleMesh::get_edge_normal, "Alias for edge_normals.")
        .def("get_edge_normals", &TriangleMesh::get_edge_normals, "Get edge normals (3*M, 3) float32 for pseudonormal sign queries.")
        .def("get_edge_normal", &TriangleMesh::get_edge_normal, "Alias for get_edge_normals.")
        .def("compute_edge_normals", &TriangleMesh::compute_edge_normals, "Compute edge normals for pseudonormal sign queries.")
        .def("compute_edge_normal", &TriangleMesh::compute_edge_normal, "Alias for compute_edge_normals.")
        .def_property_readonly("surface_area", &TriangleMesh::get_surface_area, "Total surface area of the mesh.")
        .def("get_quality", &TriangleMesh::get_quality,
             R"pbdoc(
             Computes mesh triangle quality metric $Q = \frac{2\sqrt{3} \cdot r_{in}}{r_{circ}}$.

             Returns:
                 Tuple[float, float]: (min_quality, average_quality) in range [0, 1].

             Example:
                 >>> min_q, avg_q = mesh.get_quality()
             )pbdoc")
        .def("get_aspect_ratio", &TriangleMesh::get_aspect_ratio, py::arg("mode"),
             R"pbdoc(
             Computes aspect ratio for all triangles.

             Args:
                 mode (int): Aspect ratio formula mode.

             Returns:
                 torch.Tensor: (M,) float32 aspect ratio per face.

             Example:
                 >>> ar = mesh.get_aspect_ratio(0)
             )pbdoc")
        .def("get_radii_ratio", &TriangleMesh::get_radii_ratio,
             R"pbdoc(
             Computes ratio of incircle to circumcircle radius: $2 \cdot r_{in} / r_{circ}$.

             Returns:
                 torch.Tensor: (M,) float32 ratio values per face.

             Example:
                 >>> ratios = mesh.get_radii_ratio()
             )pbdoc")
        .def("get_triangle_regularity", &TriangleMesh::get_triangle_regularity,
             R"pbdoc(
             Computes triangle regularity: $2\sqrt{3} \cdot r_{in} / l_{max}$.

             Returns:
                 torch.Tensor: (M,) float32 regularity values in [0, 1].

             Example:
                 >>> reg = mesh.get_triangle_regularity()
             )pbdoc")
        .def("get_radius_edge_ratio", &TriangleMesh::get_radius_edge_ratio,
             R"pbdoc(
             Computes radius-edge ratio: $r_{circ} / l_{min}$.

             Returns:
                 torch.Tensor: (M,) float32 radius-edge ratio values.

             Example:
                 >>> re = mesh.get_radius_edge_ratio()
             )pbdoc")
        .def("get_angle_deviation", &TriangleMesh::get_angle_deviation,
             R"pbdoc(
             Computes mean internal angle deviation from equilateral 60 degrees.

             Returns:
                 torch.Tensor: (M,) float32 angle deviation in degrees.

             Example:
                 >>> dev = mesh.get_angle_deviation()
             )pbdoc")
        .def_property_readonly("bvh", &TriangleMesh::build_bvh, "The Bounding Volume Hierarchy (MeshBVH) built for this mesh.")
        .def("build_bvh", &TriangleMesh::build_bvh, "Builds and returns the MeshBVH for the mesh.")
        .def("build_flood_fill_data", &TriangleMesh::build_flood_fill_data,
             py::arg("grid_min") = py::none(), py::arg("grid_max") = py::none(),
             py::arg("res") = py::none(), py::arg("connectivity") = 6,
             R"pbdoc(
             Pre-computes a volumetric flood-fill occupancy grid for sign_mode=3.

             Args:
                 grid_min (List[float], optional): Lower grid extents [x, y, z].
                 grid_max (List[float], optional): Upper grid extents [x, y, z].
                 res (List[int], optional): Grid resolution [rx, ry, rz].
                 connectivity (int, optional): Neighborhood connectivity (6, 18, 26). Defaults to 6.

             Example:
                 >>> mesh.build_flood_fill_data([-1,-1,-1], [1,1,1], [128,128,128])
             )pbdoc")
        .def_property_readonly("flood_fill_mask", &TriangleMesh::get_flood_fill_mask, "Pre-computed flood fill int32 mask tensor.")
        .def_property_readonly("flood_grid_min", &TriangleMesh::get_flood_grid_min, "Flood grid bounding box min coordinates.")
        .def_property_readonly("flood_grid_max", &TriangleMesh::get_flood_grid_max, "Flood grid bounding box max coordinates.")
        .def_property_readonly("flood_grid_res", &TriangleMesh::get_flood_grid_res, "Flood grid vertex resolution.")
        .def("build_flood_fill_cf_data", &TriangleMesh::build_flood_fill_cf_data,
             py::arg("grid_min") = py::none(), py::arg("grid_max") = py::none(),
             py::arg("res") = py::none(), py::arg("block_size") = py::none(), py::arg("connectivity") = 6,
             R"pbdoc(
             Pre-computes a 2-level Coarse-to-Fine (CF) volumetric flood-fill structure (< 10 MB VRAM at 1024^3).

             Args:
                 grid_min (List[float], optional): Lower grid extents [x, y, z].
                 grid_max (List[float], optional): Upper grid extents [x, y, z].
                 res (List[int], optional): Fine grid resolution [rx, ry, rz].
                 block_size (List[int], optional): Macro-block size [bx, by, bz]. If omitted, dynamically computed.
                 connectivity (int, optional): Neighborhood connectivity (6, 18, 26). Defaults to 6.

             Example:
                 >>> mesh.build_flood_fill_cf_data([-1,-1,-1], [1,1,1], [1024,1024,1024])
             )pbdoc")
        .def_property_readonly("cf_coarse_mask", &TriangleMesh::get_cf_coarse_mask, "Coarse macro-block int8 status tensor.")
        .def_property_readonly("cf_fine_masks", &TriangleMesh::get_cf_fine_masks, "Fine boundary macro-block int8 local masks.")
        .def("get_self_intersection", &TriangleMesh::get_self_intersection,
             R"pbdoc(
             Finds all self-intersecting triangle pairs in the mesh.

             Returns:
                 torch.Tensor: (K, 2) int64 index pairs of colliding triangles.

             Example:
                 >>> pairs = mesh.get_self_intersection()
             )pbdoc")
        .def("is_self_intersection", &TriangleMesh::is_self_intersection,
             R"pbdoc(
             Checks if the mesh contains any self-intersecting triangle pairs.

             Returns:
                 bool: True if self-intersection detected, False otherwise.

             Example:
                 >>> has_self_int = mesh.is_self_intersection()
             )pbdoc")
        .def("get_ray_intersection", [](TriangleMesh &self, const torch::Tensor &ray_origins, const torch::Tensor &ray_dirs,
                                         bool return_distance) -> py::object {
                 auto result = self.get_ray_intersection(ray_origins, ray_dirs, return_distance);
                 if (return_distance) {
                     return py::cast(result);
                 } else {
                     return py::cast(std::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result)));
                 }
             },
             py::arg("ray_origins"), py::arg("ray_dirs"), py::arg("return_distance") = false,
             R"pbdoc(
             Computes ray-mesh surface intersections via Möller-Trumbore algorithm.

             Args:
                 ray_origins (torch.Tensor): (R, 3) float32 ray origins on CUDA.
                 ray_dirs (torch.Tensor): (R, 3) float32 normalized ray directions on CUDA.
                 return_distance (bool, optional): Return hit distance along ray. Defaults to False.

             Returns:
                 Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
                     - ray_ids (torch.Tensor): (K,) int64 ray indices.
                     - triangle_ids (torch.Tensor): (K,) int64 hit triangle indices.
                     - intersect_points (torch.Tensor): (K, 3) float32 exact 3D hit positions.
                     - [distances] (torch.Tensor, optional): (K,) float32 hit distances $t$.

             Example:
                 >>> ray_ids, tri_ids, pts = mesh.get_ray_intersection(origins, dirs)
             )pbdoc")
        .def("query_points", &TriangleMesh::query_points,
             py::arg("query_pts"), py::arg("return_sdf") = false, py::arg("return_prj_pts") = true,
             py::arg("sign_mode") = 0, py::arg("distance_mode") = 0,
             R"pbdoc(
             Finds closest triangles and computes Signed Distance Fields (SDF).

             Args:
                 query_pts (torch.Tensor): (Q, 3) float32 query coordinates on CUDA.
                 return_sdf (bool, optional): Return signed distance instead of unsigned. Defaults to False.
                 return_prj_pts (bool, optional): Return closest surface projection coordinates. Defaults to True.
                 sign_mode (int, optional): Sign evaluation method:
                     - 0: Ray parity casting.
                     - 1: Fast Winding Number (FWN).
                     - 2: Angle-weighted pseudonormals.
                     - 3: Volumetric 3D flood-fill mask (dense).
                     - 4: Hybrid WN + pseudonormals.
                     - 5: Coarse-to-Fine (CF) Hierarchical Volumetric Flood Fill (< 10 MB VRAM).
                     Defaults to 0.
                 distance_mode (int, optional): Distance algorithm (0: Ericson closest point, 1: projected normal). Defaults to 0.

             Returns:
                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                     - query_ids (torch.Tensor): (Q,) int64 query point indices.
                     - triangle_ids (torch.Tensor): (Q,) int64 closest triangle indices.
                     - projected_points (torch.Tensor): (Q, 3) float32 closest surface coordinates.
                     - distances (torch.Tensor): (Q,) float32 signed/unsigned distances.

             Example:
                 >>> q_ids, tri_ids, prj_pts, dists = mesh.query_points(query_pts, return_sdf=True, sign_mode=5)
             )pbdoc")
        .def_property_readonly("edges", &TriangleMesh::get_edges, "Unique edges of the mesh (E, 2) int32.")
        .def_property_readonly("edge_to_triangle_offsets", &TriangleMesh::get_edge_to_triangle_offsets, "Edge to triangle CSR offsets.")
        .def_property_readonly("edge_to_triangle_counts", &TriangleMesh::get_edge_to_triangle_counts, "Edge to triangle incident counts.")
        .def_property_readonly("edge_to_triangle_indices", &TriangleMesh::get_edge_to_triangle_indices, "Edge to triangle incident face indices.")
        .def("compute_triangle_areas", &TriangleMesh::compute_triangle_areas, "Computes the areas of all triangles.")
        .def("compute_triangle_normals", &TriangleMesh::compute_triangle_normals, "Computes the normals for all triangles.")
        .def("compute_edges_to_triangle_map", &TriangleMesh::compute_edges_to_triangle_map, "Computes the edge-to-triangle connectivity map.")
        .def("get_edges_to_triangle_map", &TriangleMesh::get_edges_to_triangle_map,
             R"pbdoc(
             Gets edge-to-triangle CSR connectivity structure.

             Returns:
                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: (edges, offsets, counts, indices)

             Example:
                 >>> edges, offsets, counts, indices = mesh.get_edges_to_triangle_map()
             )pbdoc")
        .def("compute_vertices_to_triangle_map", &TriangleMesh::compute_vertices_to_triangle_map, "Computes vertex-to-triangle connectivity map.")
        .def("get_vertices_to_triangle_map", &TriangleMesh::get_vertices_to_triangle_map,
             R"pbdoc(
             Gets vertex-to-triangle CSR connectivity structure.

             Returns:
                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (offsets, counts, indices)

             Example:
                 >>> offsets, counts, indices = mesh.get_vertices_to_triangle_map()
             )pbdoc")
        .def_property_readonly("vertex_to_triangle_offsets", &TriangleMesh::get_vertex_to_triangle_offsets, "Vertex to triangle CSR offsets.")
        .def_property_readonly("vertex_to_triangle_counts", &TriangleMesh::get_vertex_to_triangle_counts, "Vertex to triangle incident counts.")
        .def_property_readonly("vertex_to_triangle_indices", &TriangleMesh::get_vertex_to_triangle_indices, "Vertex to triangle incident face indices.")
        .def("is_edge_manifold", &TriangleMesh::is_edge_manifold, py::arg("allow_boundary_edge") = true,
             R"pbdoc(
             Checks if every mesh edge is shared by at most 2 triangles.

             Args:
                 allow_boundary_edge (bool, optional): If True, open boundary edges with 1 incident face are permitted. Defaults to True.

             Returns:
                 bool: True if edge-manifold, False otherwise.

             Example:
                 >>> is_em = mesh.is_edge_manifold(allow_boundary_edge=False)
             )pbdoc")
        .def("is_vertex_manifold", &TriangleMesh::is_vertex_manifold,
             R"pbdoc(
             Checks if every vertex forms a single topological disc or cone neighborhood.

             Returns:
                 bool: True if vertex-manifold, False otherwise.

             Example:
                 >>> is_vm = mesh.is_vertex_manifold()
             )pbdoc")
        .def("is_manifold", &TriangleMesh::is_manifold, py::arg("allow_boundary_edge") = true,
             R"pbdoc(
             Checks if the mesh is fully 2-manifold (edge-manifold, vertex-manifold, without self-intersections).

             Args:
                 allow_boundary_edge (bool, optional): Allow open boundary edges. Defaults to True.

             Returns:
                 bool: True if fully 2-manifold, False otherwise.

             Example:
                 >>> is_m = mesh.is_manifold()
             )pbdoc")
        .def("get_non_manifold_vertices", &TriangleMesh::get_non_manifold_vertices,
             R"pbdoc(
             Finds indices of non-manifold vertices in the mesh.

             Returns:
                 torch.Tensor: (K,) int32 indices of non-manifold vertices.

             Example:
                 >>> nm_verts = mesh.get_non_manifold_vertices()
             )pbdoc")
        .def("get_isolated_vertices", &TriangleMesh::get_isolated_vertices,
             R"pbdoc(
             Finds indices of isolated vertices with degree 0.

             Returns:
                 torch.Tensor: (K,) int32 indices of isolated vertices.

             Example:
                 >>> iso_verts = mesh.get_isolated_vertices()
             )pbdoc")
        .def_property_readonly("num_isolated_vertices", &TriangleMesh::get_num_isolated_vertices, "Total number of isolated vertices (int).")
        .def("remove_isolated_vertices", &TriangleMesh::remove_isolated_vertices,
             R"pbdoc(
             Removes unreferenced isolated vertices and compacts triangle indices.

             Returns:
                 Tuple[torch.Tensor, torch.Tensor]: (new_vertices, new_triangles)

             Example:
                 >>> clean_verts, clean_tris = mesh.remove_isolated_vertices()
             )pbdoc")
        .def("get_voronoi_areas", &TriangleMesh::get_voronoi_areas,
             R"pbdoc(
             Computes mixed Voronoi / barycentric dual cell areas per vertex (Meyer et al. 2003).

             Returns:
                 torch.Tensor: (N,) float32 Voronoi area per vertex.

             Example:
                 >>> voronoi_a = mesh.get_voronoi_areas()
             )pbdoc")
        .def("get_gaussian_curvature", &TriangleMesh::get_gaussian_curvature,
             R"pbdoc(
             Computes discrete Gaussian curvature via Gauss-Bonnet angle defect $K_i = \frac{2\pi - \sum \theta_j}{A_{Voronoi}}$.

             Returns:
                 torch.Tensor: (N,) float32 discrete Gaussian curvature.

             Example:
                 >>> gauss_curv = mesh.get_gaussian_curvature()
             )pbdoc")
        .def("get_mean_curvature", &TriangleMesh::get_mean_curvature, py::arg("signed_curvature") = false,
             R"pbdoc(
             Computes discrete Mean curvature using cotangent Laplace-Beltrami operator $H_i = \frac{1}{2} \|\Delta_{LB} v_i\|$.

             Args:
                 signed_curvature (bool, optional): Project against vertex normal for sign. Defaults to False.

             Returns:
                 torch.Tensor: (N,) float32 discrete Mean curvature.

             Example:
                 >>> mean_curv = mesh.get_mean_curvature(signed_curvature=True)
             )pbdoc")
        .def("get_principal_curvatures", &TriangleMesh::get_principal_curvatures, py::arg("signed_curvature") = true,
             R"pbdoc(
             Computes principal curvatures $(k_1, k_2) = H \pm \sqrt{\max(0, H^2 - K)}$.

             Args:
                 signed_curvature (bool, optional): Retain sign for mean curvature $H$. Defaults to True.

             Returns:
                 Tuple[torch.Tensor, torch.Tensor]: (k1, k2) principal curvatures of shape (N,) float32.

             Example:
                 >>> k1, k2 = mesh.get_principal_curvatures()
             )pbdoc")
        .def("compute_laplacian", &TriangleMesh::compute_laplacian, py::arg("mode") = 0,
             R"pbdoc(
             Computes Laplace-Beltrami vector field on mesh vertices.

             Args:
                 mode (int, optional): Laplacian mode (0: Uniform umbrella, 1: Cotangent weights). Defaults to 0.

             Returns:
                 torch.Tensor: (N, 3) float32 Laplacian vectors.

             Example:
                 >>> lap = mesh.compute_laplacian(mode=1)
             )pbdoc")
        .def_property_readonly("vertex_lb_cotangent", &TriangleMesh::get_vertex_lb_cotangent, "Cotangent Laplace-Beltrami vectors (N, 3) float32.")
        .def_property_readonly("voronoi_areas", &TriangleMesh::get_voronoi_areas, "Vertex Voronoi areas (N,) float32.")
        .def("remove_triangles_by_mask", &TriangleMesh::remove_triangles_by_mask, py::arg("keep_mask"),
             R"pbdoc(
             Filters mesh triangles using boolean mask and removes orphaned vertices.

             Args:
                 keep_mask (torch.Tensor): (M,) bool mask where True indicates kept triangles.

             Returns:
                 Tuple[torch.Tensor, torch.Tensor]: (filtered_vertices, filtered_triangles)

             Example:
                 >>> f_verts, f_tris = mesh.remove_triangles_by_mask(mask)
             )pbdoc")
        .def("fix_normals", &TriangleMesh::fix_normals,
             R"pbdoc(
             Reorients triangle winding orders consistently so outward normals face outside.

             Example:
                 >>> mesh.fix_normals()
             )pbdoc")
        .def("sample_points", &TriangleMesh::sample_points,
             py::arg("num_points"), py::arg("uniform") = false, py::arg("return_normals") = false,
             py::arg("return_colors") = false, py::arg("use_triangle_normal") = true,
             R"pbdoc(
             Samples random 3D points on the mesh surface via area-weighted CDF sampling.

             Args:
                 num_points (int): Number of points to sample.
                 uniform (bool, optional): Stratified uniform grid sampling. Defaults to False.
                 return_normals (bool, optional): Return surface normal at each sample. Defaults to False.
                 return_colors (bool, optional): Return interpolated RGB color at each sample. Defaults to False.
                 use_triangle_normal (bool, optional): Use face normal instead of interpolated vertex normal. Defaults to True.

             Returns:
                 Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                     - points (torch.Tensor): (num_points, 3) float32 coordinates.
                     - triangle_indices (torch.Tensor): (num_points,) int32 sampled face IDs.
                     - [normals] (torch.Tensor, optional): (num_points, 3) float32 normal vectors.
                     - [colors] (torch.Tensor, optional): (num_points, 3) float32 RGB colors.

             Example:
                 >>> pts, tri_ids, normals, _ = mesh.sample_points(5000, return_normals=True)
             )pbdoc")
        .def_property_readonly("euler_characteristic", &TriangleMesh::get_euler_characteristic, "The Euler characteristic (V - E + F) of the mesh (int).")
        .def_property_readonly("genus", &TriangleMesh::get_genus, "The topological genus of the mesh (int).");
}