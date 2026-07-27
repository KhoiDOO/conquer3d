#include <torch/extension.h>
#include "../../data_structure/triangle_mesh.h"
#include "../../ops/flood_fill.h"
#include "../../check.h"
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
    this->triangle_normals = torch::empty({static_cast<int64_t>(this->num_triangles), 3}, torch::dtype(torch::kFloat32).device(this->vertices.device()));
    triangle_mesh::compute_triangle_normals(
        this->num_triangles,
        reinterpret_cast<const float3 *>(this->vertices.data_ptr<float>()),
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
        reinterpret_cast<float3 *>(this->triangle_normals.data_ptr<float>()));
}

void TriangleMesh::compute_vertex_normals(int mode)
{
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
    this->edge_normals = torch::empty({static_cast<int64_t>(this->num_triangles) * 3, 3}, torch::dtype(torch::kFloat32).device(this->triangles.device()));
    triangle_mesh::compute_edge_normals(
        this->num_triangles,
        reinterpret_cast<const int3 *>(this->triangles.data_ptr<int>()),
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
            (sign_mode == 3) ? this->flood_grid_min : std::nullopt,
            (sign_mode == 3) ? this->flood_grid_max : std::nullopt,
            (sign_mode == 3) ? this->flood_grid_res : std::nullopt);
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
        reinterpret_cast<int3 *>(this->triangles.data_ptr<int>()),
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

void bind_ds_triangle_mesh(py::module_ &m)
{
    py::class_<TriangleMesh>(m, "TriangleMesh", R"doc(
        A highly efficient Triangle Mesh data structure natively backed by CUDA.

        Args:
            in_vertices (torch.Tensor): A tensor of shape (N, 3) containing float32 vertex coordinates.
            in_triangles (torch.Tensor): A tensor of shape (M, 3) containing int32 triangle indices.
            in_vertex_normals (torch.Tensor, optional): A tensor of shape (N, 3) containing float32 vertex normals. Defaults to None.
            in_vertex_colors (torch.Tensor, optional): A tensor of shape (N, 3) containing float32 vertex colors. Defaults to None.
        )doc")
        .def(py::init<const torch::Tensor &, const torch::Tensor &, std::optional<torch::Tensor>, std::optional<torch::Tensor>>(),
             py::arg("in_vertices"),
             py::arg("in_triangles"),
             py::arg("in_vertex_normals") = std::nullopt,
             py::arg("in_vertex_colors") = std::nullopt)
        .def_property_readonly("num_triangles", &TriangleMesh::get_num_triangles, R"doc(
        Total number of triangles.

        Returns:
            int - Total number of triangles.
        )doc")
        .def_property_readonly("vertices", &TriangleMesh::get_vertices, R"doc(
        Mesh vertices coordinates.

        Returns:
            torch.Tensor - Shape (N, 3) float32 tensor of vertices.
        )doc")
        .def_property_readonly("vertex_normals", [](TriangleMesh &self) { return self.get_vertex_normals(0); }, R"doc(
        Vertex normals (unweighted mode 0 by default).

        Returns:
            torch.Tensor - Shape (N, 3) float32 tensor of vertex normals.
        )doc")
        .def("get_vertex_normals", &TriangleMesh::get_vertex_normals, py::arg("mode") = 0, R"doc(
        Get vertex normals with specific mode (0: unweighted average, 1: angle-weighted pseudonormals).

        Args:
            mode (int, optional): Normal computation mode. Defaults to 0.

        Returns:
            torch.Tensor - Shape (N, 3) float32 tensor of vertex normals.
        )doc")
        .def("compute_vertex_normals", &TriangleMesh::compute_vertex_normals, py::arg("mode") = 0, R"doc(
        Compute vertex normals with specific mode (0: unweighted average, 1: angle-weighted pseudonormals).

        Args:
            mode (int, optional): Normal computation mode. Defaults to 0.
        )doc")
        .def_property_readonly("vertex_colors", &TriangleMesh::get_vertex_colors, R"doc(
        Vertex colors.

        Returns:
            torch.Tensor - Shape (N, 3) float32 tensor of vertex colors.
        )doc")
        .def_property_readonly("vertex_degrees", &TriangleMesh::get_vertex_degrees, R"doc(
        Degree of each vertex (number of incident edges).

        Returns:
            torch.Tensor - Shape (N,) int32 tensor of vertex degrees.
        )doc")
        .def_property_readonly("valence_567_percentage", &TriangleMesh::get_valence_567_percentage, R"doc(
        Percentage of vertices whose degrees (valences) are 5, 6, or 7.

        Returns:
            float - Percentage in range [0, 100].
        )doc")
        .def_property_readonly("vertex_lb_uniform", &TriangleMesh::get_vertex_lb_uniform, R"doc(
        Uniform Laplace-Beltrami operator evaluated at each vertex.

        Returns:
            torch.Tensor - Shape (N, 3) float32 tensor of uniform laplacian vectors.
        )doc")
        .def_property_readonly("triangles", &TriangleMesh::get_triangles, R"doc(
        Mesh triangles.

        Returns:
            torch.Tensor - Shape (M, 3) int32 tensor of triangles.
        )doc")
        .def_property_readonly("triangle_areas", &TriangleMesh::get_triangle_areas, R"doc(
        Areas of each triangle.

        Returns:
            torch.Tensor - Shape (M,) float32 tensor of triangle areas.
        )doc")
        .def_property_readonly("triangle_normals", &TriangleMesh::get_triangle_normals, R"doc(
        Normals of each triangle.

        Returns:
            torch.Tensor - Shape (M, 3) float32 tensor of triangle normals.
        )doc")
        .def_property_readonly("edge_normals", &TriangleMesh::get_edge_normals, R"doc(
        Normals of each directed edge slot (3 per triangle) for sign queries.

        Returns:
            torch.Tensor - Shape (3*M, 3) float32 tensor of edge normals.
        )doc")
        .def_property_readonly("edge_normal", &TriangleMesh::get_edge_normal, R"doc(
        Alias for edge_normals.

        Returns:
            torch.Tensor - Shape (3*M, 3) float32 tensor of edge normals.
        )doc")
        .def("get_edge_normals", &TriangleMesh::get_edge_normals, R"doc(
        Get edge normals for pseudonormal sign queries.

        Returns:
            torch.Tensor - Shape (3*M, 3) float32 tensor of edge normals.
        )doc")
        .def("get_edge_normal", &TriangleMesh::get_edge_normal, R"doc(
        Alias for get_edge_normals.

        Returns:
            torch.Tensor - Shape (3*M, 3) float32 tensor of edge normals.
        )doc")
        .def("compute_edge_normals", &TriangleMesh::compute_edge_normals, R"doc(
        Compute edge normals for pseudonormal sign queries.
        )doc")
        .def("compute_edge_normal", &TriangleMesh::compute_edge_normal, R"doc(
        Alias for compute_edge_normals.
        )doc")
        .def_property_readonly("surface_area", &TriangleMesh::get_surface_area, R"doc(
        Total surface area of the mesh.

        Returns:
            torch.Tensor - Total surface area of the mesh.
        )doc")
        .def("get_quality", &TriangleMesh::get_quality, R"doc(
        Compute the global quality of the mesh.

        Returns:
            Tuple[float, float] - Minimum and average triangle quality.
        )doc")
        .def("get_aspect_ratio", &TriangleMesh::get_aspect_ratio, py::arg("mode"), R"doc(
        Compute aspect ratio for all triangles.

        Args:
            mode (int): The formula mode to use.

        Returns:
            torch.Tensor - Shape (M,) float32 tensor of triangle aspect ratios.
        )doc")
        .def("get_radii_ratio", &TriangleMesh::get_radii_ratio, R"doc(
        Compute the radii ratio (incircle / circumcircle) for all triangles.

        Returns:
            torch.Tensor - Shape (M,) float32 tensor of triangle radii ratios.
        )doc")
        .def("get_triangle_regularity", &TriangleMesh::get_triangle_regularity, R"doc(
        Compute the triangle regularity (2 * sqrt(3) * r / l) for all triangles.

        Returns:
            torch.Tensor - Shape (M,) float32 tensor of triangle regularities.
        )doc")
        .def("get_radius_edge_ratio", &TriangleMesh::get_radius_edge_ratio, R"doc(
        Compute the radius edge ratio (R / e) for all triangles.

        Returns:
            torch.Tensor - Shape (M,) float32 tensor of triangle radius edge ratios.
        )doc")
        .def("get_angle_deviation", &TriangleMesh::get_angle_deviation, R"doc(
        Compute the mean angle deviation from 60 degrees for all triangles.

        Returns:
            torch.Tensor - Shape (M,) float32 tensor of triangle angle deviations.
        )doc")
        .def_property_readonly("bvh", &TriangleMesh::build_bvh, R"doc(
        The Bounding Volume Hierarchy built for this mesh.

        Returns:
            MeshBVH - The Bounding Volume Hierarchy built for this mesh.
        )doc")
        .def("build_bvh", &TriangleMesh::build_bvh, R"doc(
        Builds and returns the Bounding Volume Hierarchy for the mesh.

        Returns:
            MeshBVH: The constructed BVH object.
        )doc")
        .def("build_flood_fill_data", &TriangleMesh::build_flood_fill_data, py::arg("grid_min") = py::none(), py::arg("grid_max") = py::none(), py::arg("res") = py::none(), py::arg("connectivity") = 6, R"doc(
        Builds volumetric flood fill mask for sign_mode=3.

        Args:
            grid_min (list, optional): Bounding grid minimum coordinates [x, y, z].
            grid_max (list, optional): Bounding grid maximum coordinates [x, y, z].
            res (list, optional): Grid vertex resolutions [rx, ry, rz].
            connectivity (int, optional): Voxel connectivity (6, 18, or 26). Defaults to 6.
        )doc")
        .def_property_readonly("flood_fill_mask", &TriangleMesh::get_flood_fill_mask, R"doc(
        The pre-computed flood fill int32 mask tensor.
        )doc")
        .def_property_readonly("flood_grid_min", &TriangleMesh::get_flood_grid_min, R"doc(
        The flood grid bounding box min coordinates.
        )doc")
        .def_property_readonly("flood_grid_max", &TriangleMesh::get_flood_grid_max, R"doc(
        The flood grid bounding box max coordinates.
        )doc")
        .def_property_readonly("flood_grid_res", &TriangleMesh::get_flood_grid_res, R"doc(
        The flood grid vertex resolution.
        )doc")
        .def("get_self_intersection", &TriangleMesh::get_self_intersection, R"doc(
        Finds all self-intersecting triangle pairs in the mesh.

        Returns:
            torch.Tensor: A tensor of shape (K, 2) containing pairs of intersecting triangle indices.
        )doc")
        .def("is_self_intersection", &TriangleMesh::is_self_intersection, R"doc(
        Checks whether the mesh contains any self-intersecting triangles.

        Returns:
            bool: True if there is at least one self-intersection, False otherwise.
        )doc")
        .def("get_ray_intersection", [](TriangleMesh &self, const torch::Tensor &ray_origins, const torch::Tensor &ray_dirs, bool return_distance) -> py::object
             {
                 auto result = self.get_ray_intersection(ray_origins, ray_dirs, return_distance);
                 if (return_distance) {
                     return py::cast(result);
                 } else {
                     return py::cast(std::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result)));
                 } }, py::arg("ray_origins"), py::arg("ray_dirs"), py::arg("return_distance") = false,
             R"doc(
        Computes intersections between a batch of rays and the mesh triangles.

        Args:
            ray_origins (torch.Tensor): Shape (R, 3) float32 tensor of ray origins.
            ray_dirs (torch.Tensor): Shape (R, 3) float32 tensor of ray directions.
            return_distance (bool, optional): If True, returns intersection distances. Defaults to False.

        Returns:
            tuple: (ray_ids, triangle_ids, intersect_points, [distances])
        )doc")
        .def("query_points", &TriangleMesh::query_points, py::arg("query_pts"), py::arg("return_sdf") = false, py::arg("return_prj_pts") = true, py::arg("sign_mode") = 0, py::arg("distance_mode") = 0,
             R"doc(
        Finds the closest triangles and computes distances/SDFs for query points.

        Args:
            query_pts (torch.Tensor): Shape (Q, 3) float32 tensor of query points.
            return_sdf (bool, optional): Whether to return Signed Distance Field values. Defaults to False.
            return_prj_pts (bool, optional): Whether to return projected points on the mesh. Defaults to True.
            sign_mode (int, optional): The method for computing signs (0: ray casting parity, 1: fast winding number, 2: angle-weighted pseudonormals, 3: volumetric flood fill + pseudonormal hybrid, 4: hybrid WN + pseudonormals). Defaults to 0.
            distance_mode (int, optional): Distance computation mode. Defaults to 0.

        Returns:
            tuple: (query_ids, triangle_ids, projected_points, distances)
        )doc")
        .def_property_readonly("edges", &TriangleMesh::get_edges, R"doc(
        Unique edges of the mesh.

        Returns:
            torch.Tensor - Shape (E, 2) int32 tensor of unique edges.
        )doc")
        .def_property_readonly("edge_to_triangle_offsets", &TriangleMesh::get_edge_to_triangle_offsets, R"doc(
        Edge to triangle offsets.

        Returns:
            torch.Tensor - Edge to triangle offsets.
        )doc")
        .def_property_readonly("edge_to_triangle_counts", &TriangleMesh::get_edge_to_triangle_counts, R"doc(
        Edge to triangle counts.

        Returns:
            torch.Tensor - Edge to triangle counts.
        )doc")
        .def_property_readonly("edge_to_triangle_indices", &TriangleMesh::get_edge_to_triangle_indices, R"doc(
        Edge to triangle indices.

        Returns:
            torch.Tensor - Edge to triangle indices.
        )doc")
        .def("compute_triangle_areas", &TriangleMesh::compute_triangle_areas, "Computes the areas of all triangles.")
        .def("compute_triangle_normals", &TriangleMesh::compute_triangle_normals, "Computes the normals for all triangles.")
        .def("compute_edges_to_triangle_map", &TriangleMesh::compute_edges_to_triangle_map, "Computes the edge-to-triangle connectivity map.")
        .def("get_edges_to_triangle_map", &TriangleMesh::get_edges_to_triangle_map, R"doc(
        Gets the edge to triangle connectivity mapping.

        Returns:
            tuple: (edges, offsets, counts, indices)
        )doc")
        .def("compute_vertices_to_triangle_map", &TriangleMesh::compute_vertices_to_triangle_map, "Computes the vertex-to-triangle connectivity map.")
        .def("get_vertices_to_triangle_map", &TriangleMesh::get_vertices_to_triangle_map, R"doc(
        Gets the vertex to triangle connectivity mapping.

        Returns:
            tuple: (offsets, counts, indices)
        )doc")
        .def_property_readonly("vertex_to_triangle_offsets", &TriangleMesh::get_vertex_to_triangle_offsets, R"doc(
        Vertex to triangle offsets.

        Returns:
            torch.Tensor - Vertex to triangle offsets.
        )doc")
        .def_property_readonly("vertex_to_triangle_counts", &TriangleMesh::get_vertex_to_triangle_counts, R"doc(
        Vertex to triangle counts.

        Returns:
            torch.Tensor - Vertex to triangle counts.
        )doc")
        .def_property_readonly("vertex_to_triangle_indices", &TriangleMesh::get_vertex_to_triangle_indices, R"doc(
        Vertex to triangle indices.

        Returns:
            torch.Tensor - Vertex to triangle indices.
        )doc")
        .def("is_edge_manifold", &TriangleMesh::is_edge_manifold, py::arg("allow_boundary_edge") = true, R"doc(
        Checks if the mesh is edge manifold.

        Args:
            allow_boundary_edge (bool, optional): Whether to permit boundary edges (count <= 2). Defaults to True.

        Returns:
            bool: True if the mesh is edge manifold.
        )doc")
        .def("is_vertex_manifold", &TriangleMesh::is_vertex_manifold, R"doc(
        Checks if the mesh is vertex manifold.

        Returns:
            bool: True if all vertices are manifold.
        )doc")
        .def("is_manifold", &TriangleMesh::is_manifold, py::arg("allow_boundary_edge") = true, R"doc(
        Checks if the mesh is fully manifold (edge, vertex, and no self-intersections).

        Args:
            allow_boundary_edge (bool, optional): Whether to permit boundary edges. Defaults to True.

        Returns:
            bool: True if fully manifold.
        )doc")
        .def("get_non_manifold_vertices", &TriangleMesh::get_non_manifold_vertices, R"doc(
        Gets all non-manifold vertices.

        Returns:
            torch.Tensor: Tensor of vertex indices that are non-manifold.
        )doc")
        .def("get_isolated_vertices", &TriangleMesh::get_isolated_vertices, R"doc(
        Gets the indices of all isolated vertices (degree 0).

        Returns:
            torch.Tensor - 1D tensor of isolated vertex indices.
        )doc")
        .def_property_readonly("num_isolated_vertices", &TriangleMesh::get_num_isolated_vertices, R"doc(
        Gets the total number of isolated vertices.

        Returns:
            int - The total number of isolated vertices.
        )doc")
        .def("remove_isolated_vertices", &TriangleMesh::remove_isolated_vertices, R"doc(
        Removes all isolated vertices from the mesh and reindexes the triangles.
        )doc")
        .def("get_voronoi_areas", &TriangleMesh::get_voronoi_areas, R"doc(
            Get the per-vertex voronoi areas.
        )doc")
        .def("get_gaussian_curvature", &TriangleMesh::get_gaussian_curvature, R"doc(
            Compute and return the discrete Gaussian curvature (angle deficit) at each vertex.
        )doc")
        .def("get_mean_curvature", &TriangleMesh::get_mean_curvature, py::arg("signed_curvature") = false, R"doc(
            Compute and return the discrete Mean curvature at each vertex using the Laplace-Beltrami operator.
            If signed_curvature is true, returns H. If false, returns |H|.
        )doc")
        .def("get_principal_curvatures", &TriangleMesh::get_principal_curvatures, py::arg("signed_curvature") = true, R"doc(
            Compute and return the two principal curvatures (k1, k2) at each vertex as an [N, 2] tensor.
            k1 is the maximum principal curvature, k2 is the minimum principal curvature.
        )doc")
        .def("compute_laplacian", &TriangleMesh::compute_laplacian, py::arg("mode") = 0, R"doc(
        Computes the Laplace-Beltrami operator.

        Args:
            mode (int): The laplacian mode. 0 for Uniform, 1 for Cotangent.

        Returns:
            torch.Tensor - Shape (N, 3) float32 tensor of laplacian vectors.
        )doc")
        .def_property_readonly("vertex_lb_cotangent", &TriangleMesh::get_vertex_lb_cotangent, R"doc(
        Cotangent Laplace-Beltrami operator evaluated at each vertex.

        Returns:
            torch.Tensor - Shape (N, 3) float32 tensor.
        )doc")
        .def_property_readonly("voronoi_areas", &TriangleMesh::get_voronoi_areas, R"doc(
        Vertex voronoi areas (1/3 of the sum of incident triangle areas).

        Returns:
            torch.Tensor - Shape (N,) float32 tensor.
        )doc")
        .def("remove_triangles_by_mask", &TriangleMesh::remove_triangles_by_mask, py::arg("keep_mask"), R"doc(
        Removes triangles from the mesh based on a boolean mask.

        Args:
            keep_mask (torch.Tensor): Shape (M,) boolean tensor indicating which triangles to keep.
        )doc")
        .def("fix_normals", &TriangleMesh::fix_normals, R"doc(
        Fixes the winding order and outward orientation of the mesh normals.
        This uses a CUDA-accelerated BFS to ensure consistent winding, 
        and computes signed volumes to ensure all disconnected components face outward.
        )doc")
        .def("sample_points", &TriangleMesh::sample_points, py::arg("num_points"), py::arg("uniform") = false, py::arg("return_normals") = false, py::arg("return_colors") = false, py::arg("use_triangle_normal") = true, R"doc(
        Samples random points on the surface of the mesh.

        Args:
            num_points (int): The number of points to sample.
            uniform (bool, optional): If True, samples uniformly by area. Defaults to False.
            return_normals (bool, optional): If True, returns normals at sampled points. Defaults to False.
            return_colors (bool, optional): If True, returns colors at sampled points. Defaults to False.
            use_triangle_normal (bool, optional): If True, uses flat triangle normals instead of interpolated vertex normals. Defaults to True.

        Returns:
            tuple: (points, triangle_indices, [normals], [colors])
        )doc")
        .def("compute_vertex_normals", &TriangleMesh::compute_vertex_normals, "Computes area-weighted vertex normals.")
        .def_property_readonly("euler_characteristic", &TriangleMesh::get_euler_characteristic, R"doc(
        The Euler characteristic (V - E + F) of the mesh.

        Returns:
            int - The Euler characteristic (V - E + F) of the mesh.
        )doc")
        .def_property_readonly("genus", &TriangleMesh::get_genus, R"doc(
        The topological genus of the mesh.

        Returns:
            int - The topological genus of the mesh.
        )doc");
}