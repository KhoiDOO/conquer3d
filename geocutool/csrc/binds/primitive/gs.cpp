#include <torch/extension.h>
#include "../../primitive/gs.h"
#include "../../check.h"

#include <cuda_fp16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

torch::Tensor compute_gs_covi_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &rotations,
    const torch::Tensor &scales,
    const bool rotnorm,
    const float tol,
    const uint32_t level)
{
    CHECK_INPUT(means);
    CHECK_INPUT(rotations);
    CHECK_INPUT(scales);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(rotations.scalar_type() == torch::kFloat32, "rotations must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(rotations.size(1) == 4, "rotations must have shape (N, 4)");
    TORCH_CHECK(scales.size(1) == 3, "scales must have shape (N, 3)");

    auto options = means.options();
    torch::Tensor covi = torch::empty({num_gaussians, 6}, options);

    gs::compute_gs_covi(
        num_gaussians,
        reinterpret_cast<const float4 *>(rotations.data_ptr<float>()),
        reinterpret_cast<const float3 *>(scales.data_ptr<float>()),
        rotnorm,
        tol,
        level,
        reinterpret_cast<float *>(covi.data_ptr<float>()));
    
    return covi;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_gs_aabb_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &scales,
    const torch::Tensor &covis,
    const std::optional<torch::Tensor> &isos,
    const float iso,
    const float tol,
    const uint32_t level)
{
    CHECK_INPUT(means);
    CHECK_INPUT(scales);
    CHECK_INPUT(covis);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(scales.size(1) == 3, "scales must have shape (N, 3)");

    float *isos_ptr = nullptr;
    if (isos.has_value())
    {
        CHECK_INPUT(isos.value());
        TORCH_CHECK(isos.value().scalar_type() == torch::kFloat32, "isos must be float32");
        TORCH_CHECK(isos.value().size(0) == num_gaussians, "isos must have shape (N,)");
        isos_ptr = isos.value().data_ptr<float>();
    }

    auto options = means.options();
    torch::Tensor aabb_min = torch::empty({num_gaussians, 3}, options);
    torch::Tensor aabb_max = torch::empty({num_gaussians, 3}, options);
    torch::Tensor contact_points = torch::empty({num_gaussians, 9}, options);

    gs_aabb::compute_gs_aabb(
        num_gaussians,
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float3 *>(scales.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        isos_ptr,
        iso,
        tol,
        level,
        reinterpret_cast<float3 *>(aabb_min.data_ptr<float>()),
        reinterpret_cast<float3 *>(aabb_max.data_ptr<float>()),
        reinterpret_cast<float3 *>(contact_points.data_ptr<float>()));
    
    return std::make_tuple(aabb_min, aabb_max, contact_points);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> compute_gs_aabb_w_covi_wrapper(
    const torch::Tensor &means,
    const torch::Tensor &rotations,
    const torch::Tensor &scales,
    const std::optional<torch::Tensor> &isos,
    const float iso,
    const float tol,
    const uint32_t level,
    const bool rotnorm)
{
    CHECK_INPUT(means);
    CHECK_INPUT(rotations);
    CHECK_INPUT(scales);

    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(rotations.scalar_type() == torch::kFloat32, "rotations must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");

    const uint32_t num_gaussians = means.size(0);
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(rotations.size(1) == 4, "rotations must have shape (N, 4)");
    TORCH_CHECK(scales.size(1) == 3, "scales must have shape (N, 3)");

    float *isos_ptr = nullptr;
    if (isos.has_value())
    {
        CHECK_INPUT(isos.value());
        TORCH_CHECK(isos.value().scalar_type() == torch::kFloat32, "isos must be float32");
        TORCH_CHECK(isos.value().size(0) == num_gaussians, "isos must have shape (N,)");
        isos_ptr = isos.value().data_ptr<float>();
    }

    auto options = means.options();
    torch::Tensor aabb_min = torch::empty({num_gaussians, 3}, options);
    torch::Tensor aabb_max = torch::empty({num_gaussians, 3}, options);
    torch::Tensor contact_points = torch::empty({num_gaussians, 9}, options);
    torch::Tensor covi = torch::empty({num_gaussians, 6}, options);

    gs_aabb::compute_gs_aabb_w_covi(
        num_gaussians,
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float4 *>(rotations.data_ptr<float>()),
        reinterpret_cast<const float3 *>(scales.data_ptr<float>()),
        isos_ptr,
        iso,
        tol,
        level,
        rotnorm,
        reinterpret_cast<float3 *>(aabb_min.data_ptr<float>()),
        reinterpret_cast<float3 *>(aabb_max.data_ptr<float>()),
        reinterpret_cast<float3 *>(contact_points.data_ptr<float>()),
        reinterpret_cast<float *>(covi.data_ptr<float>()));

    return std::make_tuple(aabb_min, aabb_max, contact_points, covi);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> query_gs_voxel_pair_intersection_brute_force_wrapper(
    const torch::Tensor &vx_aabb_mins,
    const torch::Tensor &vx_aabb_maxs,
    const torch::Tensor &means,
    const torch::Tensor &covis,
    const torch::Tensor &opacities,
    const torch::Tensor &gs_aabb_mins,
    const torch::Tensor &gs_aabb_maxs,
    const torch::Tensor &contact_points,
    const std::optional<torch::Tensor> &isos,
    const float iso,
    const float ar_threshold,
    const float p_threshold,
    const bool return_centroids,
    const int64_t max_capacity)
{
    CHECK_INPUT(vx_aabb_mins);
    CHECK_INPUT(vx_aabb_maxs);
    CHECK_INPUT(means);
    CHECK_INPUT(covis);
    CHECK_INPUT(opacities);
    CHECK_INPUT(gs_aabb_mins);
    CHECK_INPUT(gs_aabb_maxs);
    CHECK_INPUT(contact_points);

    TORCH_CHECK(vx_aabb_mins.scalar_type() == torch::kFloat32, "vx_aabb_mins must be float32");
    TORCH_CHECK(vx_aabb_maxs.scalar_type() == torch::kFloat32, "vx_aabb_maxs must be float32");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");
    TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
    TORCH_CHECK(gs_aabb_mins.scalar_type() == torch::kFloat32, "gs_aabb_mins must be float32");
    TORCH_CHECK(gs_aabb_maxs.scalar_type() == torch::kFloat32, "gs_aabb_maxs must be float32");
    TORCH_CHECK(contact_points.scalar_type() == torch::kFloat32, "contact_points must be float32");

    const uint32_t num_gaussians = means.size(0);
    const uint32_t num_voxels = vx_aabb_mins.size(0);

    TORCH_CHECK(vx_aabb_mins.size(1) == 3, "vx_aabb_mins must have shape (M, 3)");
    TORCH_CHECK(vx_aabb_maxs.size(0) == num_voxels && vx_aabb_maxs.size(1) == 3, "vx_aabb_maxs must match vx_aabb_mins shape");
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(covis.size(0) == num_gaussians && covis.size(1) == 6, "covis must have shape (N, 6)");
    TORCH_CHECK(opacities.size(0) == num_gaussians, "opacities must have shape (N)");
    TORCH_CHECK(gs_aabb_mins.size(0) == num_gaussians && gs_aabb_mins.size(1) == 3, "gs_aabb_mins must match means shape");
    TORCH_CHECK(gs_aabb_maxs.size(0) == num_gaussians && gs_aabb_maxs.size(1) == 3, "gs_aabb_maxs must match means shape");
    TORCH_CHECK(contact_points.size(0) == num_gaussians && contact_points.size(1) == 9, "contact_points must have shape (N, 9)");

    float *isos_ptr = nullptr;
    if (isos.has_value())
    {
        CHECK_INPUT(isos.value());
        TORCH_CHECK(isos.value().scalar_type() == torch::kFloat32, "isos must be float32");
        TORCH_CHECK(isos.value().size(0) == num_gaussians, "isos must have shape (N,)");
        isos_ptr = isos.value().data_ptr<float>();
    }

    auto options = means.options();
    torch::Tensor hit_mask = torch::zeros({num_voxels}, options.dtype(torch::kBool));
    torch::Tensor out_voxel_ids = torch::empty({max_capacity}, options.dtype(torch::kInt64));
    torch::Tensor out_gaus_ids = torch::empty({max_capacity}, options.dtype(torch::kInt64));
    torch::Tensor global_counter = torch::zeros({1}, options.dtype(torch::kInt64));

    torch::Tensor centroids;
    torch::Tensor densities;
    float3 *centroids_ptr = nullptr;
    float *densities_ptr = nullptr;

    if (return_centroids)
    {
        centroids = torch::empty({max_capacity, 3}, options.dtype(torch::kFloat32));
        centroids_ptr = reinterpret_cast<float3 *>(centroids.data_ptr<float>());
        densities = torch::empty({max_capacity}, options.dtype(torch::kFloat32));
        densities_ptr = reinterpret_cast<float *>(densities.data_ptr<float>());
    }

    gs_aabb::query_gs_voxel_pair_intersection_brute_force(
        num_voxels,
        num_gaussians,
        reinterpret_cast<const float3 *>(vx_aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(vx_aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        reinterpret_cast<const float *>(opacities.data_ptr<float>()),
        reinterpret_cast<const float3 *>(gs_aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(gs_aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const float3 *>(contact_points.data_ptr<float>()),
        isos_ptr,
        iso,
        ar_threshold,
        p_threshold,
        return_centroids,
        reinterpret_cast<bool *>(hit_mask.data_ptr<bool>()),
        reinterpret_cast<int64_t *>(out_voxel_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(out_gaus_ids.data_ptr<int64_t>()),
        centroids_ptr,
        densities_ptr,
        reinterpret_cast<int64_t *>(global_counter.data_ptr<int64_t>()),
        max_capacity);

    int64_t num_intersections = global_counter.item<int64_t>();
    if (num_intersections > max_capacity)
    {
        TORCH_WARN("Exceeded max capacity! Found ", num_intersections, " hits but capacity was ", max_capacity);
    }
    int64_t valid_hits = std::min(num_intersections, max_capacity);
    if (return_centroids)
    {
        return std::make_tuple(
            hit_mask,
            out_voxel_ids.slice(0, 0, valid_hits),
            out_gaus_ids.slice(0, 0, valid_hits),
            centroids.slice(0, 0, valid_hits),
            densities.slice(0, 0, valid_hits));
    }
    else
    {
        return std::make_tuple(
            hit_mask,
            out_voxel_ids.slice(0, 0, valid_hits),
            out_gaus_ids.slice(0, 0, valid_hits),
            std::nullopt,
            std::nullopt);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> query_gs_edge_pair_intersection_brute_force_wrapper(
    const torch::Tensor &edge_starts,
    const torch::Tensor &edge_ends,
    const torch::Tensor &means,
    const torch::Tensor &covis,
    const std::optional<torch::Tensor> &isos,
    const float iso,
    const int64_t max_capacity)
{
    CHECK_INPUT(edge_starts);
    CHECK_INPUT(edge_ends);
    CHECK_INPUT(means);
    CHECK_INPUT(covis);

    TORCH_CHECK(edge_starts.scalar_type() == torch::kFloat32, "edge_starts must be float32");
    TORCH_CHECK(edge_ends.scalar_type() == torch::kFloat32, "edge_ends must be float32");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    const uint32_t num_edges = edge_starts.size(0);
    const uint32_t num_gaussians = means.size(0);

    TORCH_CHECK(edge_starts.size(1) == 3, "edge_starts must have shape (E, 3)");
    TORCH_CHECK(edge_ends.size(0) == num_edges && edge_ends.size(1) == 3, "edge_ends must match edge_starts");
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(covis.size(0) == num_gaussians && covis.size(1) == 6, "covis must have shape (N, 6)");

    float *isos_ptr = nullptr;
    if (isos.has_value())
    {
        CHECK_INPUT(isos.value());
        TORCH_CHECK(isos.value().scalar_type() == torch::kFloat32, "isos must be float32");
        TORCH_CHECK(isos.value().size(0) == num_gaussians, "isos must have shape (N,)");
        isos_ptr = isos.value().data_ptr<float>();
    }

    auto options = means.options();
    torch::Tensor hit_mask = torch::zeros({num_edges}, options.dtype(torch::kBool));
    torch::Tensor out_edge_ids = torch::empty({max_capacity}, options.dtype(torch::kInt64));
    torch::Tensor out_gaus_ids = torch::empty({max_capacity}, options.dtype(torch::kInt64));
    torch::Tensor global_counter = torch::zeros({1}, options.dtype(torch::kInt64));

    gs_aabb::query_gs_edge_pair_intersection_brute_force(
        num_edges,
        num_gaussians,
        reinterpret_cast<const float3 *>(edge_starts.data_ptr<float>()),
        reinterpret_cast<const float3 *>(edge_ends.data_ptr<float>()),
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        isos_ptr,
        iso,
        reinterpret_cast<bool *>(hit_mask.data_ptr<bool>()),
        reinterpret_cast<int64_t *>(out_edge_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(out_gaus_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(global_counter.data_ptr<int64_t>()),
        max_capacity);

    int64_t num_intersections = global_counter.item<int64_t>();
    if (num_intersections > max_capacity)
    {
        TORCH_WARN("Exceeded max capacity! Found ", num_intersections, " hits but capacity was ", max_capacity);
    }
    int64_t valid_hits = std::min(num_intersections, max_capacity);

    return std::make_tuple(
        hit_mask,
        out_edge_ids.slice(0, 0, valid_hits),
        out_gaus_ids.slice(0, 0, valid_hits));
}

std::tuple<torch::Tensor, torch::Tensor> query_gs_edge_intersection_brute_force_wrapper(
    const torch::Tensor &edge_starts,
    const torch::Tensor &edge_ends,
    const torch::Tensor &means,
    const torch::Tensor &opacities,
    const torch::Tensor &covis,
    const std::optional<torch::Tensor> &isos,
    const float iso)
{
    CHECK_INPUT(edge_starts);
    CHECK_INPUT(edge_ends);
    CHECK_INPUT(means);
    CHECK_INPUT(opacities);
    CHECK_INPUT(covis);

    TORCH_CHECK(edge_starts.scalar_type() == torch::kFloat32, "edge_starts must be float32");
    TORCH_CHECK(edge_ends.scalar_type() == torch::kFloat32, "edge_ends must be float32");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    const uint32_t num_edges = edge_starts.size(0);
    const uint32_t num_gaussians = means.size(0);

    TORCH_CHECK(edge_starts.size(1) == 3, "edge_starts must have shape (E, 3)");
    TORCH_CHECK(edge_ends.size(0) == num_edges && edge_ends.size(1) == 3, "edge_ends must match edge_starts");
    TORCH_CHECK(means.size(1) == 3, "means must have shape (N, 3)");
    TORCH_CHECK(covis.size(0) == num_gaussians && covis.size(1) == 6, "covis must have shape (N, 6)");
    TORCH_CHECK(opacities.size(0) == num_gaussians, "opacities must have shape (N)");

    float *isos_ptr = nullptr;
    if (isos.has_value())
    {
        CHECK_INPUT(isos.value());
        TORCH_CHECK(isos.value().scalar_type() == torch::kFloat32, "isos must be float32");
        TORCH_CHECK(isos.value().size(0) == num_gaussians, "isos must have shape (N,)");
        isos_ptr = isos.value().data_ptr<float>();
    }

    auto options = means.options();
    torch::Tensor hit_mask = torch::zeros({num_edges}, options.dtype(torch::kBool));
    torch::Tensor out_gaus_ids = torch::full({num_edges}, -1, options.dtype(torch::kInt64));

    gs_aabb::query_gs_edge_intersection_brute_force(
        num_edges,
        num_gaussians,
        reinterpret_cast<const float3 *>(edge_starts.data_ptr<float>()),
        reinterpret_cast<const float3 *>(edge_ends.data_ptr<float>()),
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float *>(opacities.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        isos_ptr,
        iso,
        reinterpret_cast<bool *>(hit_mask.data_ptr<bool>()),
        reinterpret_cast<int64_t *>(out_gaus_ids.data_ptr<int64_t>()));

    return std::make_tuple(hit_mask, out_gaus_ids);
}

void bind_primitive_gs(py::module_ &m)
{
    m.def("compute_gs_covi_func", &compute_gs_covi_wrapper, "Compute covariance matrices for 3D Gaussians",
          py::arg("means"),
          py::arg("rotations"),
          py::arg("scales"),
          py::arg("rotnorm"),
          py::arg("tol"),
          py::arg("level"));
    m.def("compute_gs_aabb_func", &compute_gs_aabb_wrapper, "Compute AABB for 3D Gaussians",
          py::arg("means"),
          py::arg("scales"),
          py::arg("covis"),
          py::arg("isos"),
          py::arg("iso"),
          py::arg("tol"),
          py::arg("level"));
    m.def("compute_gs_aabb_w_covi_func", &compute_gs_aabb_w_covi_wrapper, "Compute AABB for 3D Gaussians",
          py::arg("means"),
          py::arg("rotations"),
          py::arg("scales"),
          py::arg("isos"),
          py::arg("iso"),
          py::arg("tol"),
          py::arg("level"),
          py::arg("rotnorm") = false);
    m.def("query_gs_voxel_pair_intersection_brute_force", &query_gs_voxel_pair_intersection_brute_force_wrapper, "Query Gaussian-Voxel PairIntersections (Brute Force)",
          py::arg("vx_aabb_mins"),
          py::arg("vx_aabb_maxs"),
          py::arg("means"),
          py::arg("covis"),
          py::arg("opacities"),
          py::arg("gs_aabb_mins"),
          py::arg("gs_aabb_maxs"),
          py::arg("contact_points"),
          py::arg("isos"),
          py::arg("iso"),
          py::arg("ar_threshold"),
          py::arg("p_threshold"),
          py::arg("return_centroids") = false,
          py::arg("max_capacity") = 10000000);
    m.def("query_gs_edge_pair_intersection_brute_force", &query_gs_edge_pair_intersection_brute_force_wrapper, "Query Gaussian-Edge Pair Intersections (Brute Force)",
          py::arg("edge_starts"),
          py::arg("edge_ends"),
          py::arg("means"),
          py::arg("covis"),
          py::arg("isos"),
          py::arg("iso"),
          py::arg("max_capacity") = 10000000);
    m.def("query_gs_edge_intersection_brute_force", &query_gs_edge_intersection_brute_force_wrapper, "Query Gaussian-Edge Intersections (Brute Force)",
          py::arg("edge_starts"),
          py::arg("edge_ends"),
          py::arg("means"),
          py::arg("opacities"),
          py::arg("covis"),
          py::arg("isos"),
          py::arg("iso"));
}