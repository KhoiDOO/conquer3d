#include <torch/extension.h>
#include "../../data_structure/pgs_bvh.h"
#include "../../check.h"

#include <pybind11/pybind11.h>
#include <optional>
#include <vector>

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> PGSBVH::query_voxel_pair(
    const torch::Tensor &vx_aabb_mins,
    const torch::Tensor &vx_aabb_maxs,
    const torch::Tensor &means,
    const torch::Tensor &normals,
    const torch::Tensor &covis,
    const torch::Tensor &gs_aabb_mins,
    const torch::Tensor &gs_aabb_maxs,
    const std::optional<torch::Tensor> &isos,
    const float iso,
    const bool return_centroids,
    const bool return_centroid_densities,
    const int64_t max_capacity)
{
    CHECK_INPUT(vx_aabb_mins);
    CHECK_INPUT(vx_aabb_maxs);
    CHECK_INPUT(means);
    CHECK_INPUT(normals);
    CHECK_INPUT(covis);
    CHECK_INPUT(gs_aabb_mins);
    CHECK_INPUT(gs_aabb_maxs);

    TORCH_CHECK(vx_aabb_mins.scalar_type() == torch::kFloat32, "vx_aabb_mins must be float32");
    TORCH_CHECK(vx_aabb_maxs.scalar_type() == torch::kFloat32, "vx_aabb_maxs must be float32");
    TORCH_CHECK(vx_aabb_mins.size(1) == 3, "vx_aabb_mins must have shape (M, 3)");
    TORCH_CHECK(vx_aabb_maxs.size(1) == 3, "vx_aabb_maxs must have shape (M, 3)");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(normals.scalar_type() == torch::kFloat32, "normals must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");
    TORCH_CHECK(gs_aabb_mins.scalar_type() == torch::kFloat32, "gs_aabb_mins must be float32");
    TORCH_CHECK(gs_aabb_maxs.scalar_type() == torch::kFloat32, "gs_aabb_maxs must be float32");

    const uint32_t num_voxels = static_cast<uint32_t>(vx_aabb_mins.size(0));
    const uint32_t num_gaussians = means.size(0);

    if (num_voxels == 0)
    {
        auto options_int64 = vx_aabb_mins.options().dtype(torch::kInt64);
        auto options_float32 = vx_aabb_mins.options().dtype(torch::kFloat32);

        return std::make_tuple(
            torch::empty({0}, options_int64.dtype(torch::kBool)), 
            torch::empty({0}, options_int64),                     
            torch::empty({0}, options_int64),                     
            return_centroids ? std::make_optional(torch::empty({0, 3}, options_float32)) : std::nullopt,
            (return_centroids && return_centroid_densities) ? std::make_optional(torch::empty({0}, options_float32)) : std::nullopt);
    }

    TORCH_CHECK(num_gaussians == this->num_objects, "Number of PGS must match the BVH");

    auto options_int64 = vx_aabb_mins.options().dtype(torch::kInt64);
    auto options_float32 = vx_aabb_mins.options().dtype(torch::kFloat32);

    float *isos_ptr = nullptr;
    if (isos.has_value())
    {
        CHECK_INPUT(isos.value());
        TORCH_CHECK(isos.value().scalar_type() == torch::kFloat32, "isos must be float32");
        TORCH_CHECK(isos.value().size(0) == num_gaussians, "isos must have shape (N,)");
        isos_ptr = isos.value().data_ptr<float>();
    }

    torch::Tensor out_voxel_ids = torch::empty({BVH_MAX_CAPACITY}, options_int64);
    torch::Tensor out_gaus_ids = torch::empty({BVH_MAX_CAPACITY}, options_int64);
    torch::Tensor centroids = torch::empty({BVH_MAX_CAPACITY, 3}, options_float32);
    torch::Tensor densities = torch::empty({BVH_MAX_CAPACITY}, options_float32);

    torch::Tensor hit_counter = torch::zeros({1}, options_int64);
    torch::Tensor hit_mask = torch::zeros({num_voxels}, options_int64.dtype(torch::kBool));

    pgs_aabb::query_pgs_voxel_pair_intersection_bvh(
        num_voxels,
        num_gaussians,
        reinterpret_cast<const float3 *>(vx_aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(vx_aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const float3 *>(this->aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(this->aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const int2 *>(this->bvh_children.data_ptr<int>()),
        reinterpret_cast<const int *>(this->object_ids.data_ptr<int>()),
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float3 *>(normals.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        reinterpret_cast<const float3 *>(gs_aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(gs_aabb_maxs.data_ptr<float>()),
        isos_ptr,
        iso,
        return_centroids,
        return_centroid_densities,
        reinterpret_cast<bool *>(hit_mask.data_ptr<bool>()),
        reinterpret_cast<int64_t *>(out_voxel_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(out_gaus_ids.data_ptr<int64_t>()),
        reinterpret_cast<float3 *>(centroids.data_ptr<float>()),
        reinterpret_cast<float *>(densities.data_ptr<float>()),
        reinterpret_cast<int64_t *>(hit_counter.data_ptr<int64_t>()),
        max_capacity);

    int64_t num_intersections = hit_counter.item<int64_t>();
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
            return_centroid_densities ? std::make_optional(densities.slice(0, 0, valid_hits)) : std::nullopt);
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

std::tuple<torch::Tensor, torch::Tensor> PGSBVH::query_edge(
    const torch::Tensor &edge_starts,
    const torch::Tensor &edge_ends,
    const torch::Tensor &means,
    const torch::Tensor &normals,
    const torch::Tensor &opacities,
    const torch::Tensor &covis,
    const std::optional<torch::Tensor> &isos,
    const float iso)
{
    CHECK_INPUT(edge_starts);
    CHECK_INPUT(edge_ends);
    CHECK_INPUT(means);
    CHECK_INPUT(normals);
    CHECK_INPUT(opacities);
    CHECK_INPUT(covis);

    TORCH_CHECK(edge_starts.scalar_type() == torch::kFloat32, "edge_starts must be float32");
    TORCH_CHECK(edge_ends.scalar_type() == torch::kFloat32, "edge_ends must be float32");
    TORCH_CHECK(edge_starts.size(1) == 3, "edge_starts must have shape (M, 3)");
    TORCH_CHECK(edge_ends.size(1) == 3, "edge_ends must have shape (M, 3)");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(normals.scalar_type() == torch::kFloat32, "normals must be float32");
    TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
    TORCH_CHECK(covis.scalar_type() == torch::kFloat32, "covis must be float32");

    const uint32_t num_edges = static_cast<uint32_t>(edge_starts.size(0));
    const uint32_t num_gaussians = means.size(0);

    if (num_edges == 0)
    {
        auto options_int64 = edge_starts.options().dtype(torch::kInt64);
        return std::make_tuple(
            torch::empty({0}, options_int64.dtype(torch::kBool)), 
            torch::empty({0}, options_int64)                      
        );
    }

    TORCH_CHECK(num_gaussians == this->num_objects, "Number of PGS must match the BVH");

    auto options_int64 = edge_starts.options().dtype(torch::kInt64);

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
    torch::Tensor out_gaus_ids = torch::empty({num_edges}, options.dtype(torch::kInt64));

    pgs_aabb::query_pgs_edge_intersection_bvh(
        num_edges,
        num_gaussians,
        reinterpret_cast<const float3 *>(edge_starts.data_ptr<float>()),
        reinterpret_cast<const float3 *>(edge_ends.data_ptr<float>()),
        reinterpret_cast<const float3 *>(this->aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(this->aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const int2 *>(this->bvh_children.data_ptr<int>()),
        reinterpret_cast<const int *>(this->object_ids.data_ptr<int>()),
        reinterpret_cast<const float3 *>(means.data_ptr<float>()),
        reinterpret_cast<const float3 *>(normals.data_ptr<float>()),
        reinterpret_cast<const float *>(opacities.data_ptr<float>()),
        reinterpret_cast<const float *>(covis.data_ptr<float>()),
        isos_ptr,
        iso,
        reinterpret_cast<bool *>(hit_mask.data_ptr<bool>()),
        reinterpret_cast<int64_t *>(out_gaus_ids.data_ptr<int64_t>()));

    return std::make_tuple(hit_mask, out_gaus_ids);
}

void bind_ds_pgs_bvh(py::module &m)
{
    py::class_<PGSBVH, BVH>(m, "PGSBVH")
        .def(py::init<const torch::Tensor &, const torch::Tensor &>(),
             py::arg("in_aabb_mins"),
             py::arg("in_aabb_maxs"),
             "Construct and build the Karras LBVH for Planar Gaussians.")
        .def("query_voxel_pair", &PGSBVH::query_voxel_pair,
             py::arg("vx_aabb_mins"),
             py::arg("vx_aabb_maxs"),
             py::arg("means"),
             py::arg("normals"),
             py::arg("covis"),
             py::arg("gs_aabb_mins"),
             py::arg("gs_aabb_maxs"),
             py::arg("isos") = std::nullopt,
             py::arg("iso") = 0.0f,
             py::arg("return_centroids") = false,
             py::arg("return_centroid_densities") = false,
             py::arg("max_capacity") = 10000000,
             "Perform exact Broad-to-Narrow phase intersection between Voxels and PGS.")
        .def("query_edge", &PGSBVH::query_edge,
             py::arg("edge_starts"),
             py::arg("edge_ends"),
             py::arg("means"),
             py::arg("normals"),
             py::arg("opacities"),
             py::arg("covis"),
             py::arg("isos") = std::nullopt,
             py::arg("iso") = 0.0f,
             "Find the single highest-density PGS intersected by each line segment.");
}