#include <torch/extension.h>
#include "../../data_structure/bvh.h"
#include "../../check.h"

#include <pybind11/pybind11.h>
#include <optional>
#include <vector>

namespace py = pybind11;

BVH::BVH(
    const torch::Tensor &in_aabb_mins,
    const torch::Tensor &in_aabb_maxs)
{
    CHECK_INPUT(in_aabb_mins);
    CHECK_INPUT(in_aabb_maxs);
    TORCH_CHECK(in_aabb_mins.scalar_type() == torch::kFloat32, "aabb_mins must be float32");
    TORCH_CHECK(in_aabb_maxs.scalar_type() == torch::kFloat32, "aabb_maxs must be float32");
    TORCH_CHECK(in_aabb_mins.size(1) == 3, "aabb_mins must have shape (N, 3)");
    TORCH_CHECK(in_aabb_maxs.size(1) == 3, "aabb_maxs must have shape (N, 3)");

    this->num_objects = static_cast<uint32_t>(in_aabb_mins.size(0));

    if (this->num_objects == 0)
    {
        throw std::runtime_error("Cannot build BVH with 0 objects.");
    }

    if (this->num_objects == 1)
    {
        this->num_nodes = 1;

        auto options_int32 = in_aabb_mins.options().dtype(torch::kInt32);

        // The Root is the only Leaf.
        this->aabb_mins = in_aabb_mins.clone();
        this->aabb_maxs = in_aabb_maxs.clone();

        // No children (-1 means null pointer)
        this->bvh_children = torch::full({1, 2}, -1, options_int32);

        // The object ID is just index 0
        this->object_ids = torch::zeros({1}, options_int32);

        // Bypass the entire Karras CUDA build!
        return;
    }

    // A Binary Tree with N leaves always has 2N - 1 total nodes!
    this->num_nodes = 2 * this->num_objects - 1;
    auto options = in_aabb_mins.options();

    // Allocate the full flat arrays
    this->aabb_mins = torch::empty({this->num_nodes, 3}, options);
    this->aabb_maxs = torch::empty({this->num_nodes, 3}, options);
    this->bvh_children = torch::empty({this->num_nodes, 2}, options.dtype(torch::kInt32));
    this->bvh_parents = torch::empty({this->num_nodes}, options.dtype(torch::kInt32));
    this->object_ids = torch::empty({this->num_objects}, options.dtype(torch::kInt32));

    bvh::build(
        this->num_objects,
        this->num_nodes,
        reinterpret_cast<const float3 *>(in_aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(in_aabb_maxs.data_ptr<float>()),
        reinterpret_cast<float3 *>(this->aabb_mins.data_ptr<float>()),
        reinterpret_cast<float3 *>(this->aabb_maxs.data_ptr<float>()),
        reinterpret_cast<int2 *>(this->bvh_children.data_ptr<int>()),
        reinterpret_cast<int *>(this->bvh_parents.data_ptr<int>()),
        reinterpret_cast<int *>(this->object_ids.data_ptr<int>()));
}

std::tuple<torch::Tensor, torch::Tensor> BVH::query(
    const torch::Tensor &query_aabb_mins,
    const torch::Tensor &query_aabb_maxs)
{
    CHECK_INPUT(query_aabb_mins);
    CHECK_INPUT(query_aabb_maxs);
    TORCH_CHECK(query_aabb_mins.scalar_type() == torch::kFloat32, "query_aabb_mins must be float32");
    TORCH_CHECK(query_aabb_maxs.scalar_type() == torch::kFloat32, "query_aabb_maxs must be float32");
    TORCH_CHECK(query_aabb_mins.size(1) == 3, "query_aabb_mins must have shape (M, 3)");
    TORCH_CHECK(query_aabb_maxs.size(1) == 3, "query_aabb_maxs must have shape (M, 3)");

    const uint32_t num_queries = static_cast<uint32_t>(query_aabb_mins.size(0));

    auto options_int64 = query_aabb_mins.options().dtype(torch::kInt64);

    if (num_queries == 0)
    {
        throw std::runtime_error("Cannot query BVH with 0 queries.");
    }

    torch::Tensor out_query_ids = torch::empty({BVH_MAX_CAPACITY}, options_int64);
    torch::Tensor out_object_ids = torch::empty({BVH_MAX_CAPACITY}, options_int64);

    torch::Tensor hit_counter = torch::zeros({1}, options_int64);

    bvh::query(
        num_queries,
        this->num_objects,
        reinterpret_cast<const float3 *>(query_aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(query_aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const float3 *>(this->aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(this->aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const int2 *>(this->bvh_children.data_ptr<int>()),
        reinterpret_cast<const int *>(this->object_ids.data_ptr<int>()),
        reinterpret_cast<int64_t *>(out_query_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(out_object_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(hit_counter.data_ptr<int64_t>()),
        static_cast<int64_t>(BVH_MAX_CAPACITY));

    int64_t num_hits = hit_counter.item<int64_t>();

    num_hits = std::min(num_hits, static_cast<int64_t>(BVH_MAX_CAPACITY));

    return std::make_tuple(
        out_query_ids.slice(0, 0, num_hits),
        out_object_ids.slice(0, 0, num_hits));
}

std::tuple<torch::Tensor, torch::Tensor> BVH::query_self()
{
    auto options_int64 = this->aabb_mins.options().dtype(torch::kInt64);

    if (this->num_objects == 0)
    {
        throw std::runtime_error("Cannot query BVH with 0 objects.");
    }

    torch::Tensor out_query_ids = torch::empty({BVH_MAX_CAPACITY}, options_int64);
    torch::Tensor out_object_ids = torch::empty({BVH_MAX_CAPACITY}, options_int64);
    torch::Tensor hit_counter = torch::zeros({1}, options_int64);

    bvh::query_self(
        this->num_objects,
        reinterpret_cast<const float3 *>(this->aabb_mins.data_ptr<float>()),
        reinterpret_cast<const float3 *>(this->aabb_maxs.data_ptr<float>()),
        reinterpret_cast<const int2 *>(this->bvh_children.data_ptr<int>()),
        reinterpret_cast<const int *>(this->object_ids.data_ptr<int>()),
        reinterpret_cast<int64_t *>(out_query_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(out_object_ids.data_ptr<int64_t>()),
        reinterpret_cast<int64_t *>(hit_counter.data_ptr<int64_t>()),
        static_cast<int64_t>(BVH_MAX_CAPACITY));

    int64_t num_hits = hit_counter.item<int64_t>();
    num_hits = std::min(num_hits, static_cast<int64_t>(BVH_MAX_CAPACITY));

    return std::make_tuple(
        out_query_ids.slice(0, 0, num_hits),
        out_object_ids.slice(0, 0, num_hits));
}

std::tuple<torch::Tensor, torch::Tensor> BVH::query_ray(
    const torch::Tensor &ray_origins,
    const torch::Tensor &ray_dirs,
    int64_t max_capacity)
{
    CHECK_INPUT(ray_origins);
    CHECK_INPUT(ray_dirs);
    TORCH_CHECK(ray_origins.size(0) == ray_dirs.size(0), "ray_origins and ray_dirs must have the same number of rays");

    int num_queries = ray_origins.size(0);

    auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(ray_origins.device());
    torch::Tensor out_query_ids = torch::empty({max_capacity}, options_i64);
    torch::Tensor out_object_ids = torch::empty({max_capacity}, options_i64);
    torch::Tensor hit_counter = torch::zeros({1}, options_i64);

    bvh::query_ray(
        num_queries,
        this->num_objects,
        (const float3 *)ray_origins.data_ptr<float>(),
        (const float3 *)ray_dirs.data_ptr<float>(),
        (const float3 *)this->aabb_mins.data_ptr<float>(),
        (const float3 *)this->aabb_maxs.data_ptr<float>(),
        (const int2 *)this->bvh_children.data_ptr<int>(),
        this->object_ids.data_ptr<int>(),
        out_query_ids.data_ptr<int64_t>(),
        out_object_ids.data_ptr<int64_t>(),
        hit_counter.data_ptr<int64_t>(),
        max_capacity);

    int64_t h_hit_counter = hit_counter.item<int64_t>();

    if (h_hit_counter >= max_capacity)
    {
        TORCH_WARN("BVH ray query capacity exceeded. Some hits were dropped.");
        h_hit_counter = max_capacity;
    }

    if (h_hit_counter == 0)
    {
        return std::make_tuple(
            torch::empty({0}, options_i64),
            torch::empty({0}, options_i64));
    }

    return std::make_tuple(
        out_query_ids.slice(0, 0, h_hit_counter),
        out_object_ids.slice(0, 0, h_hit_counter));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BVH::query_point(
    const torch::Tensor &query_points)
{
    CHECK_INPUT(query_points);

    int num_queries = query_points.size(0);

    auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(query_points.device());
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(query_points.device());

    torch::Tensor out_query_ids = torch::empty({num_queries}, options_i64);
    torch::Tensor out_object_ids = torch::empty({num_queries}, options_i64);
    torch::Tensor out_distances = torch::empty({num_queries}, options_f32);

    bvh::query_point(
        num_queries,
        this->num_objects,
        (const float3 *)query_points.data_ptr<float>(),
        (const float3 *)this->aabb_mins.data_ptr<float>(),
        (const float3 *)this->aabb_maxs.data_ptr<float>(),
        (const int2 *)this->bvh_children.data_ptr<int>(),
        this->object_ids.data_ptr<int>(),
        out_query_ids.data_ptr<int64_t>(),
        out_object_ids.data_ptr<int64_t>(),
        out_distances.data_ptr<float>());

    return std::make_tuple(out_query_ids, out_object_ids, out_distances);
}

void bind_ds_bvh(py::module_ &m)
{
    py::class_<BVH>(m, "BVH", R"doc(
        A highly efficient Bounding Volume Hierarchy (LBVH) data structure natively backed by CUDA.
        )doc")
        .def(py::init<const torch::Tensor &, const torch::Tensor &>(),
             py::arg("in_aabb_mins"),
             py::arg("in_aabb_maxs"),
             R"doc(
        Construct and build the Karras LBVH from Gaussian AABBs.

        Args:
            in_aabb_mins (torch.Tensor): Shape (N, 3) float32 tensor of AABB minimum coordinates.
            in_aabb_maxs (torch.Tensor): Shape (N, 3) float32 tensor of AABB maximum coordinates.
        )doc")

        .def("query", &BVH::query,
             py::arg("query_aabb_mins"),
             py::arg("query_aabb_maxs"),
             R"doc(
        Query the BVH with bounding boxes or segments.

        Args:
            query_aabb_mins (torch.Tensor): Shape (M, 3) float32 tensor of query AABB minimum coordinates.
            query_aabb_maxs (torch.Tensor): Shape (M, 3) float32 tensor of query AABB maximum coordinates.

        Returns:
            tuple: (query_ids, object_ids)
        )doc")

        .def("query_self", &BVH::query_self,
             R"doc(
        Query the BVH against itself.

        Returns:
            tuple: (query_ids, object_ids) representing unique overlapping pairs.
        )doc")

        .def("query_ray", &BVH::query_ray,
             py::arg("ray_origins"),
             py::arg("ray_dirs"),
             py::arg("max_capacity") = BVH_MAX_CAPACITY,
             R"doc(
        Find all ray-AABB intersections.

        Args:
            ray_origins (torch.Tensor): Shape (M, 3) float32 tensor of ray origins.
            ray_dirs (torch.Tensor): Shape (M, 3) float32 tensor of ray directions.
            max_capacity (int): Maximum global capacity for recording intersections. Defaults to BVH_MAX_CAPACITY.

        Returns:
            tuple: (ray_ids, object_ids)
        )doc")
        .def("query_point", &BVH::query_point,
             py::arg("query_points"),
             R"doc(
        Find the closest leaf AABB to each query point.

        Args:
            query_points (torch.Tensor): Shape (M, 3) float32 tensor of query points.

        Returns:
            tuple: (query_ids, object_ids, distances)
        )doc");
}