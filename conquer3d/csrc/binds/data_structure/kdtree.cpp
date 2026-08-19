#include <torch/extension.h>
#include "../../data_structure/kdtree.h"
#include "../../check.h"

#include <pybind11/pybind11.h>
#include <optional>
#include <vector>

namespace py = pybind11;

KDTree::KDTree(
    const torch::Tensor &points)
{
    CHECK_INPUT(points);
    TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(points.size(1) == 3, "points must have shape (N, 3)");

    this->num_points = static_cast<uint32_t>(points.size(0));
    auto options = points.options();

    if (this->num_points == 0)
    {
        throw std::runtime_error("Cannot build KDTree with 0 points.");
    }

    this->points = points.clone().contiguous();
    this->oinds = torch::arange(this->num_points, options.dtype(torch::kInt64));

    kdtree::build(
        this->num_points,
        reinterpret_cast<float3 *>(this->points.data_ptr<float>()),
        this->oinds.data_ptr<int64_t>());
}

std::tuple<torch::Tensor, torch::Tensor> KDTree::query(const torch::Tensor &query_points, const int k, bool exclude_self)
{
    CHECK_INPUT(query_points);
    TORCH_CHECK(query_points.scalar_type() == torch::kFloat32, "query_points must be float32");
    TORCH_CHECK(query_points.size(1) == 3, "query_points must have shape (M, 3)");
    TORCH_CHECK(k > 0, "K must be greater than 0");

    int search_k = exclude_self ? k + 1 : k;

    TORCH_CHECK(search_k <= 32, "Requested K exceeds the hardware register limit (MAX_K = 32).");
    TORCH_CHECK(search_k <= this->num_points, "Requested K is larger than the number of points in the tree.");

    const uint32_t num_queries = static_cast<uint32_t>(query_points.size(0));
    auto options = query_points.options();

    torch::Tensor out_dists = torch::empty({num_queries, search_k}, options.dtype(torch::kFloat32));
    torch::Tensor out_inds = torch::empty({num_queries, search_k}, options.dtype(torch::kInt64));

    if (num_queries > 0 && this->num_points > 0)
    {
        // Launch the CUDA Query
        kdtree::query(
            num_queries,
            this->num_points,
            search_k,
            reinterpret_cast<const float3 *>(query_points.data_ptr<float>()),
            reinterpret_cast<const float3 *>(this->points.data_ptr<float>()),
            this->oinds.data_ptr<int64_t>(),
            out_dists.data_ptr<float>(),
            out_inds.data_ptr<int64_t>());
    }

    if (exclude_self)
    {
        out_dists = out_dists.slice(1, 1, search_k);
        out_inds = out_inds.slice(1, 1, search_k);
    }

    return std::make_tuple(out_dists, out_inds);
}

void bind_ds_kdtree(py::module_ &m) {
    py::class_<KDTree>(m, "KDTree", R"pbdoc(
        GPU-accelerated complete balanced KD-Tree with non-recursive stack-based k-NN search.

        Example:
            >>> import torch
            >>> from conquer3d._C import KDTree
            >>> tree = KDTree(points)
            >>> dists, inds = tree.query(query_pts, k=3)
        )pbdoc")
        .def(py::init<const torch::Tensor &>(),
             py::arg("points"),
             R"pbdoc(
             Builds a balanced KDTree on the GPU from 3D reference points.

             Args:
                 points (torch.Tensor): (N, 3) float32 reference coordinates on CUDA.

             Example:
                 >>> tree = KDTree(points)
             )pbdoc")
        .def("query", &KDTree::query,
             py::arg("query_points"),
             py::arg("k") = 1,
             py::arg("exclude_self") = false,
             R"pbdoc(
             Queries the KDTree for the K nearest neighbors.

             Args:
                 query_points (torch.Tensor): (M, 3) float32 query coordinates on CUDA.
                 k (int, optional): Number of nearest neighbors to search for. Defaults to 1.
                 exclude_self (bool, optional): If True, excludes the exact query point (k=0). Defaults to False.

             Returns:
                 Tuple[torch.Tensor, torch.Tensor]:
                     - distances (torch.Tensor): (M, K) float32 squared Euclidean distances.
                     - indices (torch.Tensor): (M, K) int64 indices of nearest neighbors in reference points.

             Example:
                 >>> dists, inds = tree.query(query_pts, k=4, exclude_self=True)
             )pbdoc");
}