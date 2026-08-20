/**
 * @file kdtree.h
 * @brief GPU-accelerated 3D KD-Tree for fast k-Nearest Neighbor (k-NN) queries.
 */

#ifndef KDTREE_H
#define KDTREE_H

#include "../maths/maths.h"
#include "../constants.h"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <tuple>

/**
 * @brief High-level KD-Tree wrapper class exposing PyTorch tensor interfaces.
 */
class KDTree
{
private:
    uint32_t num_points;

    torch::Tensor points;
    torch::Tensor oinds;

public:
    /**
     * @brief Constructs and builds a KD-Tree over the input point cloud.
     * 
     * @param[in] points (N, 3) float32 point coordinates on CUDA device.
     */
    KDTree(const torch::Tensor &points);

    /**
     * @brief Queries the k nearest neighbors for a batch of query points.
     * 
     * @param[in] query_points  (M, 3) float32 query point coordinates on CUDA.
     * @param[in] k             Number of nearest neighbors to find per query point.
     * @param[in] exclude_self  If true, ignores distance to identical point index.
     * 
     * @return std::tuple containing:
     *         - distances (torch.Tensor): (M, k) float32 squared Euclidean distances.
     *         - indices (torch.Tensor): (M, k) int64 indices of nearest neighbors in original point set.
     */
    std::tuple<torch::Tensor, torch::Tensor> query(
        const torch::Tensor &query_points,
        const int k,
        bool exclude_self = false);
};

namespace kdtree
{
    /**
     * @brief Builds a complete binary KD-Tree in device memory using iterative median split.
     * 
     * @param[in]     num_points    Number of points in the tree ($N$).
     * @param[in,out] points        Array of (N, 3) float32 point coordinates permuted into tree order.
     * @param[in,out] original_inds Array of original indices tracking point permutations.
     */
    void build(
        const uint32_t num_points,
        float3 *__restrict__ points,
        int64_t *__restrict__ original_inds);

    /**
     * @brief Parallel batch k-NN query kernel dispatcher on GPU.
     * 
     * @param[in]  num_queries   Number of query points ($M$).
     * @param[in]  num_points    Number of reference points in the tree ($N$).
     * @param[in]  k             Number of nearest neighbors ($k$).
     * @param[in]  query_points  Device array of (M, 3) query coordinates.
     * @param[in]  tree_points   Device array of (N, 3) KD-tree ordered points.
     * @param[in]  tree_inds     Device array of original point indices.
     * @param[out] out_dists     Output buffer of size $M \times k$ for squared distances.
     * @param[out] out_inds      Output buffer of size $M \times k$ for nearest neighbor indices.
     */
    void query(
        const uint32_t num_queries,
        const uint32_t num_points,
        const uint32_t k,
        const float3 *__restrict__ query_points,
        const float3 *__restrict__ tree_points,
        const int64_t *__restrict__ tree_inds,
        float *__restrict__ out_dists,
        int64_t *__restrict__ out_inds);

    /**
     * @brief Inserts a distance and index pair into a sorted fixed-size priority queue.
     * 
     * @param[in]     dist       Distance to insert.
     * @param[in]     id         Point index to insert.
     * @param[in,out] best_dists Sorted array of best distances of size $k$.
     * @param[in,out] best_inds  Sorted array of best indices of size $k$.
     * @param[in]     k          Priority queue capacity.
     */
    __device__ __forceinline__ void push_pq(
        float dist,
        int64_t id,
        float *best_dists,
        int64_t *best_inds,
        const int k)
    {
        if (dist >= best_dists[k - 1])
            return;
        int i = k - 2;
        while (i >= 0 && best_dists[i] > dist)
        {
            best_dists[i + 1] = best_dists[i];
            best_inds[i + 1] = best_inds[i];
            i--;
        }
        best_dists[i + 1] = dist;
        best_inds[i + 1] = id;
    }

    /**
     * @brief Performs a stack-based non-recursive KD-tree traversal for a single query point.
     * 
     * @param[in]     qp          Query point coordinate float3.
     * @param[in]     num_points  Total points in the KD-tree.
     * @param[in]     tree_points Permuted KD-tree points array.
     * @param[in]     tree_inds   Original index tracking array.
     * @param[in]     k           Number of nearest neighbors to retrieve.
     * @param[in,out] best_dists  Local priority queue of best distances.
     * @param[in,out] best_inds   Local priority queue of best indices.
     */
    __device__ __forceinline__ void query_kdtree_loop(
        const float3 &qp,
        const uint32_t num_points,
        const float3 *__restrict__ tree_points,
        const int64_t *__restrict__ tree_inds,
        const int k,
        float *best_dists,
        int64_t *best_inds)
    {
        int stack[64];
        int stack_ptr = 0;

        stack[stack_ptr++] = 0;

        while (stack_ptr > 0)
        {
            int curr = stack[--stack_ptr];
            if (curr >= num_points)
                continue;

            float3 p = tree_points[curr];

#ifdef __CUDA_ARCH__
            int axis = (31 - __clz(curr + 1)) % 3;
#else
            int axis = (31 - __builtin_clz((unsigned int)(curr + 1))) % 3;
#endif

            float3 d = qp - p;
            float dist_sq = maths::dot(d, d);

            if (isfinite(dist_sq))
            {
                push_pq(dist_sq, tree_inds[curr], best_dists, best_inds, k);
            }

            float axis_dist = (axis == 0) ? d.x : ((axis == 1) ? d.y : d.z);

            int left_child = 2 * curr + 1;
            int right_child = 2 * curr + 2;

            int near_child = (axis_dist <= 0) ? left_child : right_child;
            int far_child = (axis_dist <= 0) ? right_child : left_child;

            if (far_child < num_points && (axis_dist * axis_dist <= best_dists[k - 1]))
            {
                stack[stack_ptr++] = far_child;
            }
            if (near_child < num_points)
            {
                stack[stack_ptr++] = near_child;
            }
        }
    }
}

#endif // KDTREE_H
