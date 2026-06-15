#ifndef KDTREE_H
#define KDTREE_H

#include "../maths/maths.h"
#include "../constants.h"

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

class KDTree {
private:
    uint32_t num_points;
    
    torch::Tensor points;
    torch::Tensor oinds;

public:
    KDTree(const torch::Tensor& points);

    std::tuple<torch::Tensor, torch::Tensor> query(
        const torch::Tensor& query_points, 
        const int k,
        bool exclude_self=false
    );
};

namespace kdtree {
    void build(
        const uint32_t num_points,
        float3 *__restrict__ points,
        int64_t *__restrict__ original_inds);

    void query(
        const uint32_t num_queries,
        const uint32_t num_points,
        const uint32_t k,
        const float3 *__restrict__ query_points,
        const float3 *__restrict__ tree_points,
        const int64_t *__restrict__ tree_inds,
        float *__restrict__ out_dists,
        int64_t *__restrict__ out_inds);
    
    __device__ __forceinline__ void push_pq(
        float dist, 
        int64_t id, 
        float* best_dists, 
        int64_t* best_inds, 
        const int k)
    {
        if (dist >= best_dists[k - 1]) return;
        int i = k - 2;
        while (i >= 0 && best_dists[i] > dist) {
            best_dists[i + 1] = best_dists[i];
            best_inds[i + 1]  = best_inds[i];
            i--;
        }
        best_dists[i + 1] = dist;
        best_inds[i + 1]  = id;
    }

    __device__ __forceinline__ void query_kdtree_loop(
        const float3& qp,
        const uint32_t num_points,
        const float3* __restrict__ tree_points,
        const int64_t* __restrict__ tree_inds,
        const int k,
        float* best_dists,
        int64_t* best_inds)
    {
        int stack[64];
        int stack_ptr = 0;

        stack[stack_ptr++] = 0;

        while (stack_ptr > 0) {
            int curr = stack[--stack_ptr];
            if (curr >= num_points) continue;
            
            float3 p = tree_points[curr];
            
            #ifdef __CUDA_ARCH__
                int axis = (31 - __clz(curr + 1)) % 3;
            #else
                int axis = (31 - __builtin_clz((unsigned int)(curr + 1))) % 3;
            #endif

            float3 d = qp - p;
            float dist_sq = maths::dot(d, d);

            push_pq(dist_sq, tree_inds[curr], best_dists, best_inds, k);

            float axis_dist = (axis == 0) ? d.x : ((axis == 1) ? d.y : d.z);

            int left_child = 2 * curr + 1;
            int right_child = 2 * curr + 2;

            int near_child = (axis_dist <= 0) ? left_child : right_child;
            int far_child  = (axis_dist <= 0) ? right_child : left_child;

            if (far_child < num_points && (axis_dist * axis_dist <= best_dists[k - 1])) {
                stack[stack_ptr++] = far_child;
            }
            if (near_child < num_points) {
                stack[stack_ptr++] = near_child;
            }
        }
    }
}

#endif // KDTREE_H
