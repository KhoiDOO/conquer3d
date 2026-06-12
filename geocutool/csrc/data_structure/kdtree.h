#ifndef KDTREE_H
#define KDTREE_H

#include "../base.h"

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
}

#endif // KDTREE_H
