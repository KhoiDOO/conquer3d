#include "kdtree.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>

#include <cfloat>

namespace kdtree 
{
    __device__ inline int get_left_balanced_offset(int N) {
        if (N <= 1) return 0;
        int H = 31 - __clz(N);
        int num_full_bottom = 1U << H; 
        int num_nodes_above = num_full_bottom - 1;
        int num_bottom = N - num_nodes_above;
        int left_bottom = (num_bottom < (num_full_bottom / 2)) ? num_bottom : (num_full_bottom / 2);
        int left_nodes_above = (num_nodes_above - 1) / 2;
        
        return left_nodes_above + left_bottom;
    }

    __device__ inline void get_chunk_bounds(int tag, int N, int L, int& out_start, int& out_size) {
        int path = tag + 1;
        int unsettled_start = (1U << L) - 1;
        int current_size = N;
        
        for (int d = 0; d < L; d++) {
            int bit = (path >> (L - 1 - d)) & 1;
            int left_offset = get_left_balanced_offset(current_size);
            
            if (bit == 0) {
                current_size = left_offset;
            } else {
                int settled_in_left = (1U << (L - 1 - d)) - 1;
                int unsettled_in_left = left_offset - settled_in_left;
                unsettled_start += unsettled_in_left;
                current_size = current_size - 1 - left_offset;
            }
        }
        out_start = unsettled_start;
        out_size = current_size;
    }

    struct ZipCompare {
        int axis;
        ZipCompare(int a) : axis(a) {}

        __device__ bool operator()(
            const thrust::tuple<uint32_t, float3, int64_t>& a, 
            const thrust::tuple<uint32_t, float3, int64_t>& b) const 
        {
            uint32_t tag_a = thrust::get<0>(a);
            uint32_t tag_b = thrust::get<0>(b);
            float3 pa = thrust::get<1>(a);
            float3 pb = thrust::get<1>(b);
            
            // 1. Isolate the Subtree Segments
            if (tag_a != tag_b) return tag_a < tag_b;

            // 2. Sort the coordinate inside the Segment
            if (axis == 0) return pa.x < pb.x;
            if (axis == 1) return pa.y < pb.y;
            return pa.z < pb.z;
        }
    };

    __global__ void update_tags(uint32_t* tag, int numPoints, int L) {
        int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (p_idx >= numPoints) return;
        
        // 1. Skip completely settled nodes
        int numSettled = (1U << L) - 1;
        if (p_idx < numSettled) return;

        int subtree = tag[p_idx];
        
        // 2. Compute the exact Pivot position for this subtree
        int start, size;
        get_chunk_bounds(subtree, numPoints, L, start, size);
        int pivotPos = start + get_left_balanced_offset(size);

        // 3. Mathematical mapping of Left/Right children
        if (p_idx < pivotPos)
            subtree = 2 * subtree + 1;
        else if (p_idx > pivotPos)
            subtree = 2 * subtree + 2;
        
        tag[p_idx] = subtree;
    }

    void build(
        const uint32_t num_points,
        float3 *__restrict__ points,
        int64_t *__restrict__ original_inds)
    {
        if (num_points <= 1) return;

        int numLevels = (31 - __builtin_clz(num_points)) + 1;
        int deepestLevel = numLevels - 1;

        thrust::device_vector<uint32_t> tags(num_points, 0);

        thrust::device_ptr<float3> d_points(points);
        thrust::device_ptr<int64_t> d_oinds(original_inds);

        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(tags.begin(), d_points, d_oinds));
        auto zip_end = zip_begin + num_points;

        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_points + threads - 1) / threads;

        for (int l = 0; l < deepestLevel; l++) {
            thrust::sort(thrust::device, zip_begin, zip_end, ZipCompare(l % 3));

            update_tags<<<blocks, threads>>>(
                thrust::raw_pointer_cast(tags.data()),
                num_points,
                l
            );
        }

        thrust::sort(thrust::device, zip_begin, zip_end, ZipCompare(deepestLevel % 3));
    }

    __global__ void query_kdtree_kernel(
        const uint32_t num_queries,
        const uint32_t num_points,
        const uint32_t k,
        const float3 *__restrict__ query_points,
        const float3 *__restrict__ tree_points,
        const int64_t *__restrict__ tree_inds,
        float *__restrict__ out_dists,
        int64_t *__restrict__ out_inds)
    {
        uint32_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (q_idx >= num_queries) return;

        float3 qp = query_points[q_idx];

        float best_dists[MAX_K];
        int64_t best_inds[MAX_K];
        
        #pragma unroll
        for (int i = 0; i < MAX_K; i++) {
            best_dists[i] = FLT_MAX;
            best_inds[i] = -1;
        }

        query_kdtree_loop(
            qp,
            num_points,
            tree_points,
            tree_inds,
            k,
            best_dists,
            best_inds
        );

        for (int i = 0; i < k; i++) {
            out_dists[q_idx * k + i] = best_dists[i];
            out_inds[q_idx * k + i]  = best_inds[i];
        }
    }

    void query(
        const uint32_t num_queries,
        const uint32_t num_points,
        const uint32_t k,
        const float3 *__restrict__ query_points,
        const float3 *__restrict__ tree_points,
        const int64_t *__restrict__ tree_inds,
        float *__restrict__ out_dists,
        int64_t *__restrict__ out_inds)
    {
        if (k > MAX_K) return;
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_queries + threads - 1) / threads;

        query_kdtree_kernel<<<blocks, threads>>>(
            num_queries, 
            num_points, 
            k,
            query_points, 
            tree_points, 
            tree_inds,
            out_dists, 
            out_inds
        );
    }
}