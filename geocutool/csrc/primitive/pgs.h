#ifndef PGS_H
#define PGS_H

#include "../maths/maths.h"
#include "../constants.h"
#include "../data_structure/kdtree.h"
#include "gs_math.cuh"
#include "aabb.h"
#include <cuda_runtime.h>

namespace pgs
{
    __host__ void solve_pgs_cluster_tangency_radius(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ covis,
        const int k,
        float *__restrict__ isos,
        bool *__restrict__ invalid_mask
    );
}

namespace pgs_aabb
{
    __host__ void query_pgs_voxel_pair_intersection_brute_force(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ covis,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
        const float *__restrict__ isos,
        const float iso,
        const bool return_centroids,
        const bool return_centroid_densities,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_voxel_ids,
        int64_t *__restrict__ out_gaus_ids,
        float3 *__restrict__ centroids,
        float *__restrict__ densities,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity);
    
    __host__ void query_pgs_edge_intersection_brute_force(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids);

    __host__ void query_pgs_voxel_pair_intersection_bvh(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ covis,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
        const float *__restrict__ isos,
        const float iso,
        const bool return_centroids,
        const bool return_centroid_densities,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_voxel_ids,
        int64_t *__restrict__ out_gaus_ids,
        float3 *__restrict__ centroids,
        float *__restrict__ densities,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity);
    
    __host__ void query_pgs_edge_intersection_bvh(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float3 *__restrict__ normals,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids);
}

#endif // PGS_H