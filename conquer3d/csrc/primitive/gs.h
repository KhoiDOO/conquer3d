/**
 * @file gs.h
 * @brief GPU mathematical kernels and BVH intersection dispatchers for 3D Gaussian Splatting (3DGS).
 */

#ifndef GS_H
#define GS_H

#include "../maths/maths.h"
#include "../constants.h"
#include "../data_structure/kdtree.h"
#include "gs_math.cuh"
#include "aabb.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace gs
{
    /**
     * @brief Computes inverse covariance matrices for 3D Gaussians on GPU.
     */
    __host__ void compute_gs_covi(
        const uint32_t num_gaussians,
        const float4 *__restrict__ rotations,
        const float3 *__restrict__ scales,
        const bool rotnorm,
        const float tol,
        const uint32_t level,
        float *__restrict__ covis);

    /**
     * @brief Evaluates Mahalanobis radii for 3D Gaussians using k-NN KD-Tree on GPU.
     */
    __host__ void solve_gs_neighbor_mahalanobis_radius(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const int k,
        float *__restrict__ isos);
}

namespace gs_aabb
{
    /**
     * @brief Computes tight 3D bounding boxes for confidence ellipsoids of 3D Gaussians.
     */
    __host__ void compute_gs_aabb(
        const uint32_t num_gaussians,
        const float3 *__restrict__ means,
        const float3 *__restrict__ scales,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        const float tol,
        const uint32_t level,
        float3 *__restrict__ aabb_min,
        float3 *__restrict__ aabb_max,
        float3 *__restrict__ contact_points);

    /**
     * @brief BVH-accelerated Gaussian-voxel intersection queries on GPU.
     */
    __host__ void query_gs_voxel_pair_intersection_bvh(
        const uint32_t num_voxels,
        const uint32_t num_gaussians,
        const float3 *__restrict__ vx_aabb_mins,
        const float3 *__restrict__ vx_aabb_maxs,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float *__restrict__ opacities,
        const float3 *__restrict__ gs_aabb_mins,
        const float3 *__restrict__ gs_aabb_maxs,
        const float3 *__restrict__ contact_points,
        const float *__restrict__ isos,
        const float iso,
        const float ar_threshold,
        const float p_threshold,
        const bool return_centroids,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_voxel_ids,
        int64_t *__restrict__ out_gaus_ids,
        float3 *__restrict__ centroids,
        float *__restrict__ densities,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity);

    /**
     * @brief BVH-accelerated Gaussian-edge segment pair intersection queries.
     */
    __host__ void query_gs_edge_pair_intersection_bvh(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_edge_ids,
        int64_t *__restrict__ out_gaus_ids,
        int64_t *__restrict__ global_counter,
        const int64_t max_capacity);

    /**
     * @brief BVH-accelerated Gaussian-edge segment proximity and density queries.
     */
    __host__ void query_gs_edge_intersection_bvh(
        const uint32_t num_edges,
        const uint32_t num_gaussians,
        const float3 *__restrict__ edge_starts,
        const float3 *__restrict__ edge_ends,
        const float3 *__restrict__ bvh_aabb_mins,
        const float3 *__restrict__ bvh_aabb_maxs,
        const int2 *__restrict__ bvh_children,
        const int *__restrict__ object_ids,
        const float3 *__restrict__ means,
        const float *__restrict__ opacities,
        const float *__restrict__ covis,
        const float *__restrict__ isos,
        const float iso,
        bool *__restrict__ hit_mask,
        int64_t *__restrict__ out_gaus_ids);
}

#endif // GS_H