/**
 * @file superquadric.h
 * @brief GPU dispatchers for tessellating sets of superquadric primitives into triangle meshes.
 */

#ifndef SUPERQUADRIC_H
#define SUPERQUADRIC_H

#include "../maths/maths.h"
#include "../constants.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace sq
{
    /**
     * @brief One frame of the divide-and-conquer arc-length sampler's explicit stack.
     * @details Held in global scratch rather than thread-local storage: the traversal depth
     * grows with the sample count, and a per-thread array large enough for the worst case
     * would spill anyway. One frame owns an angular interval whose endpoints are already
     * placed, the number of interior samples still to distribute inside it, and the offset
     * in the output array where that block of samples begins.
     */
    struct SuperellipseFrame
    {
        double point_a_x; /**< First coordinate of the superellipse point at `theta_a`. */
        double point_a_y; /**< Second coordinate of the superellipse point at `theta_a`. */
        double point_b_x; /**< First coordinate of the superellipse point at `theta_b`. */
        double point_b_y; /**< Second coordinate of the superellipse point at `theta_b`. */
        double theta_a;   /**< Angle at the start of the interval. */
        double theta_b;   /**< Angle at the end of the interval. */
        int budget;       /**< Interior samples still to place inside this interval. */
        int offset;       /**< Index in the output array where this interval's block begins. */
    };

    /**
     * @brief Samples every primitive's two superellipses at approximately equal arc length.
     */
    __host__ void compute_superellipse_angles(const uint32_t num_quadrics, const uint32_t resolution,
                                              const float3 *__restrict__ scales, const float2 *__restrict__ exponents,
                                              SuperellipseFrame *__restrict__ stack_scratch,
                                              double *__restrict__ out_azimuths, double *__restrict__ out_polars);

    /**
     * @brief Evaluates the parametric superquadric surface at every tessellation vertex.
     */
    __host__ void compute_superquadric_vertices(const uint32_t num_quadrics, const uint32_t resolution,
                                                const float3 *__restrict__ scales, const float2 *__restrict__ exponents,
                                                const float *__restrict__ rotations,
                                                const float3 *__restrict__ translations,
                                                const double *__restrict__ azimuths, const double *__restrict__ polars,
                                                const bool return_labels, float3 *__restrict__ out_vertices,
                                                int32_t *__restrict__ out_labels);

    /**
     * @brief Writes the triangle connectivity shared by every tessellated primitive.
     */
    __host__ void compute_superquadric_faces(const uint32_t num_quadrics, const uint32_t resolution,
                                             int3 *__restrict__ out_triangles);
} // namespace sq

#endif // SUPERQUADRIC_H
