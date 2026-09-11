/**
 * @file superquadric.cu
 * @brief CUDA kernel implementations for tessellating superquadric primitive sets into meshes.
 */

#include "superquadric.h"

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace sq
{
    /**
     * @brief Raises a value to a power while preserving its sign, as $\mathrm{sgn}(x)|x|^{p}$.
     * @details The parametric form takes fractional powers of cosines and sines that go
     * negative, so a plain `pow` would return NaN; folding the sign back in keeps the surface
     * defined over the whole angular domain.
     *
     * A unit exponent short-circuits rather than going through `pow`, whose device version has
     * a two-ULP bound where the host version is exact. Exponent 1 is both the default and the
     * ellipsoid case, and the resulting asymmetry would break the exact chord ties the
     * sampler's budget split depends on.
     *
     * @param[in] value Base, of either sign.
     * @param[in] exponent Power to raise the magnitude to.
     * @return The signed power.
     */
    __device__ __forceinline__ double signed_pow(const double value, const double exponent)
    {
        if (exponent == 1.0)
            return value;
        return copysign(pow(fabs(value), exponent), value);
    }

    /**
     * @brief Evaluates a superellipse at one angle.
     * @details The two-dimensional cross-section whose arc length the sampler equalises.
     *
     * The trigonometry is symmetrised by construction -- cosine of the magnitude, sine of the
     * magnitude with the sign restored -- rather than trusting the device library to be exactly
     * even and odd. On a primitive with equal semi-axes the two halves of every interval are
     * exact mirror images, so their chords are bitwise equal and the budget splits on an exact
     * halfway case; one ULP of asymmetry breaks that tie and relays the whole subtree.
     *
     * @param[in] theta Angle to evaluate at.
     * @param[in] semi_a Semi-axis along the first coordinate.
     * @param[in] semi_b Semi-axis along the second coordinate.
     * @param[in] exponent Shape exponent of the superellipse.
     * @param[out] out_x First coordinate of the point.
     * @param[out] out_y Second coordinate of the point.
     */
    __device__ __forceinline__ void evaluate_superellipse(const double theta, const double semi_a, const double semi_b,
                                                          const double exponent, double &out_x, double &out_y)
    {
        const double magnitude = fabs(theta);
        const double cos_theta = cos(magnitude);
        const double sin_theta = copysign(sin(magnitude), theta);
        out_x = semi_a * signed_pow(cos_theta, exponent);
        out_y = semi_b * signed_pow(sin_theta, exponent);
    }

    /**
     * @brief Measures the chord between two superellipse points without fused multiply-add.
     * @details Written with the round-to-nearest intrinsics rather than plain operators because
     * nvcc contracts `dx * dx + dy * dy` into an FMA by default, which rounds once instead of
     * twice and shifts the result by about one ULP. The budget split downstream rounds to an
     * integer, so a one-ULP shift occasionally lands on the other side of a boundary. This is the
     * same hazard that makes a plain square root preferable to the more accurate `hypot`.
     * @param[in] ax First coordinate of the first point.
     * @param[in] ay Second coordinate of the first point.
     * @param[in] bx First coordinate of the second point.
     * @param[in] by Second coordinate of the second point.
     * @return The Euclidean distance between the two points.
     */
    __device__ __forceinline__ double superellipse_chord(const double ax, const double ay, const double bx,
                                                         const double by)
    {
        const double dx = ax - bx;
        const double dy = ay - by;
        return sqrt(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
    }

    /**
     * @brief Distributes `num` angles along a superellipse at approximately equal arc length.
     * @details Sweeping the angle uniformly bunches samples at the corners of a low-exponent
     * superellipse and starves its flat sides. This bisects the angular interval, hands each half
     * a share of the budget proportional to its chord length, and walks the resulting binary tree
     * depth-first through an explicit stack.
     *
     * Two choices are load-bearing for reproducing the reference bitwise: the chord is a plain
     * square root, not the more accurate `hypot`, which differs by one ULP often enough to tip
     * the integer split; and the split uses `rint` (halfway to even), because a symmetric
     * primitive lands *exactly* on a halfway case where `round` would disagree routinely.
     *
     * @param[in] semi_a Semi-axis along the first coordinate.
     * @param[in] semi_b Semi-axis along the second coordinate.
     * @param[in] exponent Shape exponent of the superellipse.
     * @param[in] theta_a Angle the sampling starts at.
     * @param[in] theta_b Angle the sampling ends at.
     * @param[in] num Number of angles to produce, endpoints included.
     * @param[in,out] stack Scratch of at least `num + 2` frames owned by this thread.
     * @param[out] out_thetas Output buffer of `num` angles in traversal order.
     */
    __device__ void sample_superellipse_arclength(const double semi_a, const double semi_b, const double exponent,
                                                  const double theta_a, const double theta_b, const int num,
                                                  SuperellipseFrame *__restrict__ stack,
                                                  double *__restrict__ out_thetas)
    {
        for (int i = 0; i < num; ++i)
            out_thetas[i] = 0.0;
        out_thetas[0] = theta_a;
        out_thetas[num - 1] = theta_b;

        int stack_size = 0;
        SuperellipseFrame root;
        evaluate_superellipse(theta_a, semi_a, semi_b, exponent, root.point_a_x, root.point_a_y);
        evaluate_superellipse(theta_b, semi_a, semi_b, exponent, root.point_b_x, root.point_b_y);
        root.theta_a = theta_a;
        root.theta_b = theta_b;
        root.budget = num - 2;
        root.offset = 1;
        stack[stack_size++] = root;

        while (stack_size > 0)
        {
            const SuperellipseFrame frame = stack[--stack_size];
            if (frame.budget <= 0)
                continue;

            const double theta = 0.5 * (frame.theta_a + frame.theta_b);
            double mid_x, mid_y;
            evaluate_superellipse(theta, semi_a, semi_b, exponent, mid_x, mid_y);

            const double span_a = superellipse_chord(frame.point_a_x, frame.point_a_y, mid_x, mid_y);
            const double span_b = superellipse_chord(mid_x, mid_y, frame.point_b_x, frame.point_b_y);
            const double total = span_a + span_b;

            const int take_a =
                (total < 1e-12) ? (frame.budget / 2) : (int)rint(span_a / total * (double)(frame.budget - 1));
            const int take_b = frame.budget - take_a - 1;
            out_thetas[take_a + frame.offset] = theta;

            SuperellipseFrame left;
            left.point_a_x = frame.point_a_x;
            left.point_a_y = frame.point_a_y;
            left.point_b_x = mid_x;
            left.point_b_y = mid_y;
            left.theta_a = frame.theta_a;
            left.theta_b = theta;
            left.budget = take_a;
            left.offset = frame.offset;
            stack[stack_size++] = left;

            SuperellipseFrame right;
            right.point_a_x = mid_x;
            right.point_a_y = mid_y;
            right.point_b_x = frame.point_b_x;
            right.point_b_y = frame.point_b_y;
            right.theta_a = theta;
            right.theta_b = frame.theta_b;
            right.budget = take_b;
            right.offset = frame.offset + take_a + 1;
            stack[stack_size++] = right;
        }
    }

    /**
     * @brief Samples the azimuthal and polar superellipses of every primitive.
     * @details One thread per curve, so two per primitive, each walking its own
     * divide-and-conquer traversal. The parallelism is only $2K$ wide, but a parallel tree would
     * have to re-derive the traversal order and with it the exact sample placement.
     *
     * The azimuth takes one extra sample across the closed period; the caller drops the last,
     * since sampling $[-\pi, \pi]$ inclusive repeats the first column.
     *
     * @param[in] num_quadrics Number of primitives $K$.
     * @param[in] resolution Angular samples along each axis.
     * @param[in] scales Device array of $K$ semi-axis triples.
     * @param[in] exponents Device array of $K$ shape exponent pairs.
     * @param[in,out] stack_scratch Device array of $2K \times (\text{resolution} + 2)$ frames.
     * @param[out] out_azimuths Device array of $K \times (\text{resolution} + 1)$ angles.
     * @param[out] out_polars Device array of $K \times \text{resolution}$ angles.
     */
    __global__ void compute_superellipse_angles_kernel(const uint32_t num_quadrics, const uint32_t resolution,
                                                       const float3 *__restrict__ scales,
                                                       const float2 *__restrict__ exponents,
                                                       SuperellipseFrame *__restrict__ stack_scratch,
                                                       double *__restrict__ out_azimuths,
                                                       double *__restrict__ out_polars)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= 2u * num_quadrics)
            return;

        const uint32_t k = idx >> 1;
        const bool is_azimuth = (idx & 1u) == 0u;
        const int n = (int)resolution;

        const float3 scale = scales[k];
        const float2 exponent = exponents[k];

        SuperellipseFrame *stack = stack_scratch + (size_t)idx * (size_t)(resolution + 2);

        if (is_azimuth)
        {
            // The azimuthal cross-section lies in the (x, y) plane and is governed by e2.
            sample_superellipse_arclength((double)scale.x, (double)scale.y, (double)exponent.y, -CUDART_PI, CUDART_PI,
                                          n + 1, stack, out_azimuths + (size_t)k * (size_t)(resolution + 1));
        }
        else
        {
            // The polar cross-section lies in the (x, z) plane and is governed by e1.
            sample_superellipse_arclength((double)scale.x, (double)scale.z, (double)exponent.x, -0.5 * CUDART_PI,
                                          0.5 * CUDART_PI, n, stack, out_polars + (size_t)k * (size_t)resolution);
        }
    }

    /**
     * @brief Places every tessellation vertex of every primitive in world space.
     * @details One thread per output vertex. The vertex layout per primitive is one south pole,
     * then `resolution - 2` interior latitude rings of `resolution` vertices with the azimuth
     * varying fastest, then one north pole.
     *
     * The poles are placed exactly rather than evaluated: $\cos(\pm\pi/2)$ is $6\times10^{-17}$
     * rather than zero, and raising that to a small exponent yields a ring of finite radius instead
     * of a point. Everything is computed in double and narrowed only on the write, because near the
     * poles a float32 angle perturbs $\cos\eta$ by an order of magnitude and the fractional power
     * turns that into a visible displacement, precisely on the sharp corners worth capturing.
     *
     * @param[in] num_quadrics Number of primitives $K$.
     * @param[in] resolution Angular samples along each axis.
     * @param[in] scales Device array of $K$ semi-axis triples.
     * @param[in] exponents Device array of $K$ shape exponent pairs.
     * @param[in] rotations Device array of $K$ row-major local-to-world rotation matrices.
     * @param[in] translations Device array of $K$ primitive centres.
     * @param[in] azimuths Device array of $K \times (\text{resolution} + 1)$ azimuthal angles.
     * @param[in] polars Device array of $K \times \text{resolution}$ polar angles.
     * @param[in] return_labels Whether to also record each vertex's originating primitive.
     * @param[out] out_vertices Device array of $K \times (n(n-2) + 2)$ world coordinates.
     * @param[out] out_labels Device array of the same length holding primitive indices, or nullptr.
     * @warning A rotation of negative determinant is a reflection, which inverts triangle winding;
     * the azimuthal order is reversed in that case so normals keep facing outward.
     */
    __global__ void compute_superquadric_vertices_kernel(
        const uint32_t num_quadrics, const uint32_t resolution, const float3 *__restrict__ scales,
        const float2 *__restrict__ exponents, const float *__restrict__ rotations,
        const float3 *__restrict__ translations, const double *__restrict__ azimuths, const double *__restrict__ polars,
        const bool return_labels, float3 *__restrict__ out_vertices, int32_t *__restrict__ out_labels)
    {
        const uint32_t n = resolution;
        const uint32_t per_primitive = n * (n - 2u) + 2u;
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_quadrics * per_primitive)
            return;

        const uint32_t k = idx / per_primitive;
        const uint32_t local_idx = idx - k * per_primitive;

        const float3 scale = scales[k];
        const float2 exponent = exponents[k];
        const float *rot = rotations + (size_t)k * 9;
        const double a1 = (double)scale.x;
        const double a2 = (double)scale.y;
        const double a3 = (double)scale.z;

        double local_x, local_y, local_z;
        if (local_idx == 0u)
        {
            local_x = 0.0;
            local_y = 0.0;
            local_z = -a3;
        }
        else if (local_idx == per_primitive - 1u)
        {
            local_x = 0.0;
            local_y = 0.0;
            local_z = a3;
        }
        else
        {
            const uint32_t interior = local_idx - 1u;
            const uint32_t ring = interior / n;    // polar angle varies slowest
            uint32_t column = interior - ring * n; // azimuth varies fastest

            const double det = (double)rot[0] * ((double)rot[4] * (double)rot[8] - (double)rot[5] * (double)rot[7]) -
                               (double)rot[1] * ((double)rot[3] * (double)rot[8] - (double)rot[5] * (double)rot[6]) +
                               (double)rot[2] * ((double)rot[3] * (double)rot[7] - (double)rot[4] * (double)rot[6]);
            if (det < 0.0)
                column = n - 1u - column;

            const double omega = azimuths[(size_t)k * (size_t)(n + 1u) + column];
            const double eta = polars[(size_t)k * (size_t)n + ring + 1u];

            const double cos_eta = signed_pow(cos(eta), (double)exponent.x);
            local_x = a1 * cos_eta * signed_pow(cos(omega), (double)exponent.y);
            local_y = a2 * cos_eta * signed_pow(sin(omega), (double)exponent.y);
            local_z = a3 * signed_pow(sin(eta), (double)exponent.x);
        }

        const float3 translation = translations[k];
        const double world_x =
            (double)rot[0] * local_x + (double)rot[1] * local_y + (double)rot[2] * local_z + (double)translation.x;
        const double world_y =
            (double)rot[3] * local_x + (double)rot[4] * local_y + (double)rot[5] * local_z + (double)translation.y;
        const double world_z =
            (double)rot[6] * local_x + (double)rot[7] * local_y + (double)rot[8] * local_z + (double)translation.z;

        out_vertices[idx] = make_float3((float)world_x, (float)world_y, (float)world_z);
        if (return_labels)
            out_labels[idx] = (int32_t)k;
    }

    /**
     * @brief Writes the triangle connectivity of every tessellated primitive.
     * @details One thread per output triangle. Connectivity depends only on `resolution`, so every
     * primitive shares the same pattern shifted by its vertex offset. The triangles are emitted in
     * four contiguous blocks: the south fan, the two halves of the interior quads, and the north
     * fan. The azimuthal wrap is folded into the column arithmetic rather than stitched on as a
     * separate seam, which is what leaves the surface closed with no duplicated vertices.
     * @param[in] num_quadrics Number of primitives $K$.
     * @param[in] resolution Angular samples along each axis.
     * @param[out] out_triangles Device array of $K \times 2n(n-2)$ vertex index triples.
     */
    __global__ void compute_superquadric_faces_kernel(const uint32_t num_quadrics, const uint32_t resolution,
                                                      int3 *__restrict__ out_triangles)
    {
        const uint32_t n = resolution;
        const uint32_t rings = n - 2u;
        const uint32_t quad_tris = n * (rings - 1u); // one half of the interior quads
        const uint32_t per_primitive = 2u * n * rings;
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_quadrics * per_primitive)
            return;

        const uint32_t k = idx / per_primitive;
        const uint32_t local_idx = idx - k * per_primitive;
        const int32_t base = (int32_t)(k * (n * rings + 2u));
        const int32_t north = (int32_t)(1u + rings * n);

        int3 tri;
        if (local_idx < n)
        {
            const int32_t c = (int32_t)local_idx;
            const int32_t c_next = (int32_t)((local_idx + 1u) % n);
            tri = make_int3(1 + c, 0, 1 + c_next);
        }
        else if (local_idx < n + 2u * quad_tris)
        {
            const uint32_t within = local_idx - n;
            const bool second_half = within >= quad_tris;
            const uint32_t q = second_half ? (within - quad_tris) : within;
            const uint32_t r = q / n;
            const uint32_t c = q - r * n;
            const uint32_t c_next = (c + 1u) % n;
            const int32_t v00 = (int32_t)(1u + r * n + c);
            const int32_t v01 = (int32_t)(1u + r * n + c_next);
            const int32_t v10 = (int32_t)(1u + (r + 1u) * n + c);
            const int32_t v11 = (int32_t)(1u + (r + 1u) * n + c_next);
            tri = second_half ? make_int3(v10, v01, v11) : make_int3(v00, v01, v10);
        }
        else
        {
            const uint32_t c = local_idx - n - 2u * quad_tris;
            const int32_t last = (int32_t)(1u + (rings - 1u) * n);
            tri = make_int3(last + (int32_t)c, last + (int32_t)((c + 1u) % n), north);
        }

        out_triangles[idx] = make_int3(tri.x + base, tri.y + base, tri.z + base);
    }

    void compute_superellipse_angles(const uint32_t num_quadrics, const uint32_t resolution,
                                     const float3 *__restrict__ scales, const float2 *__restrict__ exponents,
                                     SuperellipseFrame *__restrict__ stack_scratch, double *__restrict__ out_azimuths,
                                     double *__restrict__ out_polars)
    {
        if (num_quadrics == 0)
            return;

        uint32_t threads = NTHREADS;
        uint32_t blocks = (2u * num_quadrics + threads - 1) / threads;

        compute_superellipse_angles_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_quadrics, resolution, scales, exponents, stack_scratch, out_azimuths, out_polars);
    }

    void compute_superquadric_vertices(const uint32_t num_quadrics, const uint32_t resolution,
                                       const float3 *__restrict__ scales, const float2 *__restrict__ exponents,
                                       const float *__restrict__ rotations, const float3 *__restrict__ translations,
                                       const double *__restrict__ azimuths, const double *__restrict__ polars,
                                       const bool return_labels, float3 *__restrict__ out_vertices,
                                       int32_t *__restrict__ out_labels)
    {
        if (num_quadrics == 0)
            return;

        const uint32_t per_primitive = resolution * (resolution - 2u) + 2u;
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_quadrics * per_primitive + threads - 1) / threads;

        compute_superquadric_vertices_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_quadrics, resolution, scales, exponents, rotations, translations, azimuths, polars, return_labels,
            out_vertices, out_labels);
    }

    void compute_superquadric_faces(const uint32_t num_quadrics, const uint32_t resolution,
                                    int3 *__restrict__ out_triangles)
    {
        if (num_quadrics == 0)
            return;

        const uint32_t per_primitive = 2u * resolution * (resolution - 2u);
        uint32_t threads = NTHREADS;
        uint32_t blocks = (num_quadrics * per_primitive + threads - 1) / threads;

        compute_superquadric_faces_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_quadrics, resolution, out_triangles);
    }
} // namespace sq
