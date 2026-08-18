#include "dc.h"
#include "dc_data.h"
#include "../maths/maths.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/unique.h>

#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDAGuard.h>

namespace conquer3d {
namespace ops {

namespace {

__device__ __forceinline__ float3 compute_trilinear_normal(
    float u, float v, float w,
    const float s[8],
    float dx, float dy, float dz
) {
    float du = (1.0f - w) * ((1.0f - v) * (s[1] - s[0]) + v * (s[2] - s[3])) +
                      w   * ((1.0f - v) * (s[5] - s[4]) + v * (s[6] - s[7]));

    float dv = (1.0f - w) * ((1.0f - u) * (s[3] - s[0]) + u * (s[2] - s[1])) +
                      w   * ((1.0f - u) * (s[7] - s[4]) + u * (s[6] - s[5]));

    float dw = (1.0f - v) * ((1.0f - u) * (s[4] - s[0]) + u * (s[5] - s[1])) +
                      v   * ((1.0f - u) * (s[7] - s[3]) + u * (s[6] - s[2]));

    float3 grad = make_float3(du / dx, dv / dy, dw / dz);
    float len = maths::norm(grad);
    if (len > 1e-8f) {
        return grad * (1.0f / len);
    }
    return make_float3(0.0f, 0.0f, 1.0f);
}

// Compute minimum angle of a 3D triangle in radians
__device__ __forceinline__ float triangle_min_angle(
    const float3 &p0, const float3 &p1, const float3 &p2
) {
    float3 e0 = p1 - p0;
    float3 e1 = p2 - p1;
    float3 e2 = p0 - p2;

    float l0 = maths::norm(e0);
    float l1 = maths::norm(e1);
    float l2 = maths::norm(e2);

    if (l0 < 1e-8f || l1 < 1e-8f || l2 < 1e-8f) return 0.0f;

    float cos0 = -maths::dot(e2, e0) / (l2 * l0);
    float cos1 = -maths::dot(e0, e1) / (l0 * l1);
    float cos2 = -maths::dot(e1, e2) / (l1 * l2);

    cos0 = fmaxf(-1.0f, fminf(1.0f, cos0));
    cos1 = fmaxf(-1.0f, fminf(1.0f, cos1));
    cos2 = fmaxf(-1.0f, fminf(1.0f, cos2));

    return fminf(acosf(cos0), fminf(acosf(cos1), acosf(cos2)));
}

} // namespace

// -----------------------------------------------------------------------------------------
// Kernel 1: Dual Vertex Generation via QEF
// -----------------------------------------------------------------------------------------
__global__ void compute_dual_vertices_kernel(
    const float3 *__restrict__ grid_vertices,
    const int *__restrict__ voxels,
    const float *__restrict__ sdf,
    const float3 *__restrict__ grid_normals,
    float iso,
    int num_voxels,
    float3 *__restrict__ dual_vertices,
    int *__restrict__ voxel_is_active,
    int *__restrict__ bipolar_edge_counts
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    // Load 8 corner indices
    int c_idx[8];
    float s[8];
    float3 p[8];
    float3 c_min = make_float3(1e30f, 1e30f, 1e30f);
    float3 c_max = make_float3(-1e30f, -1e30f, -1e30f);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        c_idx[i] = voxels[m * 8 + i];
        s[i] = sdf[c_idx[i]];
        p[i] = grid_vertices[c_idx[i]];
        c_min = maths::min(c_min, p[i]);
        c_max = maths::max(c_max, p[i]);
    }

    float dx = fmaxf(c_max.x - c_min.x, 1e-6f);
    float dy = fmaxf(c_max.y - c_min.y, 1e-6f);
    float dz = fmaxf(c_max.z - c_min.z, 1e-6f);

    float3 pts[12];
    float3 normals[12];
    int count = 0;

    // Check all 12 edges
    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        int v0 = dc_edge_corners[e][0];
        int v1 = dc_edge_corners[e][1];
        float s0 = s[v0];
        float s1 = s[v1];

        // Bipolar test
        if ((s0 < iso && s1 >= iso) || (s0 >= iso && s1 < iso)) {
            float t = (iso - s0) / (s1 - s0);
            t = fmaxf(0.0f, fminf(1.0f, t));

            float3 pt = p[v0] + (p[v1] - p[v0]) * t;
            pts[count] = pt;

            float3 n;
            if (grid_normals != nullptr) {
                float3 n0 = grid_normals[c_idx[v0]];
                float3 n1 = grid_normals[c_idx[v1]];
                float3 n_interp = n0 + (n1 - n0) * t;
                float len = maths::norm(n_interp);
                n = (len > 1e-8f) ? (n_interp * (1.0f / len)) : make_float3(0, 0, 1);
            } else {
                float u = dc_corner_uvw[v0][0] + (dc_corner_uvw[v1][0] - dc_corner_uvw[v0][0]) * t;
                float v = dc_corner_uvw[v0][1] + (dc_corner_uvw[v1][1] - dc_corner_uvw[v0][1]) * t;
                float w = dc_corner_uvw[v0][2] + (dc_corner_uvw[v1][2] - dc_corner_uvw[v0][2]) * t;
                n = compute_trilinear_normal(u, v, w, s, dx, dy, dz);
            }
            normals[count] = n;
            count++;
        }
    }

    bipolar_edge_counts[m] = count;

    if (count > 0) {
        voxel_is_active[m] = 1;
        float3 dual_v = maths::solve_qef(pts, normals, count, c_min, c_max, 0.01f);
        dual_vertices[m] = dual_v;
    } else {
        voxel_is_active[m] = 0;
        dual_vertices[m] = (c_min + c_max) * 0.5f;
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 2: Emit Bipolar Edge Instances
// -----------------------------------------------------------------------------------------
__global__ void emit_bipolar_edges_kernel(
    const int *__restrict__ voxels,
    const float *__restrict__ sdf,
    const int *__restrict__ edge_offsets,
    float iso,
    int num_voxels,
    uint64_t *__restrict__ out_edge_keys,
    uint32_t *__restrict__ out_voxel_and_edge
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    int offset = edge_offsets[m];

    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        int v0 = voxels[m * 8 + dc_edge_corners[e][0]];
        int v1 = voxels[m * 8 + dc_edge_corners[e][1]];
        float s0 = sdf[v0];
        float s1 = sdf[v1];

        if ((s0 < iso && s1 >= iso) || (s0 >= iso && s1 < iso)) {
            int g_min = v0 < v1 ? v0 : v1;
            int g_max = v0 < v1 ? v1 : v0;
            uint64_t key = ((uint64_t)g_min << 32) | (uint64_t)g_max;

            out_edge_keys[offset] = key;
            out_voxel_and_edge[offset] = ((uint32_t)m << 4) | (uint32_t)e;
            offset++;
        }
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 3: Gather Dual Quads from Edge Incidences
// -----------------------------------------------------------------------------------------
__global__ void gather_dual_quads_kernel(
    const uint64_t *__restrict__ sorted_edge_keys,
    const uint32_t *__restrict__ sorted_voxel_and_edge,
    const float3 *__restrict__ grid_vertices,
    const float *__restrict__ sdf,
    const float3 *__restrict__ dual_vertices,
    const int *__restrict__ voxel_to_compact_idx,
    int total_instances,
    int4 *__restrict__ out_quads,
    int *__restrict__ out_quad_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_instances) return;

    // We only process the start of a multi-voxel cluster sharing the same unique edge key
    uint64_t key = sorted_edge_keys[i];
    if (i > 0 && sorted_edge_keys[i - 1] == key) {
        return; // Not the first instance of this edge
    }

    // Count how many incident voxels share this edge key
    int count = 1;
    while (i + count < total_instances && sorted_edge_keys[i + count] == key) {
        count++;
    }

    if (count < 3) {
        return; // Need at least 3 cells to form a closed surface face
    }

    // Extract edge endpoints
    int v0_id = (int)(key >> 32);
    int v1_id = (int)(key & 0xFFFFFFFFULL);
    float s0 = sdf[v0_id];
    float s1 = sdf[v1_id];

    int vox_ids[4] = {-1, -1, -1, -1};
    int present_mask = 0;
    for (int k = 0; k < count && k < 4; ++k) {
        uint32_t val = sorted_voxel_and_edge[i + k];
        int vox_m = (int)(val >> 4);
        int e = (int)(val & 0xF);
        int slot = dc_edge_quadrant[e];
        vox_ids[slot] = voxel_to_compact_idx[vox_m];
        present_mask |= (1 << slot);
    }

    if (count >= 4) {
        // Orient quad: if s0 < s1, CCW order around edge: (0, 1, 2, 3)
        // if s0 > s1, reverse order: (0, 3, 2, 1)
        int4 quad;
        if (s0 < s1) {
            quad = make_int4(vox_ids[0], vox_ids[1], vox_ids[2], vox_ids[3]);
        } else {
            quad = make_int4(vox_ids[0], vox_ids[3], vox_ids[2], vox_ids[1]);
        }

        int quad_idx = atomicAdd(out_quad_count, 1);
        out_quads[quad_idx] = quad;
    } else if (count == 3) {
        // 3 slots present, 1 missing at boundary of sparse grid
        int missing_slot = 0;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (!(present_mask & (1 << s))) {
                missing_slot = s;
                break;
            }
        }
        int va = vox_ids[(missing_slot + 1) & 3];
        int vb = vox_ids[(missing_slot + 2) & 3];
        int vc = vox_ids[(missing_slot + 3) & 3];

        int4 quad;
        if (s0 < s1) {
            quad = make_int4(va, vb, vc, va);
        } else {
            quad = make_int4(va, vc, vb, va);
        }

        int quad_idx = atomicAdd(out_quad_count, 1);
        out_quads[quad_idx] = quad;
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 4: Optimal Quad-to-Triangle Splitting (Max-Min Angle Criterion)
// -----------------------------------------------------------------------------------------
__global__ void quad_to_triangle_kernel(
    const int4 *__restrict__ quads,
    const float3 *__restrict__ compact_vertices,
    int num_quads,
    int3 *__restrict__ out_triangles,
    int *__restrict__ out_tri_count
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_quads) return;

    int4 quad = quads[q];
    if (quad.x == quad.w) {
        // Single boundary triangle (quad.x, quad.y, quad.z)
        int idx = atomicAdd(out_tri_count, 1);
        out_triangles[idx] = make_int3(quad.x, quad.y, quad.z);
        return;
    }

    float3 p0 = compact_vertices[quad.x];
    float3 p1 = compact_vertices[quad.y];
    float3 p2 = compact_vertices[quad.z];
    float3 p3 = compact_vertices[quad.w];

    // Split A: (0, 1, 2) & (0, 2, 3)
    float min_angle_A = fminf(triangle_min_angle(p0, p1, p2), triangle_min_angle(p0, p2, p3));

    // Split B: (1, 2, 3) & (1, 3, 0)
    float min_angle_B = fminf(triangle_min_angle(p1, p2, p3), triangle_min_angle(p1, p3, p0));

    int idx = atomicAdd(out_tri_count, 2);
    if (min_angle_A >= min_angle_B) {
        out_triangles[idx]     = make_int3(quad.x, quad.y, quad.z);
        out_triangles[idx + 1] = make_int3(quad.x, quad.z, quad.w);
    } else {
        out_triangles[idx]     = make_int3(quad.y, quad.z, quad.w);
        out_triangles[idx + 1] = make_int3(quad.y, quad.w, quad.x);
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 5: Compact Dual Vertices & Feature Colors
// -----------------------------------------------------------------------------------------
__global__ void compact_dual_vertices_and_colors_kernel(
    const float3 *__restrict__ dual_vertices,
    const int *__restrict__ voxel_is_active,
    const int *__restrict__ voxel_to_compact_idx,
    const int *__restrict__ voxels,
    const float *__restrict__ colors,
    const float3 *__restrict__ grid_vertices,
    int num_voxels,
    int num_channels,
    float3 *__restrict__ compact_vertices,
    float *__restrict__ compact_colors
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    if (!voxel_is_active[m]) return;

    int dst_idx = voxel_to_compact_idx[m];
    float3 v = dual_vertices[m];
    compact_vertices[dst_idx] = v;

    if (colors != nullptr && compact_colors != nullptr) {
        // Trilinearly interpolate corner colors to dual vertex
        float3 c_min = grid_vertices[voxels[m * 8 + 0]];
        float3 c_max = grid_vertices[voxels[m * 8 + 6]];
        float dx = fmaxf(c_max.x - c_min.x, 1e-6f);
        float dy = fmaxf(c_max.y - c_min.y, 1e-6f);
        float dz = fmaxf(c_max.z - c_min.z, 1e-6f);

        float u = fmaxf(0.0f, fminf(1.0f, (v.x - c_min.x) / dx));
        float val_v = fmaxf(0.0f, fminf(1.0f, (v.y - c_min.y) / dy));
        float w = fmaxf(0.0f, fminf(1.0f, (v.z - c_min.z) / dz));

        #pragma unroll
        for (int ch = 0; ch < num_channels; ++ch) {
            float c0 = colors[voxels[m * 8 + 0] * num_channels + ch];
            float c1 = colors[voxels[m * 8 + 1] * num_channels + ch];
            float c2 = colors[voxels[m * 8 + 2] * num_channels + ch];
            float c3 = colors[voxels[m * 8 + 3] * num_channels + ch];
            float c4 = colors[voxels[m * 8 + 4] * num_channels + ch];
            float c5 = colors[voxels[m * 8 + 5] * num_channels + ch];
            float c6 = colors[voxels[m * 8 + 6] * num_channels + ch];
            float c7 = colors[voxels[m * 8 + 7] * num_channels + ch];

            float c_interp = 
                (1 - u) * (1 - val_v) * (1 - w) * c0 +
                u       * (1 - val_v) * (1 - w) * c1 +
                u       * val_v       * (1 - w) * c2 +
                (1 - u) * val_v       * (1 - w) * c3 +
                (1 - u) * (1 - val_v) * w       * c4 +
                u       * (1 - val_v) * w       * c5 +
                u       * val_v       * w       * c6 +
                (1 - u) * val_v       * w       * c7;

            compact_colors[dst_idx * num_channels + ch] = c_interp;
        }
    }
}

// -----------------------------------------------------------------------------------------
// Kernel 6: Analytical Differentiable Backward Kernel
// -----------------------------------------------------------------------------------------
__global__ void backward_dual_contouring_kernel(
    const float3 *__restrict__ grad_verts,
    const float *__restrict__ grad_colors,
    const float3 *__restrict__ grid_vertices,
    const int *__restrict__ voxels,
    const float *__restrict__ sdf,
    const int *__restrict__ voxel_is_active,
    const int *__restrict__ voxel_to_compact_idx,
    float iso,
    int num_voxels,
    int num_channels,
    float *__restrict__ grad_sdf,
    float *__restrict__ grad_colors_in
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_voxels) return;

    if (!voxel_is_active[m]) return;

    int compact_idx = voxel_to_compact_idx[m];
    float3 gv = grad_verts[compact_idx];

    // Distribute gradient to corner SDF values
    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        int v0 = voxels[m * 8 + dc_edge_corners[e][0]];
        int v1 = voxels[m * 8 + dc_edge_corners[e][1]];
        float s0 = sdf[v0];
        float s1 = sdf[v1];

        if ((s0 < iso && s1 >= iso) || (s0 >= iso && s1 < iso)) {
            float denom = s1 - s0;
            if (fabsf(denom) > 1e-7f) {
                float3 p0 = grid_vertices[v0];
                float3 p1 = grid_vertices[v1];
                float3 dp = p1 - p0;

                // dt / ds0 = (iso - s1) / (s1 - s0)^2
                // dt / ds1 = -(iso - s0) / (s1 - s0)^2
                float dt_ds0 = (iso - s1) / (denom * denom);
                float dt_ds1 = -(iso - s0) / (denom * denom);

                float g_dot = maths::dot(gv, dp);
                atomicAdd(&grad_sdf[v0], g_dot * dt_ds0 * 0.25f);
                atomicAdd(&grad_sdf[v1], g_dot * dt_ds1 * 0.25f);
            }
        }
    }

    if (grad_colors != nullptr && grad_colors_in != nullptr) {
        #pragma unroll
        for (int ch = 0; ch < num_channels; ++ch) {
            float gc = grad_colors[compact_idx * num_channels + ch] * 0.125f;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                atomicAdd(&grad_colors_in[voxels[m * 8 + k] * num_channels + ch], gc);
            }
        }
    }
}

// -----------------------------------------------------------------------------------------
// Host Implementation
// -----------------------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> dual_contouring(
    const at::Tensor &grid_vertices,
    const at::Tensor &voxels,
    const at::Tensor &sdf,
    const c10::optional<at::Tensor> &grid_normals,
    const c10::optional<at::Tensor> &colors,
    float iso,
    bool quad_split
) {
    at::cuda::CUDAGuard device_guard(grid_vertices.device());
    auto allocator = at::cuda::ThrustAllocator();
    auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());

    int num_voxels = voxels.size(0);
    if (num_voxels == 0) {
        int face_dim = quad_split ? 3 : 4;
        return {
            at::zeros({0, 3}, grid_vertices.options()),
            at::zeros({0, face_dim}, voxels.options()),
            colors.has_value() ? c10::optional<at::Tensor>(at::zeros({0, colors.value().size(1)}, colors.value().options())) : c10::nullopt
        };
    }

    // 1. Pass 1: Compute Dual Vertices per Voxel via QEF
    at::Tensor dual_vertices = at::empty({num_voxels, 3}, grid_vertices.options());
    at::Tensor voxel_is_active = at::zeros({num_voxels}, voxels.options());
    at::Tensor bipolar_edge_counts = at::empty({num_voxels}, voxels.options());

    int threads = 256;
    int blocks = (num_voxels + threads - 1) / threads;

    const float3 *normals_ptr = grid_normals.has_value() ? reinterpret_cast<const float3*>(grid_normals.value().data_ptr<float>()) : nullptr;

    compute_dual_vertices_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        normals_ptr,
        iso,
        num_voxels,
        reinterpret_cast<float3*>(dual_vertices.data_ptr<float>()),
        voxel_is_active.data_ptr<int>(),
        bipolar_edge_counts.data_ptr<int>()
    );

    // 2. Prefix sum for active voxels
    at::Tensor voxel_to_compact_idx = at::empty({num_voxels}, voxels.options());
    thrust::exclusive_scan(
        policy,
        voxel_is_active.data_ptr<int>(),
        voxel_is_active.data_ptr<int>() + num_voxels,
        voxel_to_compact_idx.data_ptr<int>(),
        0
    );

    int total_active_voxels = 0;
    int last_active = 0;
    int last_compact = 0;
    cudaMemcpyAsync(&last_active, voxel_is_active.data_ptr<int>() + num_voxels - 1, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
    cudaMemcpyAsync(&last_compact, voxel_to_compact_idx.data_ptr<int>() + num_voxels - 1, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    total_active_voxels = last_compact + last_active;

    if (total_active_voxels == 0) {
        int face_dim = quad_split ? 3 : 4;
        return {
            at::zeros({0, 3}, grid_vertices.options()),
            at::zeros({0, face_dim}, voxels.options()),
            colors.has_value() ? c10::optional<at::Tensor>(at::zeros({0, colors.value().size(1)}, colors.value().options())) : c10::nullopt
        };
    }

    // 3. Compact Dual Vertices & Feature Colors
    at::Tensor compact_vertices = at::empty({total_active_voxels, 3}, grid_vertices.options());
    c10::optional<at::Tensor> compact_colors = c10::nullopt;
    int num_channels = 0;
    float *compact_colors_ptr = nullptr;
    const float *colors_ptr = nullptr;

    if (colors.has_value()) {
        num_channels = colors.value().size(1);
        compact_colors = at::empty({total_active_voxels, num_channels}, colors.value().options());
        compact_colors_ptr = compact_colors.value().data_ptr<float>();
        colors_ptr = colors.value().data_ptr<float>();
    }

    compact_dual_vertices_and_colors_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float3*>(dual_vertices.data_ptr<float>()),
        voxel_is_active.data_ptr<int>(),
        voxel_to_compact_idx.data_ptr<int>(),
        voxels.data_ptr<int>(),
        colors_ptr,
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        num_voxels,
        num_channels,
        reinterpret_cast<float3*>(compact_vertices.data_ptr<float>()),
        compact_colors_ptr
    );

    // 4. Pass 2: Prefix sum for bipolar edge instances
    at::Tensor edge_offsets = at::empty({num_voxels}, voxels.options());
    thrust::exclusive_scan(
        policy,
        bipolar_edge_counts.data_ptr<int>(),
        bipolar_edge_counts.data_ptr<int>() + num_voxels,
        edge_offsets.data_ptr<int>(),
        0
    );

    int total_edge_instances = 0;
    int last_edge_count = 0;
    int last_edge_offset = 0;
    cudaMemcpyAsync(&last_edge_count, bipolar_edge_counts.data_ptr<int>() + num_voxels - 1, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
    cudaMemcpyAsync(&last_edge_offset, edge_offsets.data_ptr<int>() + num_voxels - 1, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    total_edge_instances = last_edge_offset + last_edge_count;

    if (total_edge_instances == 0) {
        int face_dim = quad_split ? 3 : 4;
        return {
            compact_vertices,
            at::zeros({0, face_dim}, voxels.options()),
            compact_colors
        };
    }

    // 5. Emit bipolar edges
    at::Tensor edge_keys = at::empty({total_edge_instances}, voxels.options().dtype(at::kLong));
    at::Tensor voxel_and_edge = at::empty({total_edge_instances}, voxels.options());

    emit_bipolar_edges_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        edge_offsets.data_ptr<int>()
        ,
        iso,
        num_voxels,
        reinterpret_cast<uint64_t*>(edge_keys.data_ptr<int64_t>()),
        reinterpret_cast<uint32_t*>(voxel_and_edge.data_ptr<int>())
    );

    // 6. Thrust Radix Sort by Edge Key
    thrust::sort_by_key(
        policy,
        reinterpret_cast<uint64_t*>(edge_keys.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(edge_keys.data_ptr<int64_t>()) + total_edge_instances,
        reinterpret_cast<uint32_t*>(voxel_and_edge.data_ptr<int>())
    );

    // 7. Pass 3: Gather Dual Quads
    int max_possible_quads = total_edge_instances / 3 + 1;
    at::Tensor raw_quads = at::empty({max_possible_quads, 4}, voxels.options());
    at::Tensor quad_count_tensor = at::zeros({1}, voxels.options());

    int edge_blocks = (total_edge_instances + threads - 1) / threads;
    gather_dual_quads_kernel<<<edge_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint64_t*>(edge_keys.data_ptr<int64_t>()),
        reinterpret_cast<const uint32_t*>(voxel_and_edge.data_ptr<int>()),
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        sdf.data_ptr<float>(),
        reinterpret_cast<const float3*>(dual_vertices.data_ptr<float>()),
        voxel_to_compact_idx.data_ptr<int>(),
        total_edge_instances,
        reinterpret_cast<int4*>(raw_quads.data_ptr<int>()),
        quad_count_tensor.data_ptr<int>()
    );

    int num_quads = 0;
    cudaMemcpyAsync(&num_quads, quad_count_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

    at::Tensor out_quads = raw_quads.slice(0, 0, num_quads);

    // 8. Optional Quad-to-Triangle Splitting
    if (quad_split) {
        at::Tensor raw_triangles = at::empty({num_quads * 2, 3}, voxels.options());
        at::Tensor tri_count_tensor = at::zeros({1}, voxels.options());
        if (num_quads > 0) {
            int quad_blocks = (num_quads + threads - 1) / threads;
            quad_to_triangle_kernel<<<quad_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const int4*>(out_quads.data_ptr<int>()),
                reinterpret_cast<const float3*>(compact_vertices.data_ptr<float>()),
                num_quads,
                reinterpret_cast<int3*>(raw_triangles.data_ptr<int>()),
                tri_count_tensor.data_ptr<int>()
            );
        }
        int num_triangles = 0;
        cudaMemcpyAsync(&num_triangles, tri_count_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream());
        cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
        at::Tensor triangles = raw_triangles.slice(0, 0, num_triangles);
        return {compact_vertices, triangles, compact_colors};
    }

    return {compact_vertices, out_quads, compact_colors};
}

std::tuple<at::Tensor, c10::optional<at::Tensor>> dual_contouring_backward(
    const at::Tensor &grad_verts,
    const c10::optional<at::Tensor> &grad_colors,
    const at::Tensor &grid_vertices,
    const at::Tensor &voxels,
    const at::Tensor &sdf,
    const c10::optional<at::Tensor> &grid_normals,
    const c10::optional<at::Tensor> &colors,
    float iso
) {
    at::cuda::CUDAGuard device_guard(grid_vertices.device());
    int num_voxels = voxels.size(0);

    at::Tensor grad_sdf = at::zeros_like(sdf);
    c10::optional<at::Tensor> grad_colors_in = c10::nullopt;
    float *grad_colors_in_ptr = nullptr;
    const float *grad_colors_ptr = nullptr;
    int num_channels = 0;

    if (colors.has_value() && grad_colors.has_value()) {
        grad_colors_in = at::zeros_like(colors.value());
        grad_colors_in_ptr = grad_colors_in.value().data_ptr<float>();
        grad_colors_ptr = grad_colors.value().data_ptr<float>();
        num_channels = colors.value().size(1);
    }

    // Reconstruct active status and compact index
    at::Tensor voxel_is_active = at::zeros({num_voxels}, voxels.options());
    at::Tensor bipolar_edge_counts = at::empty({num_voxels}, voxels.options());
    at::Tensor dual_vertices = at::empty({num_voxels, 3}, grid_vertices.options());

    int threads = 256;
    int blocks = (num_voxels + threads - 1) / threads;
    const float3 *normals_ptr = grid_normals.has_value() ? reinterpret_cast<const float3*>(grid_normals.value().data_ptr<float>()) : nullptr;

    compute_dual_vertices_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        normals_ptr,
        iso,
        num_voxels,
        reinterpret_cast<float3*>(dual_vertices.data_ptr<float>()),
        voxel_is_active.data_ptr<int>(),
        bipolar_edge_counts.data_ptr<int>()
    );

    auto allocator = at::cuda::ThrustAllocator();
    auto policy = thrust::cuda::par(allocator).on(at::cuda::getCurrentCUDAStream());
    at::Tensor voxel_to_compact_idx = at::empty({num_voxels}, voxels.options());
    thrust::exclusive_scan(
        policy,
        voxel_is_active.data_ptr<int>(),
        voxel_is_active.data_ptr<int>() + num_voxels,
        voxel_to_compact_idx.data_ptr<int>(),
        0
    );

    backward_dual_contouring_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float3*>(grad_verts.data_ptr<float>()),
        grad_colors_ptr,
        reinterpret_cast<const float3*>(grid_vertices.data_ptr<float>()),
        voxels.data_ptr<int>(),
        sdf.data_ptr<float>(),
        voxel_is_active.data_ptr<int>(),
        voxel_to_compact_idx.data_ptr<int>(),
        iso,
        num_voxels,
        num_channels,
        grad_sdf.data_ptr<float>(),
        grad_colors_in_ptr
    );

    return {grad_sdf, grad_colors_in};
}

} // namespace ops
} // namespace conquer3d
