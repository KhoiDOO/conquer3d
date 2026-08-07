#include "grid.h"
#include <stdio.h>
#include <math.h>

namespace grid {

    __global__ void get_active_voxel_ids_from_depth_kernel(
        const int num_pixels,
        const float* depth_image,
        const float4x4 c2w,
        const float3x3 intrinsics_inv,
        const int image_width,
        const int image_height,
        const float3 grid_min,
        const float3 grid_max,
        const int3 res,
        int64_t* out_voxel_ids,
        unsigned long long* valid_counter,
        bool activate_neighbor,
        float trunc_margin
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_pixels) return;

        float d = depth_image[idx];
        if (d <= 0.0f) return; // Ignore invalid depths

        int ui = idx % image_width;
        int vi = idx / image_width;

        // 1. Unproject to camera coordinates using inverse intrinsics
        float3 p_cam;
        p_cam.x = d * (intrinsics_inv.m[0][0] * ui + intrinsics_inv.m[0][1] * vi + intrinsics_inv.m[0][2]);
        p_cam.y = d * (intrinsics_inv.m[1][0] * ui + intrinsics_inv.m[1][1] * vi + intrinsics_inv.m[1][2]);
        p_cam.z = d;

        // 2. Transform to world coordinates using Camera-To-World matrix
        float4 p_cam4 = make_float4(p_cam.x, p_cam.y, p_cam.z, 1.0f);
        float4 p_world4 = c2w * p_cam4;
        
        // 3. Compute 3D grid cell indices (0-indexed)
        float spacing_x = (grid_max.x - grid_min.x) / fmaxf(1.0f, (float)(res.x - 1));
        float spacing_y = (grid_max.y - grid_min.y) / fmaxf(1.0f, (float)(res.y - 1));
        float spacing_z = (grid_max.z - grid_min.z) / fmaxf(1.0f, (float)(res.z - 1));

        int i_min = floorf((p_world4.x - grid_min.x) / spacing_x);
        int i_max = i_min;
        int j_min = floorf((p_world4.y - grid_min.y) / spacing_y);
        int j_max = j_min;
        int k_min = floorf((p_world4.z - grid_min.z) / spacing_z);
        int k_max = k_min;

        if (activate_neighbor) {
            i_min = floorf((p_world4.x - trunc_margin - grid_min.x) / spacing_x);
            i_max = floorf((p_world4.x + trunc_margin - grid_min.x) / spacing_x);
            j_min = floorf((p_world4.y - trunc_margin - grid_min.y) / spacing_y);
            j_max = floorf((p_world4.y + trunc_margin - grid_min.y) / spacing_y);
            k_min = floorf((p_world4.z - trunc_margin - grid_min.z) / spacing_z);
            k_max = floorf((p_world4.z + trunc_margin - grid_min.z) / spacing_z);
        }

        for (int i = i_min; i <= i_max; ++i) {
            for (int j = j_min; j <= j_max; ++j) {
                for (int k = k_min; k <= k_max; ++k) {
                    if (i >= 0 && i < res.x - 1 && j >= 0 && j < res.y - 1 && k >= 0 && k < res.z - 1) {
                        int64_t voxel_id = (int64_t)i * (res.y - 1) * (res.z - 1) + (int64_t)j * (res.z - 1) + (int64_t)k;
                        unsigned long long write_idx = atomicAdd(valid_counter, 1ULL);
                        out_voxel_ids[write_idx] = voxel_id;
                    }
                }
            }
        }
    }

    void get_active_voxel_ids_from_depth(
        const int num_pixels,
        const float* depth_image,
        const float4x4 c2w,
        const float3x3 intrinsics_inv,
        const int image_width,
        const int image_height,
        const float3 grid_min,
        const float3 grid_max,
        const int3 res,
        int64_t* out_voxel_ids,
        unsigned long long* valid_counter,
        bool activate_neighbor,
        float trunc_margin
    ) {
        int block_size = 256;
        int grid_size = (num_pixels + block_size - 1) / block_size;

        get_active_voxel_ids_from_depth_kernel<<<grid_size, block_size>>>(
            num_pixels,
            depth_image,
            c2w,
            intrinsics_inv,
            image_width,
            image_height,
            grid_min,
            grid_max,
            res,
            out_voxel_ids,
            valid_counter,
            activate_neighbor,
            trunc_margin
        );
    }
}
