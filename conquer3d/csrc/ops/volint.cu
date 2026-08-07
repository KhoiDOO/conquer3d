#include "volint.h"
#include <stdio.h>
#include <math.h>

__global__ void single_view_volume_integral_kernel(
    const int num_vertices,
    const float3* grid_vertices,
    float* sdf,
    float* weight,
    float3* color,
    const float* depth_image,
    const float3* color_image,
    const int image_width,
    const int image_height,
    const float4x4 extrinsics,
    const float3x3 intrinsics,
    const float trunc_margin,
    const int mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;

    // 1. Fetch world position
    float3 p_world = grid_vertices[idx];

    // 2. Transform to camera coordinates using Extrinsics (World-to-Camera)
    float4 p_world4 = make_float4(p_world.x, p_world.y, p_world.z, 1.0f);
    float4 p_cam4 = extrinsics * p_world4;
    float3 p_cam = make_float3(p_cam4.x, p_cam4.y, p_cam4.z);
    
    float xc = p_cam.x, yc = p_cam.y, zc = p_cam.z;

    // Check if the vertex is behind the camera
    if (zc <= 0.0f) return;

    // 3. Project to image plane using Intrinsics
    float3 p_uvw = intrinsics * p_cam;
    float u = p_uvw.x / p_uvw.z;
    float v = p_uvw.y / p_uvw.z;

    int ui = (int)roundf(u);
    int vi = (int)roundf(v);

    // 4. Boundary check
    if (ui >= 0 && ui < image_width && vi >= 0 && vi < image_height) {
        int pixel_idx = vi * image_width + ui;
        float d = depth_image[pixel_idx];

        if (d > 0.0f) {
            // 5. Calculate SDF based on mode
            float sdf_val;
            if (mode == 1) {
                // True Euclidean SDF (Open3D UniformTSDFVolume convention)
                float ray_length = maths::norm(p_cam);
                float depth_to_camera_distance_multiplier = ray_length / zc;
                sdf_val = (d - zc) * depth_to_camera_distance_multiplier;
            } else {
                // Projective SDF shortcut
                sdf_val = d - zc;
            }
            
            // 6. Truncate and update running average if within margin
            if (sdf_val > -trunc_margin) {
                float tsdf = fminf(1.0f, sdf_val / trunc_margin);
                
                float old_sdf = sdf[idx];
                float old_weight = weight[idx];
                
                float inv_wsum = 1.0f / (old_weight + 1.0f);
                float new_sdf = (old_sdf * old_weight + tsdf) * inv_wsum;
                
                sdf[idx] = new_sdf;
                
                // Color integration
                if (color != nullptr && color_image != nullptr) {
                    float3 old_color = color[idx];
                    float3 incoming_color = color_image[pixel_idx];
                    
                    color[idx] = (old_color * old_weight + incoming_color) * inv_wsum;
                }
                
                weight[idx] = old_weight + 1.0f;
            }
        }
    }
}

void single_view_volume_integral(
    const int num_vertices,
    const float3* grid_vertices,
    float* sdf,
    float* weight,
    float3* color,
    const float* depth_image,
    const float3* color_image,
    const int image_width,
    const int image_height,
    const float4x4 extrinsics,
    const float3x3 intrinsics,
    const float trunc_margin,
    const int mode
) {
    int block_size = 256;
    int grid_size = (num_vertices + block_size - 1) / block_size;

    single_view_volume_integral_kernel<<<grid_size, block_size>>>(
        num_vertices, 
        grid_vertices, 
        sdf, 
        weight, 
        color,
        depth_image, 
        color_image, 
        image_width, 
        image_height, 
        extrinsics, 
        intrinsics, 
        trunc_margin,
        mode
    );
}