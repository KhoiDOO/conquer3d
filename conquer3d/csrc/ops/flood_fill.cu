#include "flood_fill.h"
#include "../constants.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace ops {

    __global__ void init_perimeter_kernel(
        int* __restrict__ mask,
        int* __restrict__ frontier,
        int* __restrict__ frontier_size,
        int VX, int VY, int VZ)
    {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t num_voxels = (int64_t)VX * VY * VZ;
        if (idx >= num_voxels) return;

        int vi = idx / (VY * VZ);
        int rem = idx % (VY * VZ);
        int vj = rem / VZ;
        int vk = rem % VZ;

        if (vi == 0 || vi == VX - 1 || vj == 0 || vj == VY - 1 || vk == 0 || vk == VZ - 1)
        {
            if (mask[idx] == -2)
            {
                mask[idx] = 2; // Water (Open Sea)
                int pos = atomicAdd(frontier_size, 1);
                frontier[pos] = idx;
            }
        }
    }

    __global__ void flood_fill_step_kernel(
        int* __restrict__ mask,
        const int* __restrict__ current_frontier,
        int frontier_size,
        int* __restrict__ next_frontier,
        int* __restrict__ next_frontier_size,
        int VX, int VY, int VZ,
        int connectivity)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= frontier_size) return;

        int voxel_idx = current_frontier[idx];
        int vi = voxel_idx / (VY * VZ);
        int rem = voxel_idx % (VY * VZ);
        int vj = rem / VZ;
        int vk = rem % VZ;

        bool is_collision = false;

        for (int di = -1; di <= 1; di++)
        {
            for (int dj = -1; dj <= 1; dj++)
            {
                for (int dk = -1; dk <= 1; dk++)
                {
                    if (di == 0 && dj == 0 && dk == 0) continue;

                    int dist = abs(di) + abs(dj) + abs(dk);
                    if (connectivity == 6 && dist > 1) continue;
                    if (connectivity == 18 && dist > 2) continue;

                    int ni = vi + di;
                    int nj = vj + dj;
                    int nk = vk + dk;

                    if (ni < 0 || ni >= VX || nj < 0 || nj >= VY || nk < 0 || nk >= VZ) continue;

                    int n_idx = ni * (VY * VZ) + nj * VZ + nk;
                    int n_val = mask[n_idx];

                    if (n_val == -1)
                    {
                        // Neighbor is a Dam (Occupied boundary cell) -> Tide boundary
                        is_collision = true;
                    }
                    else if (n_val == -2)
                    {
                        // Neighbor is Dry and empty -> try to convert to Water (2)
                        int old_val = atomicCAS(&mask[n_idx], -2, 2);
                        if (old_val == -2)
                        {
                            int pos = atomicAdd(next_frontier_size, 1);
                            next_frontier[pos] = n_idx;
                        }
                    }
                }
            }
        }

        if (is_collision)
        {
            mask[voxel_idx] = 1; // Upgrade from Water (2) to Collision / Tide (1)
        }
    }

    torch::Tensor compute_flood_fill(
        const torch::Tensor& active_voxel_ids,
        int64_t vx,
        int64_t vy,
        int64_t vz,
        int connectivity
    ) {
        int64_t num_voxels = vx * vy * vz;
        auto options = torch::TensorOptions().device(active_voxel_ids.device()).dtype(torch::kInt32);

        auto mask = torch::full({num_voxels}, -2, options);

        if (active_voxel_ids.numel() > 0)
        {
            mask.index_fill_(0, active_voxel_ids.to(torch::kInt64), -1);
        }

        auto current_frontier = torch::empty({num_voxels}, options);
        auto next_frontier = torch::empty({num_voxels}, options);
        auto frontier_size = torch::zeros({1}, options);
        auto next_frontier_size = torch::zeros({1}, options);

        int threads = NTHREADS;
        int blocks = (num_voxels + threads - 1) / threads;

        init_perimeter_kernel<<<blocks, threads>>>(
            mask.data_ptr<int>(),
            current_frontier.data_ptr<int>(),
            frontier_size.data_ptr<int>(),
            static_cast<int>(vx),
            static_cast<int>(vy),
            static_cast<int>(vz)
        );

        int curr_size = frontier_size.item<int>();

        while (curr_size > 0)
        {
            next_frontier_size.zero_();
            int step_blocks = (curr_size + threads - 1) / threads;

            flood_fill_step_kernel<<<step_blocks, threads>>>(
                mask.data_ptr<int>(),
                current_frontier.data_ptr<int>(),
                curr_size,
                next_frontier.data_ptr<int>(),
                next_frontier_size.data_ptr<int>(),
                static_cast<int>(vx),
                static_cast<int>(vy),
                static_cast<int>(vz),
                connectivity
            );

            curr_size = next_frontier_size.item<int>();
            std::swap(current_frontier, next_frontier);
        }

        return mask.view({vx, vy, vz});
    }

} // namespace ops
