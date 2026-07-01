#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <string>
#include <vector>

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor> create_voxel_grid(
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> res,
    std::string device_str
) {
    TORCH_CHECK(grid_min.size() == 3, "grid_min must have 3 elements.");
    TORCH_CHECK(grid_max.size() == 3, "grid_max must have 3 elements.");
    TORCH_CHECK(res.size() == 3, "res must have 3 elements.");

    c10::Device device(device_str);
    auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

    int64_t rx = res[0];
    int64_t ry = res[1];
    int64_t rz = res[2];

    auto x = torch::linspace(grid_min[0], grid_max[0], rx, options);
    auto y = torch::linspace(grid_min[1], grid_max[1], ry, options);
    auto z = torch::linspace(grid_min[2], grid_max[2], rz, options);

    auto grids = torch::meshgrid({x, y, z}, "ij");
    auto grid_x = grids[0].flatten();
    auto grid_y = grids[1].flatten();
    auto grid_z = grids[2].flatten();
    
    auto grid_vertices = torch::stack({grid_x, grid_y, grid_z}, 1).contiguous();

    auto options_int = options.dtype(torch::kInt64);
    auto i = torch::arange(rx - 1, options_int);
    auto j = torch::arange(ry - 1, options_int);
    auto k = torch::arange(rz - 1, options_int);

    auto idx_grids = torch::meshgrid({i, j, k}, "ij");
    auto vi = idx_grids[0];
    auto vj = idx_grids[1];
    auto vk = idx_grids[2];

    // Voxel connectivity indexing mapping matching mc.cu
    auto v0 = vi * ry * rz + vj * rz + vk;
    auto v1 = (vi + 1) * ry * rz + vj * rz + vk;
    auto v2 = (vi + 1) * ry * rz + (vj + 1) * rz + vk;
    auto v3 = vi * ry * rz + (vj + 1) * rz + vk;
    auto v4 = vi * ry * rz + vj * rz + (vk + 1);
    auto v5 = (vi + 1) * ry * rz + vj * rz + (vk + 1);
    auto v6 = (vi + 1) * ry * rz + (vj + 1) * rz + (vk + 1);
    auto v7 = vi * ry * rz + (vj + 1) * rz + (vk + 1);

    auto voxels = torch::stack({v0, v1, v2, v3, v4, v5, v6, v7}, -1).view({-1, 8}).contiguous().to(torch::kInt32);

    return std::make_tuple(grid_vertices, voxels);
}

void bind_ds_grid(py::module_& m) {
    m.def("create_voxel_grid", &create_voxel_grid, "Creates a structured 3D voxel grid.",
          py::arg("grid_min"), py::arg("grid_max"), py::arg("res"), py::arg("device_str") = "cuda");
}
