#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <string>
#include <vector>
#include "../../data_structure/mesh_bvh.h"
#include "../../data_structure/triangle_mesh.h"

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_voxel_grid(
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

    auto i_v = torch::arange(rx, options_int);
    auto j_v = torch::arange(ry, options_int);
    auto k_v = torch::arange(rz, options_int);
    auto idx_grids_v = torch::meshgrid({i_v, j_v, k_v}, "ij");
    auto out_idx_grids = torch::stack({idx_grids_v[0].flatten(), idx_grids_v[1].flatten(), idx_grids_v[2].flatten()}, -1).contiguous();

    return std::make_tuple(grid_vertices, voxels, out_idx_grids);
}

torch::Tensor compute_grid_normal(torch::Tensor sdf, torch::Tensor grid_vertices, torch::Tensor idx_grids, std::vector<int64_t> res) {
    TORCH_CHECK(res.size() == 3, "res must have 3 elements.");
    
    auto i = idx_grids.select(1, 0);
    auto j = idx_grids.select(1, 1);
    auto k = idx_grids.select(1, 2);
    
    auto rx = res[0];
    auto ry = res[1];
    auto rz = res[2];
    
    auto i_prev = torch::clamp(i - 1, 0, rx - 1);
    auto i_next = torch::clamp(i + 1, 0, rx - 1);
    
    auto j_prev = torch::clamp(j - 1, 0, ry - 1);
    auto j_next = torch::clamp(j + 1, 0, ry - 1);
    
    auto k_prev = torch::clamp(k - 1, 0, rz - 1);
    auto k_next = torch::clamp(k + 1, 0, rz - 1);
    
    auto idx_x_prev = i_prev * ry * rz + j * rz + k;
    auto idx_x_next = i_next * ry * rz + j * rz + k;
    
    auto idx_y_prev = i * ry * rz + j_prev * rz + k;
    auto idx_y_next = i * ry * rz + j_next * rz + k;
    
    auto idx_z_prev = i * ry * rz + j * rz + k_prev;
    auto idx_z_next = i * ry * rz + j * rz + k_next;
    
    auto sdf_flat = sdf.flatten();
    
    auto diff_x = sdf_flat.index_select(0, idx_x_next) - sdf_flat.index_select(0, idx_x_prev);
    auto diff_y = sdf_flat.index_select(0, idx_y_next) - sdf_flat.index_select(0, idx_y_prev);
    auto diff_z = sdf_flat.index_select(0, idx_z_next) - sdf_flat.index_select(0, idx_z_prev);

    auto vx = grid_vertices.select(1, 0);
    auto vy = grid_vertices.select(1, 1);
    auto vz = grid_vertices.select(1, 2);

    auto dx = vx.index_select(0, idx_x_next) - vx.index_select(0, idx_x_prev);
    auto dy = vy.index_select(0, idx_y_next) - vy.index_select(0, idx_y_prev);
    auto dz = vz.index_select(0, idx_z_next) - vz.index_select(0, idx_z_prev);

    dx = torch::where(dx == 0, torch::tensor(1e-5f, dx.options()), dx);
    dy = torch::where(dy == 0, torch::tensor(1e-5f, dy.options()), dy);
    dz = torch::where(dz == 0, torch::tensor(1e-5f, dz.options()), dz);

    auto grad_x = diff_x / dx;
    auto grad_y = diff_y / dy;
    auto grad_z = diff_z / dz;
    
    auto normals = torch::stack({grad_x, grad_y, grad_z}, 1);
    return torch::nn::functional::normalize(normals, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
}

torch::Tensor compute_active_voxels(torch::Tensor voxels, torch::Tensor sdf, float iso) {
    TORCH_CHECK(voxels.device() == sdf.device(), "voxels and sdf must be on the same device.");
    TORCH_CHECK(voxels.dtype() == torch::kInt32, "voxels must be int32.");
    TORCH_CHECK(sdf.dtype() == torch::kFloat32, "sdf must be float32.");
    
    auto voxels_flat = voxels.flatten().to(torch::kInt64);
    auto voxel_sdfs = sdf.index_select(0, voxels_flat).view({-1, 8});
    
    auto below_iso = voxel_sdfs < iso;
    auto any_below = below_iso.any(1);
    auto any_above = (~below_iso).any(1);
    
    auto active_mask = any_below & any_above;
    auto active_indices = torch::nonzero(active_mask).squeeze(1);
    
    return active_indices;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_voxel_grid_from_tmesh(
    std::vector<float> grid_min,
    std::vector<float> grid_max,
    std::vector<int64_t> res,
    TriangleMesh &tmesh
) {
    TORCH_CHECK(grid_min.size() == 3, "grid_min must have 3 elements.");
    TORCH_CHECK(grid_max.size() == 3, "grid_max must have 3 elements.");
    TORCH_CHECK(res.size() == 3, "res must have 3 elements.");

    int64_t rx = res[0];
    int64_t ry = res[1];
    int64_t rz = res[2];

    MeshBVH bvh = tmesh.build_bvh();
    auto vertices = tmesh.get_vertices();
    auto triangles = tmesh.get_triangles();

    auto active_voxel_ids = bvh.get_active_voxel_ids_from_grid(grid_min, grid_max, res, vertices, triangles);

    if (active_voxel_ids.size(0) == 0) {
        return std::make_tuple(
            torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device())),
            torch::empty({0, 8}, torch::TensorOptions().dtype(torch::kInt32).device(vertices.device())),
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(vertices.device()))
        );
    }

    auto vi = active_voxel_ids.div((ry - 1) * (rz - 1), "trunc");
    auto rem = active_voxel_ids.remainder((ry - 1) * (rz - 1));
    auto vj = rem.div(rz - 1, "trunc");
    auto vk = rem.remainder(rz - 1);

    auto v0 = vi * ry * rz + vj * rz + vk;
    auto v1 = (vi + 1) * ry * rz + vj * rz + vk;
    auto v2 = (vi + 1) * ry * rz + (vj + 1) * rz + vk;
    auto v3 = vi * ry * rz + (vj + 1) * rz + vk;
    auto v4 = vi * ry * rz + vj * rz + (vk + 1);
    auto v5 = (vi + 1) * ry * rz + vj * rz + (vk + 1);
    auto v6 = (vi + 1) * ry * rz + (vj + 1) * rz + (vk + 1);
    auto v7 = vi * ry * rz + (vj + 1) * rz + (vk + 1);

    auto active_voxels = torch::stack({v0, v1, v2, v3, v4, v5, v6, v7}, -1);

    torch::Tensor unique_vert_ids, inverse_indices, counts;
    std::tie(unique_vert_ids, inverse_indices, counts) = torch::_unique2(active_voxels.flatten(), true, true, false);

    auto remapped_voxels = inverse_indices.view({-1, 8}).to(torch::kInt32);

    auto u_i = unique_vert_ids.div(ry * rz, "trunc");
    auto u_rem = unique_vert_ids.remainder(ry * rz);
    auto u_j = u_rem.div(rz, "trunc");
    auto u_k = u_rem.remainder(rz);

    float spacing_x = (grid_max[0] - grid_min[0]) / (rx - 1);
    float spacing_y = (grid_max[1] - grid_min[1]) / (ry - 1);
    float spacing_z = (grid_max[2] - grid_min[2]) / (rz - 1);

    auto x = grid_min[0] + u_i.to(torch::kFloat32) * spacing_x;
    auto y = grid_min[1] + u_j.to(torch::kFloat32) * spacing_y;
    auto z = grid_min[2] + u_k.to(torch::kFloat32) * spacing_z;

    auto sparse_grid_vertices = torch::stack({x, y, z}, 1).contiguous();

    return std::make_tuple(sparse_grid_vertices, remapped_voxels, unique_vert_ids);
}

void bind_ds_grid(py::module_& m) {
    m.def("create_voxel_grid", &create_voxel_grid, "Creates a structured 3D voxel grid.",
          py::arg("grid_min"), py::arg("grid_max"), py::arg("res"), py::arg("device_str") = "cuda");
    m.def("create_voxel_grid_from_tmesh", &create_voxel_grid_from_tmesh, "Creates a sparse 3D voxel grid strictly around the surface.",
          py::arg("grid_min"), py::arg("grid_max"), py::arg("res"), py::arg("tmesh"));
    m.def("compute_grid_normal", &compute_grid_normal, "Computes surface normals for a grid.",
          py::arg("sdf"), py::arg("grid_vertices"), py::arg("idx_grids"), py::arg("res"));
    m.def("compute_active_voxels", &compute_active_voxels, "Computes active voxels intersecting the surface",
          py::arg("voxels"), py::arg("sdf"), py::arg("iso"));
}
