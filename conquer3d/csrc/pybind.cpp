#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_primitive_gs(py::module_& m);
void bind_primitive_pgs(py::module_& m);

void bind_ds_kdtree(py::module_& m);
void bind_ds_bvh(py::module_& m);
void bind_ds_gs_bvh(py::module_& m);
void bind_ds_pgs_bvh(py::module_& m);
void bind_ds_mesh_bvh(py::module_& m);
void bind_ds_triangle_mesh(py::module_& m);
void bind_ds_grid(py::module_& m);

void bind_creation_triangle_creation(py::module_& m);

void bind_ops_mc(py::module_& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Geocutool Python bindings";

    bind_primitive_gs(m);
    bind_primitive_pgs(m);

    bind_ds_kdtree(m);
    bind_ds_bvh(m);
    bind_ds_gs_bvh(m);
    bind_ds_pgs_bvh(m);
    bind_ds_mesh_bvh(m);
    bind_ds_triangle_mesh(m);
    bind_ds_grid(m);
    
    bind_creation_triangle_creation(m);
    
    bind_ops_mc(m);
}