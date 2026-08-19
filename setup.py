import glob
import os
import subprocess
import sys
import multiprocessing

# Limit the number of CPUs used for building to half of the available CPUs
num_cpus = multiprocessing.cpu_count()
os.environ["MAX_JOBS"] = str(max(1, num_cpus // 2))

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# Monkey-patch PyTorch's CUDA version check to allow building the extension 
# even if the local nvcc version doesn't perfectly match the PyTorch compiled CUDA version.
import torch.utils.cpp_extension
if hasattr(torch.utils.cpp_extension, "_check_cuda_version"):
    torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None


def get_extensions():
    """Build C++ and CUDA extensions for the conquer3d package."""
    
    csrc_dir = os.path.join("conquer3d", "csrc")
    main_file = [os.path.join(csrc_dir, "pybind.cpp")]
    source_cuda = glob.glob(os.path.join(csrc_dir, "**", "*.cu"), recursive=True)
    source_cpp = [
        f for f in glob.glob(os.path.join(csrc_dir, "**", "*.cpp"), recursive=True)
        if os.path.basename(f) != "pybind.cpp"
    ]
    
    sources = main_file + source_cpp
    extension = CppExtension

    define_macros = []
    extra_compile_args = {}
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1"
    
    # Try to find CUDA_HOME: first check standard location, then CONDA_PREFIX
    cuda_home = CUDA_HOME
    if cuda_available and cuda_home is None:
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            print(f"Checking for CUDA in CONDA_PREFIX: {conda_prefix}")
            # Check if CUDA headers exist in the conda environment
            cuda_include = os.path.join(conda_prefix, "include", "cuda.h")
            cuda_lib = os.path.join(conda_prefix, "lib", "libcudart.so")
            if os.path.isfile(cuda_include) or os.path.isfile(cuda_lib):
                cuda_home = conda_prefix
                os.environ["CUDA_HOME"] = cuda_home
                torch.utils.cpp_extension.CUDA_HOME = cuda_home
                print(f"Found CUDA in conda environment: {cuda_home}")
    
    if (cuda_available and cuda_home is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        if cuda_home and not os.getenv("CUDA_HOME"):
            os.environ["CUDA_HOME"] = cuda_home
            torch.utils.cpp_extension.CUDA_HOME = cuda_home
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            try:
                major, minor = torch.cuda.get_device_capability()
                nvcc_flags = [f"-arch=sm_{major}{minor}"]
            except Exception:
                # If no GPU is available during build, rely on TORCH_CUDA_ARCH_LIST or defaults
                nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(" ")
        
        # Suppress harmless nvcc warnings (like conda's compiler-bindir redefinition)
        nvcc_flags.append("-w")
        nvcc_flags.append("-allow-unsupported-compiler")
        
        extra_compile_args = {
            "cxx": ["-O3", "-Wno-attributes"],
            "nvcc": nvcc_flags,
        }
        print(f"Building CUDA extension with CUDA_HOME: {cuda_home}")
    else:
        print("Building CPU-only extension (CUDA not available)")

    sources = [s for s in sources]
    include_dirs = [os.path.join("conquer3d", "csrc")]
    
    if cuda_available and cuda_home is not None:
        target_inc = os.path.join(cuda_home, "targets", "x86_64-linux", "include")
        if os.path.isdir(target_inc):
            include_dirs.append(target_inc)
        conda_inc = os.path.join(cuda_home, "include")
        if os.path.isdir(conda_inc) and conda_inc not in include_dirs:
            include_dirs.append(conda_inc)
    
    # Ensure torch libraries are in the RPATH
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    library_dirs = [torch_lib_path]
    runtime_library_dirs = [torch_lib_path]

    print("sources:", sources)
    print("include_dirs:", include_dirs)

    ext_modules = [
        extension(
            "conquer3d._C",
            sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=runtime_library_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

# Create a base class using your existing no_python_abi_suffix logic
BaseBuildExt = BuildExtension.with_options(no_python_abi_suffix=True)

class CustomBuildExt(BaseBuildExt):
    """Custom build extension to generate PyBind11 stubs after compilation."""
    def run(self):
        # 1. Run the normal compilation process
        super().run()
        
        # 2. Automatically generate the .pyi stubs
        print("\n--- Generating PyBind11 Stubs ---")
        import sys, os, subprocess
        build_lib = os.path.abspath(self.build_lib)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{build_lib}:{env.get('PYTHONPATH', '')}"
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pybind11_stubgen", 
                "conquer3d._C", 
                "-o", 
                ".",
                "--ignore-all-errors"
            ], env=env, cwd=build_lib)
            print(f"Successfully generated conquer3d/_C.pyi in {build_lib}!")
        except Exception as e:
            print(f"Warning: Failed to generate .pyi stubs automatically ({e}). Continuing build.")
        print("---------------------------------\n")

setup(
    packages=find_packages(exclude=["examples", "tests"]),
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": CustomBuildExt,
    },
    include_package_data=True,
    package_data={
        "conquer3d": ["*.pyi", "*.so", "*.pyd"],
    },
    zip_safe=False
)