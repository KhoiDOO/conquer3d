#!/usr/bin/env bash
# ==============================================================================
# Script: create_envs.sh
# Description: Automatically creates Conda environments for building conquer3d
#              wheels across Python versions (3.10-3.14) and PyTorch/CUDA pairs.
#              Includes pip, gxx_linux-64=13, gcc_linux-64=13, sparsehash, and cuda-toolkit.
# ==============================================================================

set -e

# Target matrix configuration
PYTHON_VERSIONS=(
    "3.10" 
    "3.11" 
    "3.12" 
    "3.13" 
    "3.14"
)
TORCH_CUDA_PAIRS=(
    "2.8.0:12.8.2:cu128"
    "2.11.0:12.8.2:cu128"
)

echo "=== conquer3d Conda Environment Creator ==="

for pyver in "${PYTHON_VERSIONS[@]}"; do
    for tuple in "${TORCH_CUDA_PAIRS[@]}"; do
        # Parse tuple: torch_ver:cuda_ver:cuda_tag
        torch_ver=$(echo "$tuple" | cut -d':' -f1)
        cuda_ver=$(echo "$tuple" | cut -d':' -f2)
        cuda_tag=$(echo "$tuple" | cut -d':' -f3)
        
        # Clean versions for environment name
        py_clean=$(echo "$pyver" | tr -d '.')
        torch_clean=$(echo "$torch_ver" | tr -d '.')
        
        env_name="c3d_py${py_clean}_pt${torch_clean}_${cuda_tag}"
        
        echo ""
        echo "============================================================"
        echo "Creating Conda environment: ${env_name}"
        echo "  Python:       ${pyver}"
        echo "  PyTorch:      ${torch_ver}"
        echo "  CUDA Toolkit: ${cuda_ver} (${cuda_tag})"
        echo "  GCC/G++:      13 (conda-forge)"
        echo "============================================================"
        
        # Create environment with pip, GCC 13 and sparsehash from conda-forge if it doesn't already exist
        if conda info --envs | grep -q "^${env_name} "; then
            echo "Environment '${env_name}' already exists."
        else
            conda create -c conda-forge -c nvidia -n "${env_name}" "python=${pyver}" pip gxx_linux-64=13 gcc_linux-64=13 cuda-toolkit==${cuda_ver} sparsehash -y
        fi
        
        # Install CUDA toolkit from nvidia channel
        # conda install -n "${env_name}" -c nvidia "cuda-toolkit==${cuda_ver}" -y || true
        
        # Determine matching torchvision version
        if [ "${torch_ver}" == "2.8.0" ]; then
            torchvision_ver="0.23.0"
        elif [ "${torch_ver}" == "2.11.0" ]; then
            torchvision_ver="0.26.0"
        else
            torchvision_ver="0.26.0"
        fi

        # Install build dependencies and PyTorch
        conda run -n "${env_name}" python -m pip install --upgrade pip setuptools wheel ninja pybind11-stubgen numpy scipy trimesh tqdm gdown pillow || true
        conda run -n "${env_name}" pip install "torch==${torch_ver}" "torchvision==${torchvision_ver}" --index-url "https://download.pytorch.org/whl/${cuda_tag}"
        
        echo "Finished setting up '${env_name}'."
    done
done

echo ""
echo "=== All Conda environments created successfully! ==="
