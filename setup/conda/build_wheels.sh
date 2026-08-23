#!/usr/bin/env bash
# ==============================================================================
# Script: build_wheels.sh
# Description: Loops over Conda build environments created by create_envs.sh,
#              compiles CUDA wheels into wheels/, and renames them to match the
#              naming convention defined in .github/workflows/build_wheels.yml.
#              Skips building if the final target wheel already exists.
# ==============================================================================

set -e

# Find root directory of conquer3d repository
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"

# Target matrix configuration
PYTHON_VERSIONS=(
    # "3.10" 
    "3.11" 
    # "3.12" 
    # "3.13" 
    # "3.14"
)
TORCH_CUDA_PAIRS=(
    # "2.8.0:cu128"
    "2.11.0:cu128"
)

mkdir -p "${WHEELS_DIR}"
cd "${REPO_ROOT}"

echo "=== conquer3d Local Multi-Python/CUDA Wheel Builder ==="
echo "Output Directory: ${WHEELS_DIR}"

for pyver in "${PYTHON_VERSIONS[@]}"; do
    for pair in "${TORCH_CUDA_PAIRS[@]}"; do
        torch_ver="${pair%%:*}"
        cuda_tag="${pair##*:}"
        
        py_clean=$(echo "$pyver" | tr -d '.')
        torch_clean=$(echo "$torch_ver" | tr -d '.')
        
        env_name="c3d_py${py_clean}_pt${torch_clean}_${cuda_tag}"
        
        # Check if final wheel already exists in wheels/
        existing_whl=$(ls "${WHEELS_DIR}"/c3d-*+pt${torch_clean}${cuda_tag}-cp${py_clean}-cp${py_clean}-linux_x86_64.whl 2>/dev/null || true)
        if [ -n "${existing_whl}" ]; then
            echo ""
            echo "============================================================"
            echo "Skipping ${env_name}: Wheel already exists ($(basename ${existing_whl}))"
            echo "============================================================"
            continue
        fi

        # Verify environment exists
        if ! conda info --envs | grep -q "^${env_name} "; then
            echo "Skipping missing Conda environment '${env_name}'. Run setup/conda/create_envs.sh first."
            continue
        fi
        
        # Verify torch is installed in target environment
        if ! conda run -n "${env_name}" python -c "import torch" >/dev/null 2>&1; then
            if [ "${torch_ver}" == "2.8.0" ]; then
                torchvision_ver="0.23.0"
            elif [ "${torch_ver}" == "2.11.0" ]; then
                torchvision_ver="0.26.0"
            else
                torchvision_ver="0.26.0"
            fi
            echo "Installing torch into environment '${env_name}'..."
            conda run -n "${env_name}" pip install "torch==${torch_ver}" "torchvision==${torchvision_ver}" --index-url "https://download.pytorch.org/whl/${cuda_tag}"
        fi
        
        echo ""
        echo "============================================================"
        echo "Building Wheel for Environment: ${env_name}"
        echo "============================================================"
        
        # Build wheel inside target conda environment
        FORCE_CUDA=1 MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
        conda run -n "${env_name}" python setup.py bdist_wheel --dist-dir "${WHEELS_DIR}"
        
        # Rename newly built wheel: conquer3d-<ver>-cp... -> c3d-<ver>+pt<torch_tag><cuda_whl_tag>-cp...
        for f in "${WHEELS_DIR}"/conquer3d-*.whl; do
            if [ -f "$f" ]; then
                new_f=$(echo "$f" | sed "s/conquer3d-/c3d-/" | sed "s/-cp/+pt${torch_clean}${cuda_tag}-cp/")
                mv "$f" "$new_f"
                echo "Exported wheel: $(basename "$new_f")"
            fi
        done
    done
done

echo ""
echo "=== All wheels built and exported to ${WHEELS_DIR}! ==="
ls -la "${WHEELS_DIR}"
