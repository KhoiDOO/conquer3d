#!/usr/bin/env bash
# ==============================================================================
# Script: upload_wheels.sh
# Description: Uploads all prebuilt CUDA wheels from wheels/ directly to a
#              GitHub Release using the GitHub CLI (gh).
# Usage:       bash setup/conda/upload_wheels.sh [release_tag]
# Example:     bash setup/conda/upload_wheels.sh v0.6.7
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WHEELS_DIR="${REPO_ROOT}/wheels"

# Target tag defaults to latest git tag or version from pyproject.toml
TAG="${1:-$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.6.7")}"

echo "=== conquer3d GitHub Release Wheel Uploader ==="
echo "Target Release Tag: ${TAG}"
echo "Wheels Directory:   ${WHEELS_DIR}"

if [ ! -d "${WHEELS_DIR}" ] || [ -z "$(ls -A "${WHEELS_DIR}"/*.whl 2>/dev/null)" ]; then
    echo "Error: No .whl files found in ${WHEELS_DIR}. Run setup/conda/build_wheels.sh first."
    exit 1
fi

echo ""
echo "Uploading prebuilt wheels to GitHub Release ${TAG}..."
gh release upload "${TAG}" "${WHEELS_DIR}"/*.whl --clobber

echo ""
echo "=== Successfully uploaded all wheels to GitHub Release ${TAG}! ==="
