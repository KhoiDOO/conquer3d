#!/usr/bin/env bash
# ==============================================================================
# Script: clear.sh
# Description: Automatically finds and removes all Conda environments
#              starting with 'c3d_'.
# ==============================================================================

set -e

echo "=== conquer3d Conda Environment Cleanup ==="

# Find all conda environment names starting with c3d_
ENVS=$(conda env list | awk '{print $1}' | grep '^c3d_' || true)

if [ -z "$ENVS" ]; then
    echo "No Conda environments starting with 'c3d_' were found."
    exit 0
fi

echo "Found the following Conda environments to remove:"
echo "$ENVS"
echo ""

for env_name in $ENVS; do
    echo "Removing Conda environment: ${env_name}..."
    conda env remove -n "${env_name}" -y
    echo "Removed '${env_name}'."
done

echo ""
echo "=== All 'c3d_' Conda environments removed successfully! ==="
