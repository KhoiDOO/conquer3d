#!/bin/bash

# Parse arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <mode>"
    echo "Available modes: base, fairness, odt, fairodt, vol, fairvol, odtvol, fairodtvol"
    exit 1
fi

MODE=$1

# Array of all available 3D assets in conquer3d
assets=(
  "Alligator" "Armadillo" "Beast" "BeetleAlt" "Beetle" "Bimba"
  "Cheburashka" "Cow" "Fandisk" "HappyBuddha" "Homer" "Horse"
  "Igea" "Lucy" "MaxPlanck" "Nefertiti" "Ogre" "RockerArm"
  "Spot" "StanfordBunny" "Suzanne" "Teapot" "Woody"
  "XYZRGBDragon" "Iphiagenia"
)

# Output directory for the results based on mode
mkdir -p ./${MODE}

# Determine flags based on mode
FLAGS=""
if [ "$MODE" = "base" ]; then
    FLAGS=""
elif [ "$MODE" = "fairness" ]; then
    FLAGS="--use_fairness"
elif [ "$MODE" = "odt" ]; then
    FLAGS="--use_eodt"
elif [ "$MODE" = "fairodt" ]; then
    FLAGS="--use_fairness --use_eodt"
elif [ "$MODE" = "vol" ]; then
    FLAGS="--use_evol"
elif [ "$MODE" = "fairvol" ]; then
    FLAGS="--use_fairness --use_evol"
elif [ "$MODE" = "odtvol" ]; then
    FLAGS="--use_eodt --use_evol"
elif [ "$MODE" = "fairodtvol" ]; then
    FLAGS="--use_fairness --use_eodt --use_evol"
else
    echo "Unknown mode: $MODE"
    echo "Available modes: base, fairness, odt, fairodt, vol, fairvol, odtvol, fairodtvol"
    exit 1
fi

# Run DMTet optimization for each asset
for asset in "${assets[@]}"
do
  echo "========================================================="
  echo "Running Differentiable Marching Tetrahedra ($MODE) for: $asset"
  echo "========================================================="

  # Format output filename: lowercase with .obj extension
  output_file="./${MODE}/${asset,,}.obj"

  if [ -f "$output_file" ]; then
    echo "Skipping $asset: $output_file already exists."
    continue
  fi

  python main.py \
    --input "$asset" \
    --output "$output_file" \
    $FLAGS

  echo "Finished $asset. Saved to $output_file."
  echo ""
done

echo "All 25 optimizations completed successfully for mode: $MODE!"
