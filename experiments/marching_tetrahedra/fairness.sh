#!/bin/bash

# Array of all available 3D assets in conquer3d
assets=(
  "Alligator" "Armadillo" "Beast" "BeetleAlt" "Beetle" "Bimba"
  "Cheburashka" "Cow" "Fandisk" "HappyBuddha" "Homer" "Horse"
  "Igea" "Lucy" "MaxPlanck" "Nefertiti" "Ogre" "RockerArm"
  "Spot" "StanfordBunny" "Suzanne" "Teapot" "Woody"
  "XYZRGBDragon" "Iphiagenia"
)

# Output directory for the results
mkdir -p ./fairness

# Run DMTet optimization for each asset
for asset in "${assets[@]}"
do
  echo "========================================================="
  echo "Running Differentiable Marching Tetrahedra for: $asset"
  echo "========================================================="

  # Format output filename: lowercase with .obj extension
  # (e.g., Armadillo -> ./fairness/armadillo.obj)
  output_file="./fairness/${asset,,}.obj"

  if [ -f "$output_file" ]; then
    echo "Skipping $asset: $output_file already exists."
    continue
  fi

  python main.py \
    --input "$asset" \
    --output "$output_file" \
    --use_fairness

  echo "Finished $asset. Saved to $output_file."
  echo ""
done

echo "All 25 optimizations completed successfully!"