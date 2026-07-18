#!/bin/bash

# Array of all available 3D assets in conquer3d
assets=(
  "Alligator" "Armadillo" "Beast" "BeetleAlt" "Beetle" "Bimba"
  "Cheburashka" "Cow" "Fandisk" "HappyBuddha" "Homer" "Horse"
  "Igea" "Lucy" "MaxPlanck" "Nefertiti" "Ogre" "RockerArm"
  "Spot" "StanfordBunny" "Suzanne" "Teapot" "Woody"
  "XYZRGBDragon" "Iphiagenia"
)

resolutions=(128 256 512 1024 1536)

for res in "${resolutions[@]}"
do
  echo "========================================================="
  echo "Processing all assets at resolution: $res"
  echo "========================================================="
  
  for asset in "${assets[@]}"
  do
    echo "---------------------------------------------------------"
    echo "Running Marching Cubes for: $asset at resolution $res"
    echo "---------------------------------------------------------"

    output_file="./${res}/${asset}_${res}.obj"

    if [ -f "$output_file" ]; then
      echo "Skipping $asset at $res: $output_file already exists."
      continue
    fi

    python main.py \
      --input "$asset" \
      --res "$res"

    echo "Finished $asset at resolution $res."
    echo ""
  done
done

echo "All evaluations for all resolutions completed successfully!"
