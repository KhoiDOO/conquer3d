#!/bin/bash

# Array of all available 3D assets in conquer3d
assets=(
    "Armadillo" 
    "Beast" 
    "BeetleAlt" 
    "Beetle" 
    "Bimba"
    "Cheburashka" 
    "Cow" 
    "Fandisk" 
    "HappyBuddha" 
    "Homer" 
    "Horse"
    "Igea" 
    "Lucy" 
    "MaxPlanck" 
    "Nefertiti" 
    "Ogre" 
    "RockerArm"
    "Spot" 
    "StanfordBunny" 
    "Suzanne" 
    "Teapot" 
    "XYZRGBDragon" 
    "Iphiagenia"
)

resolutions=(
    128 
    256 
    512 
    1024 
    1536
    2048
)

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

        asset_lower=$(echo "$asset" | tr '[:upper:]' '[:lower:]')
        output_file="./${res}/${asset_lower}.obj"

        if [ -f "$output_file" ]; then
        echo "Skipping $asset at $res: $output_file already exists."
        continue
        fi

        python main.py \
        --input "$asset" \
        --res "$res" \
        --chunk_size 100000

        echo "Finished $asset at resolution $res."
        echo ""
    done
done

echo "All evaluations for all resolutions completed successfully!"
