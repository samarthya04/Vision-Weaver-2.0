#!/bin/bash

# --- Configuration ---
# Set the directory to fix
BASE_DIR="data/celeb_div2k_imgnet"
# Set the scale suffix to remove (e.g., "x4" for X4 scale)
SCALE_SUFFIX="x4"
SCALE_DIR="X4"

# Define the splits to process
SPLITS=("train" "val" "test")

echo "Starting filename cleanup in: ${BASE_DIR}"
echo "Will remove '${SCALE_SUFFIX}.png' suffix from files..."

# Loop over train, val, and test
for split in "${SPLITS[@]}"; do
    
    # Define the Low-Resolution (LR) directory path
    LR_DIR="${BASE_DIR}/${split}/${SCALE_DIR}/LR"

    # Check if the directory exists
    if [ ! -d "$LR_DIR" ]; then
        echo "Warning: Directory not found, skipping: $LR_DIR"
        continue
    fi
    
    echo "--- Processing directory: $LR_DIR ---"
    
    # Loop over all .png files ending with the scale suffix (e.g., *x4.png)
    for file in "$LR_DIR"/*${SCALE_SUFFIX}.png; do
    
        # Check if the file exists (prevents errors if no files match)
        [ -f "$file" ] || continue
        
        filename=$(basename "$file")
        
        # Create the new filename by removing the suffix
        # Example: '0801x4.png' -> '0801.png'
        new_filename="${filename%${SCALE_SUFFIX}.png}.png"
        
        # Rename the file
        if mv "$file" "$LR_DIR/$new_filename"; then
            echo "Renamed: $filename -> $new_filename"
        else
            echo "Error renaming $filename"
        fi
    done
done

echo "---"
echo "Filename cleanup complete."
echo "You can now re-run your training command."