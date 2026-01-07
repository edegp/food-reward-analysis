#!/bin/bash

# Copy confounds files from external drive to local need_info directory
# This script copies *_desc-confounds_timeseries.tsv files needed for DNN analysis

set -e

SOURCE_DIR="/Volumes/Transcend/hit/food-brain/fMRIprep/derivatives"
TARGET_DIR="./fMRIprep/need_info"

echo "Copying confounds files from external drive..."

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    echo "Please make sure the external drive is mounted."
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Copy all confounds files maintaining directory structure
echo "Searching for confounds files..."
file_count=0

for sub_dir in "$SOURCE_DIR"/sub-*/; do
    if [ -d "$sub_dir" ]; then
        sub_name=$(basename "$sub_dir")
        echo "Processing $sub_name..."

        for ses_dir in "$sub_dir"ses-*/func/; do
            if [ -d "$ses_dir" ]; then
                ses_name=$(basename "$(dirname "$ses_dir")")

                # Create target directory structure
                target_sub_dir="$TARGET_DIR/$sub_name/$ses_name/func"
                mkdir -p "$target_sub_dir"

                # Copy confounds files
                for confounds_file in "$ses_dir"*_desc-confounds_timeseries.tsv; do
                    if [ -f "$confounds_file" ]; then
                        cp "$confounds_file" "$target_sub_dir/"
                        echo "  Copied: $(basename "$confounds_file")"
                        ((file_count++))
                    fi
                done
            fi
        done
    fi
done

echo ""
echo "Done! Copied $file_count confounds files to $TARGET_DIR"