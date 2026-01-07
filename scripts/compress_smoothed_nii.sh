#!/bin/bash
# Compress all .nii files in smoothed directory to .nii.gz
# SPM can directly read .nii.gz files

SMOOTHED_DIR="/Users/yuhiaoki/dev/hit/food-brain/fMRIprep/smoothed"

echo "================================================"
echo "Compressing fMRI NIfTI files with gzip"
echo "================================================"
echo "Directory: $SMOOTHED_DIR"
echo ""

# Count files
TOTAL=$(find "$SMOOTHED_DIR" -name "*.nii" -type f | wc -l | tr -d ' ')
echo "Found $TOTAL .nii files to compress"
echo ""

# Check if pigz (parallel gzip) is available for faster compression
if command -v pigz &> /dev/null; then
    GZIP_CMD="pigz -9"
    echo "Using pigz (parallel gzip) for faster compression"
else
    GZIP_CMD="gzip -9"
    echo "Using standard gzip"
fi

echo ""
echo "Starting compression..."
echo "This may take 30-60 minutes for 239 files"
echo ""

# Compress files one by one
COUNT=0
find "$SMOOTHED_DIR" -name "*.nii" -type f | while read -r FILE; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Compressing: $(basename "$FILE")"

    # Compress the file (creates .nii.gz and removes .nii)
    $GZIP_CMD "$FILE"

    if [ $? -eq 0 ]; then
        echo "  ✓ Compressed successfully"
    else
        echo "  ✗ Failed to compress"
    fi
done

echo ""
echo "================================================"
echo "Compression complete!"
echo "================================================"

# Show space saved
AFTER_SIZE=$(du -sh "$SMOOTHED_DIR" | cut -f1)
echo "New size: $AFTER_SIZE"
echo ""
echo "All .nii files have been compressed to .nii.gz"
echo "SPM can read .nii.gz files directly - no code changes needed"
