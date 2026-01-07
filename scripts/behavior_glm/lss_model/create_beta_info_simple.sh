#!/bin/bash
# Create beta_info.csv files from behavioral data

ROOT_DIR="/Users/yuhiaoki/dev/hit/food-brain"
BEHAVIOR_DIR="$ROOT_DIR/Food_Behavior"
RESULT_DIR="$ROOT_DIR/results/first_level_analysis"

# Process each subject
for SUB_NUM in {1..20}; do
    SUB_ID=$(printf "%03d" $SUB_NUM)
    echo "Processing subject $SUB_ID..."

    # Find latest LSS GLM directory
    LSS_DIR="$RESULT_DIR/sub-$SUB_ID/glm_model/lss_glm"
    if [ ! -d "$LSS_DIR" ]; then
        echo "  WARNING: LSS GLM directory not found"
        continue
    fi

    # Get latest timestamp directory
    LATEST_DIR=$(ls -1dt "$LSS_DIR"/*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_DIR" ]; then
        echo "  WARNING: No LSS GLM results found"
        continue
    fi

    # Check if beta files exist
    BETA_COUNT=$(ls -1 "$LATEST_DIR"/beta_*.nii 2>/dev/null | wc -l)
    if [ "$BETA_COUNT" -eq 0 ]; then
        echo "  WARNING: No beta files found"
        continue
    fi

    echo "  Found $BETA_COUNT beta files in $LATEST_DIR"

    # Create output directory
    OUTPUT_DIR="$LATEST_DIR/beta_values"
    mkdir -p "$OUTPUT_DIR"

    # Create beta_info.csv
    # We'll extract image names from behavioral data files
    BEHAV_DIR="$BEHAVIOR_DIR/sub-$SUB_ID"

    # Initialize CSV
    echo "image_id,beta_index" > "$OUTPUT_DIR/beta_info.csv"

    # Counter for beta index
    BETA_IDX=1

    # Process each run's behavioral data
    for CSV_FILE in "$BEHAV_DIR"/GLM_all_combined_*.csv; do
        if [ ! -f "$CSV_FILE" ]; then
            continue
        fi

        # Extract ImageName column (assuming it's in the CSV)
        # Skip header, get ImageName column, remove missing values
        tail -n +2 "$CSV_FILE" | awk -F',' '{print $NF}' | grep -v '^$' | sort -u | while read IMG_ID; do
            # Format image ID (pad with zeros if needed)
            FORMATTED_ID=$(printf "%04d" "$IMG_ID" 2>/dev/null || echo "$IMG_ID")
            echo "$FORMATTED_ID,$BETA_IDX" >> "$OUTPUT_DIR/beta_info.csv"
            BETA_IDX=$((BETA_IDX + 1))
        done
    done

    echo "  Created: $OUTPUT_DIR/beta_info.csv"
done

echo "Done!"
