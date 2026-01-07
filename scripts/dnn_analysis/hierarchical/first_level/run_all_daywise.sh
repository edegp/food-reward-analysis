#!/bin/bash
# Run Day-wise Hierarchical DNN GLM for all 31 subjects
# Usage: ./run_all_daywise.sh

cd /Users/yuhiaoki/dev/hit/food-brain

for source in clip convnext; do
    echo "=========================================="
    echo "Starting $source analysis..."
    echo "=========================================="

    for subj in $(seq -f "%03g" 1 31); do
        echo ""
        echo "[$source] Subject $subj started at $(date '+%Y-%m-%d %H:%M:%S')"

        matlab -nodisplay -nosplash -r "addpath('scripts/dnn_analysis/hierarchical/first_level'); try, run_glm_daywise('$subj', '$source'); catch ME, fprintf('ERROR: %s\n', ME.message); end; exit;" 2>&1

        echo "[$source] Subject $subj finished at $(date '+%Y-%m-%d %H:%M:%S')"
    done

    echo ""
    echo "=========================================="
    echo "$source analysis complete!"
    echo "=========================================="
done

echo ""
echo "All analyses complete!"
