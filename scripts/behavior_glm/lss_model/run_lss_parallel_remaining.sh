#!/bin/bash
# Run LSS GLM for remaining subjects (002-020) in parallel (2 groups)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/lss_glm"

# Create log directory
mkdir -p "$LOG_DIR"

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo "LSS GLM Parallel Execution (Remaining)"
echo "========================================"
echo "Starting parallel execution at: $(date)"
echo "Log directory: $LOG_DIR"
echo ""

# Function to run LSS GLM for a subject range
run_lss_batch() {
    local start_idx=$1
    local end_idx=$2
    local batch_name=$3
    local log_file="$LOG_DIR/lss_glm_${batch_name}_${TIMESTAMP}.log"

    echo "[$(date)] Starting batch $batch_name (subjects $start_idx-$end_idx)"

    /Applications/MATLAB_R2025a.app/bin/matlab -batch \
        "addpath('$SCRIPT_DIR'); run_lss_glm($start_idx, $end_idx);" \
        > "$log_file" 2>&1

    local exit_code=$?
    echo "[$(date)] Batch $batch_name completed with exit code: $exit_code"
    return $exit_code
}

# Run two batches in parallel
# Batch 1: subjects 2-10
# Batch 2: subjects 11-20
run_lss_batch 2 10 "batch1_remaining" &
PID1=$!

run_lss_batch 11 20 "batch2_remaining" &
PID2=$!

echo "Batch 1 (subjects 2-10) started with PID: $PID1"
echo "Batch 2 (subjects 11-20) started with PID: $PID2"
echo ""
echo "Monitoring progress..."
echo "You can tail the logs with:"
echo "  tail -f $LOG_DIR/lss_glm_batch1_remaining_${TIMESTAMP}.log"
echo "  tail -f $LOG_DIR/lss_glm_batch2_remaining_${TIMESTAMP}.log"
echo ""

# Wait for both processes to complete
wait $PID1
EXIT1=$?

wait $PID2
EXIT2=$?

echo ""
echo "========================================"
echo "Execution Summary"
echo "========================================"
echo "Batch 1 (subjects 2-10): Exit code $EXIT1"
echo "Batch 2 (subjects 11-20): Exit code $EXIT2"
echo "Completed at: $(date)"

# Check if both succeeded
if [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ]; then
    echo "Status: SUCCESS"
    echo ""
    echo "Verifying results..."

    # Count completed subjects
    COMPLETED=$(find "$PROJECT_ROOT/results/first_level_analysis" -type f -name "SPM.mat" -path "*/lss_glm/*" | wc -l | tr -d ' ')
    echo "Found $COMPLETED completed LSS GLM analyses"

    exit 0
else
    echo "Status: FAILED"
    echo "Check log files for details"
    exit 1
fi
