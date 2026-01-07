#!/usr/bin/env bash
set -euo pipefail

# Best effort increase for "Too many open files" crashes inside fMRIPrep
NOFILE_LIMIT=${NOFILE_LIMIT:-524288}
if ! ulimit -n "${NOFILE_LIMIT}" 2>/dev/null; then
  echo "[fMRIPrep] Warning: Unable to raise NOFILE_LIMIT to ${NOFILE_LIMIT}. Continuing with current limit."
fi

# Read participants from environment variable (space-separated)
PARTICIPANTS_STR=${PARTICIPANTS:-}
if [[ -z "${PARTICIPANTS_STR}" ]]; then
  echo "[fMRIPrep] ERROR: PARTICIPANTS environment variable is empty or not set." >&2
  echo "Set it to a space-separated list, e.g.: PARTICIPANTS=\"09 10 11 12 13 14 15 16\"" >&2
  exit 1
fi

# Optional thread overrides via env; default to 12 and 6 if not provided
NTHREADS_VAL=${NTHREADS:-12}
OMP_NTHREADS_VAL=${OMP_NTHREADS:-4}

# Convert to array and process in chunks of 2
read -r -a PARTICIPANTS_ARR <<< "${PARTICIPANTS_STR}"
TOTAL=${#PARTICIPANTS_ARR[@]}
echo "[fMRIPrep] Participants (${TOTAL}): ${PARTICIPANTS_STR}"
BATCH_SIZE=4


for ((i=0; i<${TOTAL}; i+=BATCH_SIZE)); do
  # Slice up to 2 items safely
  BATCH_ARR=("${PARTICIPANTS_ARR[@]:i:BATCH_SIZE}")
  echo "[fMRIPrep] Running batch: ${BATCH_ARR[*]}"

  # Clear work dir between batches to avoid crosstalk
  if (( i >= 12 )) && [ -d /work ]; then
    rm -rf /work/*
  fi

  fmriprep /data /out participant \
    --participant-label "${BATCH_ARR[@]}" \
    --session-label "01" \
    --fs-license-file /opt/freesurfer/license.txt \
    --work-dir /work \
    --nthreads "${NTHREADS_VAL}" \
    --omp-nthreads "${OMP_NTHREADS_VAL}"

  echo "[fMRIPrep] Completed batch: ${BATCH_ARR[*]}"
  echo
done

echo "[fMRIPrep] All batches completed."
