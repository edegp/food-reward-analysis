#!/usr/bin/env bash
set -euo pipefail

# Configuration
BIDS_ROOT="${1:-./fMRIprep/bids}"   # pass root if different
DATASET_DESCRIPTION_NAME="Food valuation fMRI Dataset"
BIDS_VERSION="1.8.0"
AUTHORS='["Yuhi Aoki"]'
# MODE: all (use all functional runs) or first2 (limit to first 2 runs)
MODE="${MODE:-all}"

if [ ! -d "$BIDS_ROOT" ]; then
  echo "[ERROR] BIDS root '$BIDS_ROOT' not found" >&2
  exit 1
fi

echo "[INFO] Using BIDS root: $BIDS_ROOT"

# Create dataset_description.json at root if missing
if [ ! -f "$BIDS_ROOT/dataset_description.json" ]; then
  cat > "$BIDS_ROOT/dataset_description.json" <<JSON
{
  "Name": "$DATASET_DESCRIPTION_NAME",
  "BIDSVersion": "$BIDS_VERSION",
  "Authors": $AUTHORS
}
JSON
  echo "[INFO] Created dataset_description.json"
else
  echo "[INFO] dataset_description.json already exists (skipped)"
fi

# Function to update a single fmap json
update_json() {
  local json_file="$1"; shift
  local intended_list=("$@")
  if [ ! -f "$json_file" ]; then
    echo "[WARN] Missing fmap JSON: $json_file" >&2
    return 0
  fi

  # Build JSON fragment
  local tmp_file
  tmp_file="${json_file}.tmp.$$"

  # Escape paths for JSON (they don't include quotes so simple quoting is fine)
  printf '[INFO] Updating IntendedFor in %s\n' "$json_file"

  # If python available use it (safer), else fallback to sed injection
  if command -v python3 >/dev/null 2>&1; then
  python3 - "$json_file" "${intended_list[@]}" <<'PYCODE'
import json,sys,os
json_path=sys.argv[1]
paths=sys.argv[2:]
with open(json_path,'r') as f:
  data=json.load(f)
data['IntendedFor']=paths
with open(json_path,'w') as f:
  json.dump(data,f,indent=4,ensure_ascii=False)
  f.write('\n')
PYCODE
  else
    # crude fallback: remove existing IntendedFor block then append before final }
    # This fallback assumes final } at end of file.
    sed -i '' '/"IntendedFor" *:/,/],/d' "$json_file" 2>/dev/null || true
    {
      sed '/^[[:space:]]*}$/d' "$json_file"
      echo '    "IntendedFor": ['
      local first=1
      for p in "${intended_list[@]}"; do
        if [ $first -eq 0 ]; then echo ","; fi | cat >/dev/null; # no-op
        first=0
        printf '    "%s"' "$p"
      done | sed 's/^/        /'
      echo ''
      echo '    ],'
      echo '}'
    } > "$tmp_file" && mv "$tmp_file" "$json_file"
  fi
}

subject_count=0
update_count=0

for sub_dir in "$BIDS_ROOT"/sub-*; do
  [ -d "$sub_dir" ] || continue
  sub=$(basename "$sub_dir")
  [[ "$sub" == tmp_* ]] && continue
  subject_count=$((subject_count+1))
  echo "[INFO] Subject: $sub"

  for ses_dir in "$sub_dir"/ses-*; do
    [ -d "$ses_dir" ] || continue
    ses=$(basename "$ses_dir")
    func_dir="$ses_dir/func"
    fmap_dir="$ses_dir/fmap"
    [ -d "$func_dir" ] || { echo "[WARN] No func dir for $sub $ses"; continue; }
    [ -d "$fmap_dir" ] || { echo "[WARN] No fmap dir for $sub $ses"; continue; }

    # macOS bash doesn't have `mapfile`/`readarray`. Use a glob loop to collect files.
    bold_files=()
    for f in "$func_dir"/${sub}_${ses}_task-*_run-*_bold.nii.gz; do
      [ -f "$f" ] || continue
      bold_files+=("$f")
    done
    if [ ${#bold_files[@]} -eq 0 ]; then
      echo "[WARN] No bold runs in $func_dir"; continue
    fi

    # Build relative IntendedFor paths (relative to sub-XX root per BIDS spec: should not include leading subject folder)
    intended=()
    sel_files=("${bold_files[@]}")
    if [ "$MODE" = "first2" ] && [ ${#sel_files[@]} -gt 2 ]; then
      sel_files=(${sel_files[@]:0:2})
    fi
    for f in "${sel_files[@]}"; do
      bf=$(basename "$f")
      # relative path from subject root
      intended+=("${ses}/func/${bf}")
    done

    # Update both AP and PA (dir-AP / dir-PA) jsons if present
    for dircode in AP PA; do
      jf="$fmap_dir/${sub}_${ses}_dir-${dircode}_epi.json"
      if [ -f "$jf" ]; then
        update_json "$jf" "${intended[@]}"
        update_count=$((update_count+1))
      else
        echo "[INFO] Missing fmap JSON (skip): $jf"
      fi
    done
  done
done

rm -rf "$BIDS_ROOT"/tmp_dcm2bids/*

echo "[INFO] Completed. Subjects: $subject_count, JSONs updated: $update_count"
echo "[INFO] MODE=$MODE (set MODE=first2 to only include first two runs)"
