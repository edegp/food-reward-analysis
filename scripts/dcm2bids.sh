#!/usr/bin/env bash
set -ex

dicom_dir="/dicoms"
participants_filter=()
participants_set=false
nofile_limit="${NOFILE_LIMIT:-524288}"

if ! ulimit -n "$nofile_limit" 2>/dev/null; then
  echo "Warning: Unable to raise NOFILE_LIMIT to ${nofile_limit}. Continuing with current limit."
fi

tmp_root="$(mktemp -d /tmp/dcm2bids_out.XXXXXX)"
cleanup_tmp() {
  rm -rf "${tmp_root}"
}
trap cleanup_tmp EXIT

if [[ -n "${PARTICIPANTS:-}" ]]; then
  # shellcheck disable=SC2206 # Intentional word splitting to build array
  participants_filter=(${PARTICIPANTS})
  participants_set=true
  echo "Restricting processing to participants: ${participants_filter[*]}"
fi

if [ ! -d "$dicom_dir" ]; then
  echo "Error: DICOM directory '$dicom_dir' does not exist."
  exit 1
fi

echo "Starting DICOM to BIDS conversion..."

# Determine copy command based on rsync availability
if command -v rsync >/dev/null 2>&1; then
  copy_cmd=(rsync -a --exclude tmp_dcm2bids)
else
  echo "Warning: rsync not found, falling back to cp -a (may be slower)."
  copy_cmd=(cp -a)
fi

for dir in "$dicom_dir"/*/*/; do
  echo "Found directory: $dir"
  if [ -d "$dir" ]; then
    # Compute relative path under $dicom_dir (preserves spaces)
    rel_path="${dir#${dicom_dir}/}"
    rel_path="${rel_path%/}"
    echo "Relative path: $rel_path"

    # parent (e.g. Suzuki_Food_007C) and child (e.g. Hias_Suzuki - 4eb...)
    parent_dir="${rel_path%%/*}"
    child_dir="${rel_path#*/}"
    echo "Parent: $parent_dir"
    echo "Child: $child_dir"

    # participant number: last underscore block of parent_dir, first 3 chars
    parent_parts=(${parent_dir//_/ })
    last_block="${parent_parts[-1]}"
    p_num="${last_block:1:2}"
    echo "Participant number: $p_num"

    # session letter is the final char of last_block (A/B/C)
    last_char="${last_block: -1}"
    case "$last_char" in
      A) s_num="01" ;;
      B) s_num="02" ;;
      C) s_num="03" ;;
      *)
        echo "Warning: Unexpected session code in '$last_block'. Skipping."
        continue
        ;;
    esac

    if $participants_set; then
      match=false
      for target in "${participants_filter[@]}"; do
        if [[ "$p_num" == "$target" ]]; then
          match=true
          break
        fi
      done
      if ! $match; then
        echo "Skipping participant $p_num (not in PARTICIPANTS filter)."
        continue
      fi
    fi

    echo "Processing directory: $rel_path with Participant: $p_num, Session: $s_num"
    run_dir="$(mktemp -d "${tmp_root}/run.XXXXXX")"
    dcm2bids -d "${dicom_dir}/${rel_path}" -p "$p_num" -s "$s_num" -c /config/config.json -o "${run_dir}"
    if [[ "${copy_cmd[0]}" == "rsync" ]]; then
      "${copy_cmd[@]}" "${run_dir}/" /bids/
    else
      cp -a "${run_dir}/." /bids/
    fi
    rm -rf "${run_dir}"
  fi
done

chmod u+x /app/update_intended_for.sh

/app/update_intended_for.sh /bids

echo "Script finished."
