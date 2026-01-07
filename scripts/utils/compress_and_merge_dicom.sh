#!/usr/bin/env bash
set -u -o pipefail

# Resolve repository root and switch to it (this script lives in scripts/utils)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="data_food_dicom"
FINAL_ZIP="data_food_dicom_2.zip"
DRY_RUN=0
CLEANUP=0
FLATTEN=0
STREAM=0

for arg in "$@"; do
  case "$arg" in
    --dry-run|-n)
      DRY_RUN=1
      ;;
    --cleanup)
      CLEANUP=1
      ;;
    --flatten)
      FLATTEN=1
      ;;
    --stream)
      STREAM=1
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--dry-run] [--cleanup] [--flatten] [--stream]" >&2
      exit 2
      ;;
  esac
done

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' command not found. Please install it and retry." >&2
  exit 1
fi
if [ "$FLATTEN" -eq 1 ] && [ "$DRY_RUN" -ne 1 ]; then
  if ! command -v unzip >/dev/null 2>&1; then
    echo "Error: 'unzip' command not found but --flatten was requested. Please install it and retry." >&2
    exit 1
  fi
fi

# List of target subdirectories in order
TARGETS=(
  "${BASE_DIR}/Suzuki_Food_009A"
  "${BASE_DIR}/Suzuki_Food_009B"
  "${BASE_DIR}/Suzuki_Food_009C"
  "${BASE_DIR}/Suzuki_Food_010A"
  "${BASE_DIR}/Suzuki_Food_010B"
  "${BASE_DIR}/Suzuki_Food_010C"
  "${BASE_DIR}/Suzuki_Food_011A"
  "${BASE_DIR}/Suzuki_Food_011B"
  "${BASE_DIR}/Suzuki_Food_011C"
  "${BASE_DIR}/Suzuki_Food_012A"
  "${BASE_DIR}/Suzuki_Food_012B"
  "${BASE_DIR}/Suzuki_Food_012C"
  "${BASE_DIR}/Suzuki_Food_013A"
  "${BASE_DIR}/Suzuki_Food_013B"
  "${BASE_DIR}/Suzuki_Food_013C"
  "${BASE_DIR}/Suzuki_Food_014A"
  "${BASE_DIR}/Suzuki_Food_014B"
  "${BASE_DIR}/Suzuki_Food_014C"
  "${BASE_DIR}/Suzuki_Food_015A"
  "${BASE_DIR}/Suzuki_Food_015B"
  "${BASE_DIR}/Suzuki_Food_015C"
  "${BASE_DIR}/Suzuki_Food_016A"
  "${BASE_DIR}/Suzuki_Food_016B"
  "${BASE_DIR}/Suzuki_Food_016C"
)

created_zips=()

zip_one_dir() {
  local dir_path="$1"
  local zip_path="${dir_path}.zip"

  if [ ! -d "$dir_path" ]; then
    if [ -f "$zip_path" ]; then
      echo "Skip: directory not found but zip already exists -> $zip_path"
      created_zips+=("$zip_path")
      return 0
    fi
    echo "Warn: directory not found, skipping -> $dir_path" >&2
    return 0
  fi

  echo "Zipping: $dir_path -> $zip_path"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY-RUN: zip -r -9 -q \"$zip_path\" \"$dir_path\""
  else
    # Create zip with maximum compression; -q to reduce noise.
    if ! zip -r -9 -q "$zip_path" "$dir_path"; then
      echo "Error: Failed to zip $dir_path" >&2
      return 1
    fi
  fi

  echo "Deleting original directory: $dir_path"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY-RUN: rm -rf \"$dir_path\""
  else
    rm -rf "$dir_path"
  fi

  created_zips+=("$zip_path")
}

combine_all_zips() {
  local final_zip_path="$FINAL_ZIP"

  # Ensure we have at least one zip to combine
  if [ ${#created_zips[@]} -eq 0 ]; then
    echo "No zip files to combine. Exiting."
    return 0
  fi

  # Remove existing final zip if present to avoid mixing old content
  if [ -f "$final_zip_path" ]; then
    echo "Removing existing final archive: $final_zip_path"
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "DRY-RUN: rm -f \"$final_zip_path\""
    else
      rm -f "$final_zip_path"
    fi
  fi

  if [ "$FLATTEN" -eq 1 ]; then
    echo "Flatten mode: merging contents of ${#created_zips[@]} archives into: $final_zip_path"
    if [ "$DRY_RUN" -eq 1 ]; then
      local tmpdir="/tmp/food-brain-dryrun-flatten"
      echo "DRY-RUN: tmpdir=$tmpdir"
      for z in "${created_zips[@]}"; do
        echo "DRY-RUN: unzip -q \"$z\" -d \"$tmpdir\""
      done
      echo "DRY-RUN: (cd \"$tmpdir\" && zip -r -9 -q \"$final_zip_path\" .)"
      echo "DRY-RUN: rm -rf \"$tmpdir\""
    else
      local tmpdir
      tmpdir=$(mktemp -d)
      # Unzip each zip into temp directory
      for z in "${created_zips[@]}"; do
        if ! unzip -q "$z" -d "$tmpdir"; then
          echo "Error: Failed to unzip $z" >&2
          rm -rf "$tmpdir"
          return 1
        fi
      done
      # Create final zip from merged contents
      (
        cd "$tmpdir" || exit 1
        if ! zip -r -9 -q "$OLDPWD/$final_zip_path" .; then
          echo "Error: Failed to create $final_zip_path from flattened contents" >&2
          rm -rf "$tmpdir"
          return 1
        fi
      )
      rm -rf "$tmpdir"
    fi
  else
    echo "Combining ${#created_zips[@]} archives into: $final_zip_path (zip-of-zips)"
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "DRY-RUN: zip -q -9 \"$final_zip_path\" ${created_zips[*]}"
    else
      # Create a zip-of-zips to avoid re-expanding compressed data
      # Use zip without -r since inputs are files.
      if ! zip -q -9 "$final_zip_path" "${created_zips[@]}"; then
        echo "Error: Failed to create $final_zip_path" >&2
        return 1
      fi
    fi
  fi

  if [ "$CLEANUP" -eq 1 ]; then
    echo "Cleanup enabled: removing intermediate zip files"
    if [ "$DRY_RUN" -eq 1 ]; then
      for z in "${created_zips[@]}"; do echo "DRY-RUN: rm -f \"$z\""; done
    else
      rm -f "${created_zips[@]}"
    fi
  fi
}

# Stream zip-of-zips: process each target, append to final zip, and delete along the way
stream_zip_of_zips() {
  # Start with a fresh final archive
  if [ -f "$FINAL_ZIP" ]; then
    echo "Removing existing final archive before streaming: $FINAL_ZIP"
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "DRY-RUN: rm -f \"$FINAL_ZIP\""
    else
      rm -f "$FINAL_ZIP"
    fi
  fi

  for dir_path in "${TARGETS[@]}"; do
    local zip_path="${dir_path}.zip"

    if [ -d "$dir_path" ]; then
      echo "Streaming: zipping $dir_path -> $zip_path, appending to $FINAL_ZIP, then deleting"
      if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN: zip -r -9 -q \"$zip_path\" \"$dir_path\""
        echo "DRY-RUN: rm -rf \"$dir_path\""
        echo "DRY-RUN: zip -q -9 \"$FINAL_ZIP\" \"$zip_path\""
        if [ "$CLEANUP" -eq 1 ]; then echo "DRY-RUN: rm -f \"$zip_path\""; fi
      else
        if ! zip -r -9 -q "$zip_path" "$dir_path"; then
          echo "Error: Failed to zip $dir_path" >&2
          continue
        fi
        rm -rf "$dir_path"
        if ! zip -q -9 "$FINAL_ZIP" "$zip_path"; then
          echo "Error: Failed to append $zip_path to $FINAL_ZIP" >&2
          # Keep intermediate zip for recovery
          continue
        fi
        # Only remove intermediate zip after a successful append
        if [ "$CLEANUP" -eq 1 ]; then rm -f "$zip_path"; fi
      fi
      continue
    fi

    if [ -f "$zip_path" ]; then
      echo "Streaming: appending existing $zip_path -> $FINAL_ZIP"
      if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN: zip -q -9 \"$FINAL_ZIP\" \"$zip_path\""
        if [ "$CLEANUP" -eq 1 ]; then echo "DRY-RUN: rm -f \"$zip_path\""; fi
      else
        if ! zip -q -9 "$FINAL_ZIP" "$zip_path"; then
          echo "Error: Failed to append $zip_path to $FINAL_ZIP" >&2
          # Keep intermediate zip for recovery
          continue
        fi
        # Only remove intermediate zip after a successful append
        if [ "$CLEANUP" -eq 1 ]; then rm -f "$zip_path"; fi
      fi
      continue
    fi

    echo "Warn: neither directory nor zip found, skipping -> $dir_path" >&2
  done
}

main() {
  echo "Starting compression and merge process in: $(pwd)"
  echo "Base directory: $BASE_DIR"
  echo "Final archive:   $FINAL_ZIP"
  if [ "$DRY_RUN" -eq 1 ]; then echo "Mode: DRY-RUN (no changes will be made)"; fi
  if [ "$CLEANUP" -eq 1 ]; then echo "Mode: CLEANUP (intermediate zips will be removed)"; fi
  if [ "$FLATTEN" -eq 1 ]; then echo "Mode: FLATTEN (final archive will contain actual files/folders)"; fi
  if [ "$STREAM" -eq 1 ]; then echo "Mode: STREAM (zip-of-zips; merge incrementally while deleting)"; fi

  if [ "$STREAM" -eq 1 ]; then
    # Stream zip-of-zips (ignore FLATTEN here as request is for zip-of-zips streaming)
    stream_zip_of_zips
  else
    for target in "${TARGETS[@]}"; do
      zip_one_dir "$target" || {
        echo "Continuing after error with: $target" >&2
      }
    done
    combine_all_zips || exit 1
  fi

  echo "Done."
}

main "$@"
