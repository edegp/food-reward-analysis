#!/usr/bin/env bash
set -euo pipefail

# Mirror fMRIPrep BOLD files (derivatives and raw BIDS) into Google Drive and replace originals with symlinks.
# Applies to:
#  - fMRIprep derivatives: fMRIprep/derivatives/sub-*/ses-*/func/*_space-*_desc-preproc_bold.nii.gz
#  - BIDS bold files:      fMRIprep/bids/sub-*/ses-*/func/*_bold.nii[.gz]
#  - Smoothed outputs:     fMRIprep/smoothed/sub-*/ses-*/func/*.nii and fMRIprep/smoothed/sub-*.zip (restore only)
#
# Usage:
#   scripts/utils/symlink_to_gdrive.sh [--dry-run] [--copy] [--drive-root <path>] [--subject <id>] [--restore] [--trash-root <path>]
#
# Options:
#   --dry-run        Print planned actions without modifying files
#   --copy           Copy instead of move (still replaces original with a symlink to the copy)
#   --drive-root     Override detected Google Drive root directory
#   --subject, -s    Target specific subject (e.g., 011A or sub-011A). If omitted, process all subjects
#   --restore, -r    Restore: replace symlinked files with real files from Google Drive (reverse operation)
#   --trash-root     Override Google Drive .Trash directory (auto-detected by default)
#
# Notes:
# - Default DRIVE roots tried (first existing is used):
#   ~/Library/CloudStorage/GoogleDrive-My\ Drive
#   ~/Library/CloudStorage/GoogleDrive-マイドライブ
#   ~/Library/CloudStorage/GoogleDrive-*
# - Destination base becomes: $DRIVE_ROOT/hit/food-brain

DRY_RUN=0
USE_COPY=0
DRIVE_ROOT_OVERRIDE=""
RESTORE=0
SUBJECT=""
TRASH_ROOT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1; shift;;
    --copy)
      USE_COPY=1; shift;;
    --drive-root)
      DRIVE_ROOT_OVERRIDE=${2:-}; shift 2;;
    --trash-root)
      TRASH_ROOT_OVERRIDE=${2:-}; shift 2;;
    --subject|-s)
      SUBJECT=${2:-}; shift 2;;
    --restore|-r)
      RESTORE=1; shift;;
    *)
      echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DERIV="$BASE_DIR/fMRIprep/derivatives"
SRC_BIDS="$BASE_DIR/fMRIprep/bids"
SRC_SMOOTHED="$BASE_DIR/fMRIprep/smoothed"

# Detect Google Drive root
if [[ -n "$DRIVE_ROOT_OVERRIDE" ]]; then
  DRIVE_ROOT="$DRIVE_ROOT_OVERRIDE"
else
  candidates=(
    "$HOME/Library/CloudStorage/GoogleDrive-dm240001@g.hit-u.ac.jp/マイドライブ"
  )
  # Include any GoogleDrive-* as fallback
  while IFS= read -r -d '' p; do candidates+=("$p"); done < <(find "$HOME/Library/CloudStorage" -maxdepth 1 -type d -name 'GoogleDrive-*' -print0 2>/dev/null || true)
  DRIVE_ROOT=""
  for c in "${candidates[@]}"; do
    if [[ -d "$c" ]]; then DRIVE_ROOT="$c"; break; fi
  done
fi

if [[ -z "${DRIVE_ROOT:-}" || ! -d "$DRIVE_ROOT" ]]; then
  echo "ERROR: Google Drive root not found. Use --drive-root to specify it." >&2
  exit 1
fi

DEST_PROJECT="$DRIVE_ROOT/hit/food-brain"

# Detect Google Drive .Trash directory
if [[ -n "$TRASH_ROOT_OVERRIDE" ]]; then
  TRASH_ROOT="$TRASH_ROOT_OVERRIDE"
else
  # Typically sibling of マイドライブ within GoogleDrive-<account>
  TRASH_ROOT="$(dirname "$DRIVE_ROOT")/.Trash"
fi

# Iterate over all target BOLDs (portable for macOS bash)
echo "Using Google Drive root: $DRIVE_ROOT"
echo "Project destination: $DEST_PROJECT"
if [[ -d "$TRASH_ROOT" ]]; then
  echo "Drive Trash: $TRASH_ROOT"
fi

processed=0
found_any=0

# Subject filter glob
SUBJECT_GLOB="sub-*"
if [[ -n "$SUBJECT" ]]; then
  if [[ "$SUBJECT" == sub-* ]]; then
    SUBJECT_GLOB="$SUBJECT"
  else
    SUBJECT_GLOB="sub-$SUBJECT"
  fi
  echo "Subject filter: $SUBJECT_GLOB"
fi

# Combine find results from derivatives and bids (both produce NIfTI BOLD files)
while IFS= read -r -d '' f; do
  found_any=1
  rel="${f#${BASE_DIR}/}"
  dest="$DEST_PROJECT/$rel"
  dest_dir="$(dirname "$dest")"

  if (( RESTORE )); then
    # Restore mode: replace symlink with the real file from Drive
    if [[ -L "$f" ]]; then
      target="$(readlink "$f" || true)"
      src_path=""
      # Prefer the canonical Drive destination if it exists
      if [[ -e "$dest" ]]; then
        src_path="$dest"
      else
        # If the current symlink target exists, use it directly
        if [[ -n "$target" && -e "$target" ]]; then
          src_path="$target"
        else
          # Try to locate the file in Google Drive .Trash
          # Prefer exact subpath match under hit/food-brain/$rel if present; else fall back to basename search
          if [[ -d "$TRASH_ROOT" ]]; then
            candidate_exact=""
            # Look for exact tail match of hit/food-brain/$rel
            while IFS= read -r -d '' p; do
              if [[ "$p" == *"/hit/food-brain/$rel" ]]; then
                candidate_exact="$p"; break
              fi
            done < <(find "$TRASH_ROOT" -type f -name "$(basename "$f")" -print0 2>/dev/null || true)

            if [[ -n "$candidate_exact" ]]; then
              src_path="$candidate_exact"
            else
              # Fallback: first file with same name that contains /hit/food-brain/ and subject token if provided
              subject_token="$SUBJECT_GLOB"
              while IFS= read -r -d '' p; do
                if [[ "$p" == *"/hit/food-brain/"* ]]; then
                  if [[ -z "$SUBJECT" || "$p" == *"/$subject_token/"* ]]; then
                    src_path="$p"; break
                  fi
                fi
              done < <(find "$TRASH_ROOT" -type f -name "$(basename "$f")" -print0 2>/dev/null || true)
            fi
          fi
        fi
      fi

      if [[ -z "$src_path" ]]; then
        echo "[skip] Source not found in Drive or Trash: $dest" >&2
        continue
      fi

      if (( DRY_RUN )); then
        echo "[dry-run] rm -f '$f' && cp -f '$src_path' '$f'"
        processed=$((processed+1))
        continue
      fi

      rm -f "$f"
      # Always copy from Trash/Drive back to local (non-destructive to Drive)
      cp -f "$src_path" "$f"
      echo "[restored] $f from $src_path"
      processed=$((processed+1))
    else
      echo "[skip] Not a symlink (already restored?): $f" >&2
    fi
  else
    # Forward mode: move/copy to Drive and symlink back
    if (( DRY_RUN )); then
      echo "[dry-run] mkdir -p '$dest_dir'"
      if (( USE_COPY )); then
        echo "[dry-run] cp -f '$f' '$dest' && rm -f '$f'"
      else
        echo "[dry-run] mv '$f' '$dest'"
      fi
      echo "[dry-run] ln -sfn '$dest' '$f'"
      processed=$((processed+1))
      continue
    fi

    mkdir -p "$dest_dir"
    if (( USE_COPY )); then
      cp -f "$f" "$dest"
      rm -f "$f"
    else
      mv "$f" "$dest"
    fi
    ln -sfn "$dest" "$f"
    echo "[linked] $f -> $dest"
    processed=$((processed+1))
  fi
done < <(
  (
    if (( RESTORE )); then
      find "$SRC_DERIV" -type l -path "*/$SUBJECT_GLOB/ses-*/*" -print0 2>/dev/null || true
    else
      find "$SRC_DERIV" -type f -path "*/$SUBJECT_GLOB/ses-*/*" -print0 2>/dev/null || true
    fi
  )
  (
    if (( RESTORE )); then
      find "$SRC_BIDS" -type l -path "*/$SUBJECT_GLOB/ses-*/*" -print0 2>/dev/null || true
    else
      find "$SRC_BIDS" -type f -path "*/$SUBJECT_GLOB/ses-*/*" -print0 2>/dev/null || true
    fi
  )
  (
    # Restore support for smoothed outputs. Only act on symlinks.
    if (( RESTORE )); then
      # Subject/session NIfTI files under smoothed
      if [[ -d "$SRC_SMOOTHED" ]]; then
        find "$SRC_SMOOTHED" -type l -path "*/$SUBJECT_GLOB/ses-*/*" -print0 2>/dev/null || true
        # Also include top-level sub-*.zip under smoothed
        find "$SRC_SMOOTHED" -type l -name "$SUBJECT_GLOB.zip" -print0 2>/dev/null || true
      fi
    fi
  )
)

if (( ! found_any )); then
  echo "No matching BOLD files found under $SRC_DERIV or $SRC_BIDS" >&2
else
  if (( RESTORE )); then
    echo "Done (restore). Processed $processed files."
  else
    echo "Done. Processed $processed files."
  fi
fi
