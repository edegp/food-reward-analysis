#!/usr/bin/env bash
set -euo pipefail

# Clean up common leftover files under fMRIprep (macOS .DS_Store, empty scaffold dirs)
#
# Usage:
#   scripts/utils/cleanup_leftovers.sh [--dry-run] [--target <path>]
#
# Defaults:
#   --target defaults to <repo>/fMRIprep

DRY_RUN=0
TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift;;
    --target) TARGET=${2:-}; shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "$TARGET" ]]; then
  TARGET="$BASE_DIR/fMRIprep"
fi

if [[ ! -d "$TARGET" ]]; then
  echo "ERROR: Target directory not found: $TARGET" >&2
  exit 1
fi

echo "Cleanup target: $TARGET"

# 1) Remove macOS Finder artifacts
found_ds=0
while IFS= read -r -d '' f; do
  found_ds=1
  if (( DRY_RUN )); then
    echo "[dry-run] rm -f '$f'"
  else
    rm -f "$f"
    echo "[removed] $f"
  fi
done < <(find "$TARGET" -type f -name .DS_Store -print0 2>/dev/null || true)

# 2) Remove empty scaffold directories (common in new_scaffold)
#    We keep the top-level TARGET itself; only delete empty children.
while IFS= read -r -d '' d; do
  # Extra safety: avoid following symlinks, remove only real empty dirs
  if [[ -L "$d" ]]; then
    continue
  fi
  if (( DRY_RUN )); then
    echo "[dry-run] rmdir '$d'"
  else
    rmdir "$d" && echo "[removed] $d" || true
  fi
done < <(find "$TARGET" -mindepth 1 -type d -empty -print0 2>/dev/null || true)

echo "Cleanup done."
