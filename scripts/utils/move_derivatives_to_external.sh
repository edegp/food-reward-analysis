#!/usr/bin/env bash
set -euo pipefail

# Move or copy the entire fMRIprep/derivatives directory to an external drive
# and replace the original with a symlink. Supports restore as well.
#
# Usage:
#   scripts/utils/move_derivatives_to_external.sh [--dry-run] [--copy] [--external-root <path>] [--restore]
#
# Options:
#   --dry-run           Print planned actions only
#   --copy              Copy instead of move (then replace original with symlink)
#   --external-root     External drive mount root (default: /Volumes/Transcend)
#   --restore, -r       Restore: replace symlinked derivatives with real directory from external drive

DRY_RUN=0
USE_COPY=0
EXT_ROOT="/Volumes/Transcend"
RESTORE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift;;
    --copy) USE_COPY=1; shift;;
    --external-root) EXT_ROOT=${2:-}; shift 2;;
    --restore|-r) RESTORE=1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$BASE_DIR/fMRIprep/derivatives"
DEST_DIR="$EXT_ROOT/hit/food-brain/fMRIprep/derivatives"

echo "External root: $EXT_ROOT"
echo "Source derivatives: $SRC_DIR"
echo "Destination derivatives: $DEST_DIR"

# Preflight: ensure external root exists and is writable
if [[ ! -d "$EXT_ROOT" ]]; then
  echo "ERROR: External root does not exist: $EXT_ROOT" >&2
  exit 1
fi

if (( ! DRY_RUN )); then
  test_dir="$EXT_ROOT/.write_test_$$"
  if ! mkdir -p "$test_dir" 2>/dev/null; then
    echo "ERROR: Cannot write to $EXT_ROOT (read-only or permission denied)." >&2
    echo "Hint: On macOS, NTFS volumes mount read-only by default. Consider using an APFS/HFS+/exFAT volume, or set EXTERNAL_ROOT to a writable path." >&2
    exit 1
  else
    rmdir "$test_dir" 2>/dev/null || true
  fi
fi

if [[ ! -d "$(dirname "$DEST_DIR")" ]]; then
  if (( DRY_RUN )); then
    echo "[dry-run] mkdir -p '$(dirname "$DEST_DIR")'"
  else
    mkdir -p "$(dirname "$DEST_DIR")"
  fi
fi

if (( RESTORE )); then
  # Restore derivatives from external drive back to workspace
  if [[ -L "$SRC_DIR" ]]; then
    # Remove symlink and copy/move back
    if (( DRY_RUN )); then
      echo "[dry-run] rm -f '$SRC_DIR'"
      echo "[dry-run] mkdir -p '$(dirname "$SRC_DIR")'"
      echo "[dry-run] rsync -a --info=progress2 '$DEST_DIR/' '$SRC_DIR/'"
    else
      rm -f "$SRC_DIR"
      mkdir -p "$(dirname "$SRC_DIR")"
      mkdir -p "$SRC_DIR"
      rsync -a "$DEST_DIR/" "$SRC_DIR/"
    fi
    echo "[restored] $SRC_DIR from $DEST_DIR"
  else
    echo "[skip] Source is not a symlink: $SRC_DIR (already restored?)" >&2
  fi
  exit 0
fi

# Forward: move/copy to external and symlink back
if [[ ! -d "$SRC_DIR" ]]; then
  echo "ERROR: $SRC_DIR does not exist or is not a directory" >&2
  exit 1
fi

if (( DRY_RUN )); then
  echo "[dry-run] mkdir -p '$(dirname "$DEST_DIR")'"
  if (( USE_COPY )); then
    echo "[dry-run] mkdir -p '$DEST_DIR'"
    echo "[dry-run] rsync -a --info=progress2 '$SRC_DIR/' '$DEST_DIR/'"
    echo "[dry-run] rm -rf '$SRC_DIR'"
  else
    echo "[dry-run] mv '$SRC_DIR' '$DEST_DIR'  # (parent ensured)"
  fi
  echo "[dry-run] ln -sfn '$DEST_DIR' '$SRC_DIR'"
  exit 0
fi

# Ensure parent exists
mkdir -p "$(dirname "$DEST_DIR")"
if (( USE_COPY )); then
  mkdir -p "$DEST_DIR"
  rsync -a "$SRC_DIR/" "$DEST_DIR/"
  rm -rf "$SRC_DIR"
else
  # Move: if DEST_DIR exists and is empty, remove it first to allow rename
  if [[ -d "$DEST_DIR" && -z "$(ls -A "$DEST_DIR" 2>/dev/null || true)" ]]; then
    rmdir "$DEST_DIR" || true
  fi
  if [[ ! -e "$DEST_DIR" ]]; then
    mv "$SRC_DIR" "$DEST_DIR"
  else
    # Destination exists and has contents: fall back to rsync then remove
    rsync -a "$SRC_DIR/" "$DEST_DIR/"
    rm -rf "$SRC_DIR"
  fi
fi
ln -sfn "$DEST_DIR" "$SRC_DIR"
echo "[linked] $SRC_DIR -> $DEST_DIR"
