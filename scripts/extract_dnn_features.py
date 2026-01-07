#!/usr/bin/env python3
"""
Split raw DNN features stored in .npz archives into per-layer .npy files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Locations relative to the repository root
INPUT_DIR = Path("data_images/dnn_raw_features")
OUTPUT_DIR = Path("results/dnn_analysis/dnn_features_raw")
SUFFIX = "_raw_features"


def extract_layers(npz_path: Path) -> None:
    """Write each layer in a single .npz archive to separate .npy files."""
    source = npz_path.stem
    if source.endswith(SUFFIX):
        source = source[: -len(SUFFIX)]

    with np.load(npz_path, allow_pickle=False) as archive:
        for idx, key in enumerate(archive.files):
            features = archive[key]
            output_path = OUTPUT_DIR / f"{source}_layer_{idx:02d}_features.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, features)
            print(
                f"{npz_path.name}: wrote {output_path} "
                f"(layer key={key}, shape={features.shape}, dtype={features.dtype})"
            )


def main() -> None:
    npz_files = sorted(INPUT_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {INPUT_DIR}")

    for npz_path in npz_files:
        extract_layers(npz_path)


if __name__ == "__main__":
    main()
