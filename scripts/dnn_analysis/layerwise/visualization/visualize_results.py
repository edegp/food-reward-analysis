#!/usr/bin/env python3
"""
Visualize Design I second-level results using glass brain plots.

Design I: Low-Correlation 3-Layer Analysis
- Early, Middle, Late layers with 3 PCs each
- F-contrasts: F: Early Layer, F: Middle Layer, F: Late Layer
- T-contrasts: Layer averages, layer comparisons
"""

import argparse
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn import plotting, image
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
from scipy import stats


def find_latest_results(source_label: str, results_dir: Path) -> Path:
    """Find the latest results directory for a given source."""
    design_dir = results_dir / f"designI_{source_label}"
    if not design_dir.exists():
        raise FileNotFoundError(f"Design I results not found: {design_dir}")

    # Find latest timestamp directory
    dirs = [d for d in design_dir.iterdir() if d.is_dir() and d.name.startswith("2025")]
    if not dirs:
        raise FileNotFoundError(f"No timestamped directories in {design_dir}")

    latest = max(dirs, key=lambda x: x.name)
    return latest


def get_contrast_info():
    """Define contrast information for Design I."""
    # Based on create_second_level_designI.m
    contrasts = {
        # F-contrasts (spmF images)
        1: {"name": "F: Early Layer", "type": "F", "file_prefix": "spmF"},
        2: {"name": "F: Middle Layer", "type": "F", "file_prefix": "spmF"},
        3: {"name": "F: Late Layer", "type": "F", "file_prefix": "spmF"},
        # T-contrasts (spmT images)
        4: {"name": "T: Early average", "type": "T", "file_prefix": "spmT"},
        5: {"name": "T: Middle average", "type": "T", "file_prefix": "spmT"},
        6: {"name": "T: Late average", "type": "T", "file_prefix": "spmT"},
        7: {"name": "T: All average", "type": "T", "file_prefix": "spmT"},
        8: {"name": "T: Early > Middle", "type": "T", "file_prefix": "spmT"},
        9: {"name": "T: Middle > Early", "type": "T", "file_prefix": "spmT"},
        10: {"name": "T: Middle > Late", "type": "T", "file_prefix": "spmT"},
        11: {"name": "T: Late > Middle", "type": "T", "file_prefix": "spmT"},
        12: {"name": "T: Early > Late", "type": "T", "file_prefix": "spmT"},
        13: {"name": "T: Late > Early", "type": "T", "file_prefix": "spmT"},
    }
    return contrasts


def threshold_fdr(stat_img, alpha=0.05, df=30):
    """Apply FDR correction to statistical image."""
    data = stat_img.get_fdata()

    # Get p-values from t-statistics (two-tailed)
    p_vals = 2 * (1 - stats.t.cdf(np.abs(data), df))
    p_vals[data == 0] = 1  # Set zeros to p=1

    # FDR correction (Benjamini-Hochberg)
    p_flat = p_vals.flatten()
    n = len(p_flat)
    sorted_idx = np.argsort(p_flat)
    sorted_p = p_flat[sorted_idx]

    # Find threshold
    threshold_idx = np.where(sorted_p <= alpha * np.arange(1, n + 1) / n)[0]
    if len(threshold_idx) == 0:
        return None, None

    p_threshold = sorted_p[threshold_idx[-1]]

    # Get corresponding t-threshold
    t_threshold = stats.t.ppf(1 - p_threshold / 2, df)

    # Create thresholded image
    thresholded_data = np.where(np.abs(data) >= t_threshold, data, 0)
    thresholded_img = nib.Nifti1Image(thresholded_data, stat_img.affine)

    return thresholded_img, t_threshold


def visualize_glass_brain(results_dir: Path, source_label: str, output_dir: Path,
                          threshold: float = 3.0, correction: str = "none"):
    """Create glass brain visualizations for Design I results."""

    contrasts = get_contrast_info()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer colors for consistency
    layer_colors = {
        "Early": "#3498db",   # Blue
        "Middle": "#e74c3c",  # Red
        "Late": "#2ecc71",    # Green
    }

    print(f"\n=== Visualizing Design I Results: {source_label.upper()} ===")
    print(f"Results directory: {results_dir}")
    print(f"Threshold: {threshold} (correction: {correction})")

    # Create summary figure with all F-contrasts
    fig_f, axes_f = plt.subplots(3, 1, figsize=(12, 12))
    fig_f.suptitle(f"Design I F-contrasts: {source_label.upper()}\n(p < 0.001 uncorrected)",
                   fontsize=14, fontweight='bold')

    # Create summary figure with layer T-contrasts
    fig_t, axes_t = plt.subplots(3, 1, figsize=(12, 12))
    fig_t.suptitle(f"Design I T-contrasts (Layer Averages): {source_label.upper()}\n(p < 0.001 uncorrected)",
                   fontsize=14, fontweight='bold')

    # Process each contrast
    for con_idx, con_info in contrasts.items():
        stat_file = results_dir / f"{con_info['file_prefix']}_{con_idx:04d}.nii"

        if not stat_file.exists():
            print(f"  Warning: {stat_file.name} not found")
            continue

        stat_img = nib.load(str(stat_file))

        # Determine threshold based on contrast type
        if con_info["type"] == "F":
            # F-statistics: use uncorrected p < 0.001 (F > ~7 for df1=3, df2=30)
            display_threshold = 7.0
        else:
            # T-statistics
            display_threshold = threshold

        print(f"  Processing: {con_info['name']}")

        # Individual glass brain
        fig_single = plt.figure(figsize=(12, 4))

        try:
            display = plotting.plot_glass_brain(
                stat_img,
                threshold=display_threshold,
                display_mode='lyrz',
                colorbar=True,
                black_bg=False,
                title=f"{con_info['name']} ({source_label.upper()})",
                figure=fig_single
            )

            # Save individual figure
            safe_name = con_info['name'].replace(' ', '_').replace(':', '').replace('>', 'gt')
            fig_single.savefig(output_dir / f"glass_brain_{source_label}_{safe_name}.png",
                              dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_single)

        except Exception as e:
            print(f"    Warning: Could not plot {con_info['name']}: {e}")
            plt.close(fig_single)
            continue

        # Add to summary figures
        if con_info["type"] == "F" and con_idx <= 3:
            ax_idx = con_idx - 1
            layer_name = ["Early", "Middle", "Late"][ax_idx]

            plotting.plot_glass_brain(
                stat_img,
                threshold=display_threshold,
                display_mode='lyrz',
                axes=axes_f[ax_idx],
                colorbar=True,
                black_bg=False,
                title=f"F: {layer_name} Layer"
            )

        elif con_info["type"] == "T" and con_idx in [4, 5, 6]:
            ax_idx = con_idx - 4
            layer_name = ["Early", "Middle", "Late"][ax_idx]

            plotting.plot_glass_brain(
                stat_img,
                threshold=display_threshold,
                display_mode='lyrz',
                axes=axes_t[ax_idx],
                colorbar=True,
                black_bg=False,
                title=f"T: {layer_name} average"
            )

    # Save summary figures
    fig_f.tight_layout()
    fig_f.savefig(output_dir / f"glass_brain_{source_label}_F_contrasts_summary.png",
                  dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_f)

    fig_t.tight_layout()
    fig_t.savefig(output_dir / f"glass_brain_{source_label}_T_layer_summary.png",
                  dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_t)

    # Create layer comparison figure
    fig_comp, axes_comp = plt.subplots(2, 3, figsize=(15, 8))
    fig_comp.suptitle(f"Design I Layer Comparisons: {source_label.upper()}\n(threshold = {threshold})",
                      fontsize=14, fontweight='bold')

    comparison_contrasts = [
        (8, "Early > Middle"),
        (9, "Middle > Early"),
        (10, "Middle > Late"),
        (11, "Late > Middle"),
        (12, "Early > Late"),
        (13, "Late > Early"),
    ]

    for idx, (con_idx, title) in enumerate(comparison_contrasts):
        row = idx // 3
        col = idx % 3

        stat_file = results_dir / f"spmT_{con_idx:04d}.nii"
        if stat_file.exists():
            stat_img = nib.load(str(stat_file))
            try:
                plotting.plot_glass_brain(
                    stat_img,
                    threshold=threshold,
                    display_mode='lyrz',
                    axes=axes_comp[row, col],
                    colorbar=True,
                    black_bg=False,
                    title=title
                )
            except:
                axes_comp[row, col].set_title(f"{title}\n(no suprathreshold voxels)")
                axes_comp[row, col].axis('off')

    fig_comp.tight_layout()
    fig_comp.savefig(output_dir / f"glass_brain_{source_label}_layer_comparisons.png",
                     dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_comp)

    print(f"\nFigures saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Visualize Design I second-level results")
    parser.add_argument("--source", type=str, choices=["clip", "convnext", "both"],
                        default="both", help="Source model to visualize")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="T-statistic threshold for display")
    parser.add_argument("--correction", type=str, choices=["none", "fdr"],
                        default="none", help="Multiple comparison correction")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures")

    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent
    results_base = project_root / "results" / "dnn_analysis" / "second_level"

    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = project_root / "results" / "dnn_analysis" / "figures" / "designI"

    sources = ["clip", "convnext"] if args.source == "both" else [args.source]

    for source in sources:
        try:
            results_dir = find_latest_results(source, results_base)
            output_dir = output_base / source

            visualize_glass_brain(
                results_dir=results_dir,
                source_label=source,
                output_dir=output_dir,
                threshold=args.threshold,
                correction=args.correction
            )

        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
