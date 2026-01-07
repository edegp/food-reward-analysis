"""Visualize Design N v3 second-level results (Layer+Shared F-contrasts)
Following the style of visualize_layerwise_results.py
"""
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from nilearn import plotting
from nilearn.glm import threshold_stats_img
import nibabel as nib
import numpy as np

# Set font for Japanese text
plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'sans-serif']


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Design N v3 second-level results"
    )
    parser.add_argument(
        "--spm-dir",
        type=str,
        help="Path to SPM output directory. If not specified, uses latest.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="convnext",
        choices=["convnext", "clip"],
        help="Source label",
    )
    parser.add_argument(
        "--correction",
        type=str,
        default="fwe",
        choices=["fwe", "bonferroni", "fdr", "fpr"],
        help="Correction method (fwe/bonferroni=FWE)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="P-value threshold"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--include-shared",
        action="store_true",
        help="Include Shared PCs in visualization (use _withShared contrasts)",
    )
    parser.add_argument(
        "--include-global",
        action="store_true",
        help="Include Global and Shared F-contrasts",
    )
    args = parser.parse_args()

    # Find SPM directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent.parent.parent.parent
    results_base = project_dir / "results" / "dnn_analysis" / "second_level"

    if args.spm_dir:
        spm_dir = Path(args.spm_dir)
    else:
        # Find latest v3 run
        v3_dir = results_base / f"hierarchical_{args.source}_v3"
        if not v3_dir.exists():
            raise FileNotFoundError(f"V3 results directory not found: {v3_dir}")

        runs = sorted([d for d in v3_dir.iterdir() if d.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No runs found in: {v3_dir}")
        spm_dir = runs[-1]

    print(f"Using SPM directory: {spm_dir}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_dir / "results" / "dnn_analysis" / "visualization" / "hierarchical"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Define F-contrast groups in order
    # Choose between 'only' (Layer-Specific only) and 'withShared' (Layer + Shared)
    if args.include_shared and args.include_global:
        # withShared plus Global and Shared F-tests
        ordered_groups = [
            ("Global_F", "Global"),
            ("Shared_F", "Shared"),
            ("Initial_withShared", "初期層+Shared"),
            ("Middle_withShared", "中間層+Shared"),
            ("Late_withShared", "後期層+Shared"),
            ("Final_withShared", "最終層+Shared"),
        ]
    elif args.include_shared:
        # Only 4 withShared contrasts (no Global/Shared)
        ordered_groups = [
            ("Initial_withShared", "初期層"),
            ("Middle_withShared", "中間層"),
            ("Late_withShared", "後期層"),
            ("Final_withShared", "最終層"),
        ]
    elif args.include_global:
        ordered_groups = [
            ("Global_F", "Global"),
            ("Shared_F", "Shared"),
            ("Initial_only", "初期層"),
            ("Middle_only", "中間層"),
            ("Late_only", "後期層"),
            ("Final_only", "最終層"),
        ]
    else:
        ordered_groups = [
            ("Initial_only", "初期層"),
            ("Middle_only", "中間層"),
            ("Late_only", "後期層"),
            ("Final_only", "最終層"),
        ]

    # First pass: determine which groups have significant voxels
    groups_with_results = []

    for fc_dir_name, display_name in ordered_groups:
        fc_dir = spm_dir / fc_dir_name
        stat_file = fc_dir / "spmF_0001.nii"

        if not stat_file.exists():
            print(f"Warning: {stat_file} not found")
            continue

        try:
            stat_img = nib.load(str(stat_file))

            # Threshold the image
            # Map fwe to bonferroni for nilearn
            height_control = 'bonferroni' if args.correction == 'fwe' else args.correction
            thresholded_img, threshold_value = threshold_stats_img(
                stat_img,
                alpha=args.threshold,
                height_control=height_control,
                two_sided=False,
            )

            data = thresholded_img.get_fdata()
            n_significant = int(np.sum(data != 0))

            if n_significant > 0:
                peak_value = np.max(data)
                groups_with_results.append({
                    "name": display_name,
                    "dir_name": fc_dir_name,
                    "stat_file": stat_file,
                    "thresholded_img": thresholded_img,
                    "threshold": threshold_value,
                    "n_voxels": n_significant,
                    "peak_F": peak_value,
                })
                print(f"{display_name}: {n_significant:,} voxels, peak F={peak_value:.2f}, threshold={threshold_value:.2f}")
            else:
                print(f"{display_name}: No significant voxels at {args.correction.upper()} p < {args.threshold}")

        except Exception as e:
            print(f"Error processing {display_name}: {e}")
            continue

    if not groups_with_results:
        print("No significant results to display")
        return

    # Create figure following layerwise style
    n_groups = len(groups_with_results)
    fig = plt.figure(figsize=(14, 3 * n_groups), facecolor='white')

    # Use GridSpec for flexible layout (same as cluster FWE script)
    gs = GridSpec(n_groups, 2, figure=fig, width_ratios=[6, 1],
                  hspace=0.3, wspace=0.05)

    # Create integrated glass brain view for each row
    for row_idx, group in enumerate(groups_with_results):
        ax_brain = fig.add_subplot(gs[row_idx, 0])
        ax_brain.set_facecolor('white')

        display = plotting.plot_glass_brain(
            group["thresholded_img"],
            threshold=group["threshold"],
            display_mode='lzry',  # Left, Z (axial), Right, Y (coronal)
            colorbar=True,
            plot_abs=False,
            axes=ax_brain,
            cmap='hot',
            vmin=group["threshold"],
            vmax=group["peak_F"],
            title=None,
            black_bg=False
        )

        # Increase colorbar font size
        if hasattr(display, '_cbar') and display._cbar is not None:
            display._cbar.ax.tick_params(labelsize=14)

        # Layer name only (same style as cluster FWE script)
        ax_text = fig.add_subplot(gs[row_idx, 1])
        ax_text.axis('off')
        ax_text.set_facecolor('white')

        ax_text.text(0.1, 0.5, group['name'],
                    fontsize=21,
                    fontweight='bold',
                    verticalalignment='center')

    plt.suptitle(f'{args.source.upper()}',
                fontsize=21, fontweight='bold')

    # Save figure
    output_filename = f'{args.source}_hierarchical_{args.correction}_p{args.threshold}.png'
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"Results Summary ({args.correction.upper()} p < {args.threshold})")
    print("=" * 70)
    print(f"{'Contrast':<25} {'Sig Voxels':>15} {'Max F':>10} {'Threshold':>10}")
    print("-" * 70)
    for g in groups_with_results:
        print(f"{g['name']:<25} {g['n_voxels']:>15,} {g['peak_F']:>10.2f} {g['threshold']:>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
