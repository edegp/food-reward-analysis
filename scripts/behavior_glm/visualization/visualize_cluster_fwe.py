"""Visualize Cluster-level FWE results for Behavior GLM

This script visualizes pre-computed cluster FWE results from SPM.
The cluster FWE analysis must be run first using run_cluster_fwe.m
which uses SPM's proper RFT calculation.

Usage:
    # First run MATLAB to compute cluster FWE:
    # matlab -batch "addpath('scripts/behavior_glm/visualization'); run_cluster_fwe('glm_001p_6')"

    # Then visualize:
    python visualize_cluster_fwe.py --model glm_001p_6
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from nilearn import plotting
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Cluster-level FWE results for Behavior GLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="glm_001p_6",
        help="Model name (default: glm_001p_6)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures",
    )
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent.parent.parent
    cluster_fwe_dir = project_dir / "results" / "behavior_glm" / "cluster_fwe" / args.model

    if not cluster_fwe_dir.exists():
        print(f"Cluster FWE results not found: {cluster_fwe_dir}")
        print("\nPlease run the MATLAB script first:")
        print(f"  matlab -batch \"addpath('scripts/behavior_glm/visualization'); run_cluster_fwe('{args.model}')\"")
        return

    print(f"Using Cluster FWE directory: {cluster_fwe_dir}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_dir / "results" / "behavior_glm" / "visualization" / "cluster_fwe"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Read summary CSV
    csv_file = cluster_fwe_dir / "cluster_fwe_results.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"\nLoaded {len(df)} contrasts from CSV")
    else:
        df = None
        print("No CSV summary found")

    # Find NIfTI files
    nii_files = list(cluster_fwe_dir.glob("*_clusterFWE.nii"))
    print(f"Found {len(nii_files)} cluster FWE NIfTI files")

    if not nii_files:
        print("No cluster FWE NIfTI files found!")
        print("Please run the MATLAB script first to compute cluster FWE results.")
        return

    # Collect results with data
    results = []
    for nii_file in nii_files:
        name = nii_file.stem.replace("_clusterFWE", "")
        try:
            img = nib.load(str(nii_file))
            data = img.get_fdata()
            n_voxels = int(np.sum(data != 0))

            if n_voxels > 0:
                peak_val = np.max(data)

                # Count clusters
                binary_mask = data > 0
                _, n_clusters = ndimage.label(binary_mask)

                # Get k_threshold from CSV if available
                k_threshold = "?"
                if df is not None:
                    row = df[df['name'] == name]
                    if not row.empty:
                        k_threshold = int(row.iloc[0]['k_threshold'])

                display_name = name.replace("_", " × ").replace("x", " × ")
                results.append({
                    "name": name,
                    "display_name": display_name,
                    "img": img,
                    "n_voxels": n_voxels,
                    "n_clusters": n_clusters,
                    "peak_val": peak_val,
                    "k_threshold": k_threshold,
                })
                print(f"{display_name}: {n_clusters} clusters, {n_voxels:,} voxels, peak T={peak_val:.2f}, k={k_threshold}")

        except Exception as e:
            print(f"Error loading {nii_file}: {e}")

    if not results:
        print("\nNo significant clusters to visualize!")
        return

    print(f"\nVisualizing {len(results)} contrasts with significant clusters")

    # Create figure (single column, no text panel)
    n_results = len(results)
    fig = plt.figure(figsize=(14, 3.5 * n_results), facecolor='white')

    for row_idx, result in enumerate(results):
        ax_brain = fig.add_subplot(n_results, 1, row_idx + 1)
        ax_brain.set_facecolor('white')

        # Get threshold for vmin (colorbar starts from threshold)
        data = result["img"].get_fdata()
        nonzero_vals = data[data > 0]
        if len(nonzero_vals) > 0:
            threshold = np.min(nonzero_vals)
        else:
            threshold = 0

        display = plotting.plot_glass_brain(
            result["img"],
            threshold=threshold,
            display_mode='lzry',
            colorbar=True,
            plot_abs=False,
            axes=ax_brain,
            cmap='hot',
            vmin=threshold,
            vmax=result["peak_val"],
            title=None,
            black_bg=False
        )

        # Increase colorbar font size
        if hasattr(display, '_cbar') and display._cbar is not None:
            display._cbar.ax.tick_params(labelsize=14)

        # Add ROI contours
        roi_dir = project_dir / "rois" / "AAL2"
        roi_masks = [
            (roi_dir / "vmPFC_mask.nii", "blue", "vmPFC"),
            (roi_dir / "OFC_medial_bi_mask.nii", "green", "OFC"),
            (roi_dir / "Striatum_L_mask.nii", "cyan", "L-Str"),
            (roi_dir / "Striatum_R_mask.nii", "magenta", "R-Str"),
        ]

        legend_handles = []
        legend_labels = []
        from matplotlib.patches import Rectangle

        for mask_file, color, label in roi_masks:
            if mask_file.exists():
                display.add_contours(
                    str(mask_file),
                    antialiased=False,
                    linewidths=1.5,
                    levels=[0.5],
                    colors=[color],
                )
                legend_handles.append(Rectangle((0, 0), 1, 1, fc=color))
                legend_labels.append(label)

        # Add legend (positioned higher and more to the left)
        ax_brain.legend(legend_handles, legend_labels, loc='upper left',
                       fontsize=10, framealpha=0.8, bbox_to_anchor=(-0.05, 1.12))

    # タイトルは論文のキャプションに記載するため削除
    # plt.suptitle(f'{args.model}: Cluster-level FWE (p < 0.05)\nCluster-forming threshold: p < 0.001 uncorrected',
    #             fontsize=14, fontweight='bold')

    # Save figure
    output_filename = f'{args.model}_cluster_fwe.png'
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"Cluster-level FWE Results Summary (proper RFT)")
    print(f"Model: {args.model}")
    print("=" * 70)
    print(f"{'Contrast':<25} {'k_thresh':>10} {'Clusters':>10} {'Voxels':>10} {'Peak T':>10}")
    print("-" * 70)
    for result in results:
        print(f"{result['display_name']:<25} {result['k_threshold']:>10} {result['n_clusters']:>10} {result['n_voxels']:>10,} {result['peak_val']:>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
