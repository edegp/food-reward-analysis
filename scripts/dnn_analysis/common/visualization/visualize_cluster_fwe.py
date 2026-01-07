"""Visualize Cluster-level FWE results for Design N v3 and Design O"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from nilearn import plotting
import nibabel as nib
import numpy as np
import pandas as pd

# Set font for Japanese text
plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'sans-serif']


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Cluster-level FWE results"
    )
    parser.add_argument(
        "--design",
        type=str,
        required=True,
        choices=["hierarchical", "designO"],
        help="Design to visualize",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="clip",
        choices=["convnext", "clip"],
        help="Source label",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures",
    )
    args = parser.parse_args()

    # Find cluster FWE directory (use v2 if available)
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent.parent.parent.parent

    # Try multiple naming conventions for v2 results
    v2_base = project_dir / "results" / "dnn_analysis" / "cluster_fwe_v2"
    v1_base = project_dir / "results" / "dnn_analysis" / "cluster_fwe"

    # Possible directory names (try v3 suffix first, then without)
    possible_names = [
        f"{args.design}_v3_{args.source}",  # e.g., hierarchical_v3_clip
        f"{args.design}_{args.source}",      # e.g., hierarchical_clip
    ]

    cluster_fwe_dir = None
    # Try v2 directories first
    for name in possible_names:
        candidate = v2_base / name
        if candidate.exists():
            cluster_fwe_dir = candidate
            print("Using v2 results (proper RFT calculation)")
            break

    # Fall back to v1 directories
    if cluster_fwe_dir is None:
        for name in possible_names:
            candidate = v1_base / name
            if candidate.exists():
                cluster_fwe_dir = candidate
                break

    if cluster_fwe_dir is None or not cluster_fwe_dir.exists():
        raise FileNotFoundError(f"Cluster FWE directory not found. Tried: {possible_names}")

    print(f"Using Cluster FWE directory: {cluster_fwe_dir}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_dir / "results" / "dnn_analysis" / "visualization" / "cluster_fwe"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Read summary CSV
    csv_file = cluster_fwe_dir / "cluster_fwe_results.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"\nLoaded {len(df)} contrasts from CSV")

    # Find NIfTI files
    nii_files = list(cluster_fwe_dir.glob("*_clusterFWE.nii"))
    print(f"Found {len(nii_files)} cluster FWE NIfTI files")

    if not nii_files:
        print("No cluster FWE NIfTI files found!")
        return

    # Define ordering based on design (F-contrasts only)
    if args.design == "hierarchical":
        ordered_names = [
            "Initial_only",
            "Middle_only",
            "Late_only",
            "Final_only",
        ]
        display_names = {
            "Initial_only": "初期層",
            "Middle_only": "中間層",
            "Late_only": "後期層",
            "Final_only": "最終層",
        }
    else:  # designO - F-contrasts only, in order (Early, Middle, Late)
        ordered_names = [
            "F_EarlyLayer", "F_MiddleLayer", "F_LateLayer"
        ]
        display_names = {
            "F_EarlyLayer": "Early Layer",
            "F_MiddleLayer": "Middle Layer",
            "F_LateLayer": "Late Layer"
        }
        # Only process F-contrasts for Design O
        nii_files = [f for f in nii_files if f.stem.startswith("F_")]

    # Build file mapping
    file_map = {}
    for nii_file in nii_files:
        name = nii_file.stem.replace("_clusterFWE", "")
        file_map[name] = nii_file

    # Collect results with data
    results = []
    for name in ordered_names:
        if name in file_map:
            nii_file = file_map[name]
            try:
                img = nib.load(str(nii_file))
                data = img.get_fdata()
                n_voxels = int(np.sum(data != 0))
                if n_voxels > 0:
                    peak_val = np.max(data)
                    display_name = display_names.get(name, name)
                    results.append({
                        "name": name,
                        "display_name": display_name,
                        "img": img,
                        "n_voxels": n_voxels,
                        "peak_val": peak_val,
                    })
                    print(f"{display_name}: {n_voxels:,} voxels, peak={peak_val:.2f}")
            except Exception as e:
                print(f"Error loading {nii_file}: {e}")
        else:
            # Try alternative naming from CSV
            row = df[df['name'].str.contains(name.replace("_", " ").replace("  ", " > "), case=False, na=False)]
            if not row.empty and row.iloc[0]['n_voxels'] > 0:
                print(f"Note: {name} found in CSV but not as NIfTI file")

    # Note: Only showing ordered_names, not adding extra files

    if not results:
        print("No significant clusters to visualize!")
        return

    print(f"\nVisualizing {len(results)} contrasts with significant clusters")

    # Create figure (layer name only on right side)
    n_results = len(results)
    fig = plt.figure(figsize=(14, 3 * n_results), facecolor='white')

    gs = GridSpec(n_results, 2, figure=fig, width_ratios=[6, 1],
                  hspace=0.3, wspace=0.05)

    for row_idx, result in enumerate(results):
        ax_brain = fig.add_subplot(gs[row_idx, 0])
        ax_brain.set_facecolor('white')

        # Get threshold (minimum non-zero value in the image)
        data = result["img"].get_fdata()
        non_zero = data[data > 0]
        threshold = np.min(non_zero) if len(non_zero) > 0 else 0

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

        # Layer name only
        ax_text = fig.add_subplot(gs[row_idx, 1])
        ax_text.axis('off')
        ax_text.set_facecolor('white')

        ax_text.text(0.1, 0.5, result['display_name'],
                    fontsize=21,
                    fontweight='bold',
                    verticalalignment='center')

    plt.suptitle(f'{args.source.upper()}',
                fontsize=21, fontweight='bold')

    # Save figure
    design_filename = "hierarchical" if args.design == "hierarchical" else args.design
    output_filename = f'{args.source}_{design_filename}_cluster_fwe.png'
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"Cluster-level FWE Results Summary")
    print(f"Design: {args.design}, Source: {args.source}")
    print("=" * 70)
    print(f"{'Contrast':<30} {'Clusters':>10} {'Voxels':>10} {'Peak':>10}")
    print("-" * 70)
    for result in results:
        # Get cluster count from CSV
        row = df[df['name'] == result['name']]
        n_clusters = row.iloc[0]['n_clusters'] if not row.empty else "?"
        print(f"{result['display_name']:<30} {n_clusters:>10} {result['n_voxels']:>10,} {result['peak_val']:>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
