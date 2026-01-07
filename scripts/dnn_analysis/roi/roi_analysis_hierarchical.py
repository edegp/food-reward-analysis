#!/usr/bin/env python3
"""
ROI Analysis for Hierarchical using Cluster FWE corrected maps (v2 - proper RFT)

This script analyzes ROI statistics using cluster FWE corrected maps.
A ROI is considered significant if it contains any voxels from a significant cluster.
"""

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'


def extract_roi_stats(cluster_fwe_img, roi_mask):
    """
    Extract statistics from ROI using cluster FWE corrected map
    """
    # Resample ROI mask to match stat image if needed
    if roi_mask.shape != cluster_fwe_img.shape[:3]:
        roi_mask = image.resample_to_img(roi_mask, cluster_fwe_img, interpolation='nearest')

    # Get data
    fwe_data = cluster_fwe_img.get_fdata()
    roi_data = roi_mask.get_fdata()

    # Extract values within ROI
    roi_mask_bool = roi_data > 0
    roi_values = fwe_data[roi_mask_bool]

    # Count significant voxels (non-zero in cluster FWE map)
    n_significant = int(np.sum(roi_values > 0))
    n_total = int(np.sum(roi_mask_bool))

    result = {
        'n_voxels_roi': n_total,
        'n_significant': n_significant,
        'pct_significant': 100 * n_significant / n_total if n_total > 0 else 0,
        'has_significant': n_significant > 0,
        'max_F': float(np.max(roi_values)) if n_significant > 0 else 0,
        'mean_F_sig': float(np.mean(roi_values[roi_values > 0])) if n_significant > 0 else 0
    }

    return result


def run_roi_analysis(cluster_fwe_dir: Path, roi_dir: Path, output_dir: Path, source: str):
    """Run ROI analysis for Hierarchical"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find ROI masks
    roi_files = list(roi_dir.glob('*_mask.nii')) + list(roi_dir.glob('*_mask.nii.gz'))
    print(f"Found {len(roi_files)} ROI masks")

    # Find cluster FWE NIfTI files
    nii_files = list(cluster_fwe_dir.glob('*_clusterFWE.nii'))
    print(f"Found {len(nii_files)} cluster FWE maps")

    if not nii_files:
        print("No cluster FWE maps found!")
        return None

    results = []

    for nii_file in nii_files:
        contrast_name = nii_file.stem.replace('_clusterFWE', '')
        print(f"\nProcessing: {contrast_name}")

        fwe_img = nib.load(str(nii_file))

        for roi_file in roi_files:
            roi_name = roi_file.stem.replace('_mask', '')
            roi_mask = nib.load(str(roi_file))

            stats = extract_roi_stats(fwe_img, roi_mask)

            results.append({
                'source': source,
                'contrast': contrast_name,
                'roi': roi_name,
                **stats
            })

            if stats['has_significant']:
                print(f"  {roi_name}: {stats['n_significant']} voxels ({stats['pct_significant']:.1f}%)")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    csv_path = output_dir / f'{source}_hierarchical_roi_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    return df


def create_heatmap(df: pd.DataFrame, output_dir: Path, source: str):
    """Create heatmap of ROI significance"""

    # Pivot table for heatmap
    pivot = df.pivot_table(
        index='roi',
        columns='contrast',
        values='pct_significant',
        aggfunc='first'
    )

    # Define ROI order (visual hierarchy)
    roi_order = [
        'V1_bi', 'EarlyVisual_bi', 'LOC_bi', 'Fusiform_bi', 'IT_bi',
        'SPL_bi', 'IPL_bi', 'Precuneus_bi',
        'Hippocampus_bi', 'PHC_bi',
        'TemporalPole_bi', 'STG_bi', 'MTG_bi',
        'Insula_bi', 'Amygdala_bi',
        'ACC_bi', 'PCC_bi',
        'OFC_bi', 'vmPFC_bi',
        'Striatum_bi', 'NAcc_bi', 'Caudate_bi', 'Putamen_bi',
        'Thalamus_bi'
    ]

    # Filter to available ROIs
    available_rois = [r for r in roi_order if r in pivot.index]
    pivot = pivot.reindex(available_rois)

    # Contrast order
    contrast_order = [
        'Initial_only', 'Initial_withShared',
        'Middle_only', 'Middle_withShared',
        'Late_only', 'Late_withShared',
        'Final_only', 'Final_withShared',
        'Shared_F', 'Global_F'
    ]
    available_contrasts = [c for c in contrast_order if c in pivot.columns]
    pivot = pivot[available_contrasts]

    # Japanese labels
    roi_labels_jp = {
        'V1_bi': 'V1',
        'EarlyVisual_bi': '初期視覚野',
        'LOC_bi': 'LOC',
        'Fusiform_bi': '紡錘状回',
        'IT_bi': 'IT',
        'SPL_bi': '上頭頂小葉',
        'IPL_bi': '下頭頂小葉',
        'Precuneus_bi': '楔前部',
        'Hippocampus_bi': '海馬',
        'PHC_bi': '海馬傍回',
        'TemporalPole_bi': '側頭極',
        'STG_bi': '上側頭回',
        'MTG_bi': '中側頭回',
        'Insula_bi': '島皮質',
        'Amygdala_bi': '扁桃体',
        'ACC_bi': 'ACC',
        'PCC_bi': 'PCC',
        'OFC_bi': 'OFC',
        'vmPFC_bi': 'vmPFC',
        'Striatum_bi': '線条体',
        'NAcc_bi': '側坐核',
        'Caudate_bi': '尾状核',
        'Putamen_bi': '被殻',
        'Thalamus_bi': '視床'
    }

    contrast_labels_jp = {
        'Initial_only': '初期層のみ',
        'Initial_withShared': '初期層+共有',
        'Middle_only': '中間層のみ',
        'Middle_withShared': '中間層+共有',
        'Late_only': '後期層のみ',
        'Late_withShared': '後期層+共有',
        'Final_only': '最終層のみ',
        'Final_withShared': '最終層+共有',
        'Shared_F': '共有PC',
        'Global_F': '全体'
    }

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    # Labels
    ax.set_xticks(range(len(available_contrasts)))
    ax.set_xticklabels([contrast_labels_jp.get(c, c) for c in available_contrasts],
                       rotation=45, ha='right', fontsize=12)

    ax.set_yticks(range(len(available_rois)))
    ax.set_yticklabels([roi_labels_jp.get(r, r) for r in available_rois], fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('有意ボクセル割合 (%)', fontsize=14)

    # Add text annotations
    for i in range(len(available_rois)):
        for j in range(len(available_contrasts)):
            val = pivot.values[i, j]
            if not np.isnan(val) and val > 0:
                color = 'white' if val > 50 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=9, color=color, fontweight='bold')

    ax.set_title(f'{source.upper()} Hierarchical: ROI × コントラスト\n(クラスターFWE補正 p < 0.05)',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / f'{source}_hierarchical_roi_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_barplot(clip_df: pd.DataFrame, convnext_df: pd.DataFrame, output_dir: Path):
    """Create bar plot comparing CLIP and ConvNeXt"""

    # Focus on layer-specific contrasts
    layer_contrasts = ['Initial_only', 'Middle_only', 'Late_only', 'Final_only']

    # Key ROIs for food/reward processing
    key_rois = [
        'V1_bi', 'LOC_bi', 'Fusiform_bi', 'IT_bi',
        'Insula_bi', 'OFC_bi', 'vmPFC_bi',
        'Amygdala_bi', 'Striatum_bi', 'NAcc_bi'
    ]

    roi_labels_jp = {
        'V1_bi': 'V1',
        'LOC_bi': 'LOC',
        'Fusiform_bi': '紡錘状回',
        'IT_bi': 'IT',
        'Insula_bi': '島皮質',
        'OFC_bi': 'OFC',
        'vmPFC_bi': 'vmPFC',
        'Amygdala_bi': '扁桃体',
        'Striatum_bi': '線条体',
        'NAcc_bi': '側坐核'
    }

    layer_labels_jp = {
        'Initial_only': '初期層',
        'Middle_only': '中間層',
        'Late_only': '後期層',
        'Final_only': '最終層'
    }

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bar_width = 0.2

    for ax, (model_df, model_name) in zip(axes, [(clip_df, 'CLIP'), (convnext_df, 'ConvNeXt')]):
        x = np.arange(len(key_rois))

        for i, contrast in enumerate(layer_contrasts):
            values = []
            has_sig = []

            for roi in key_rois:
                row = model_df[(model_df['contrast'] == contrast) & (model_df['roi'] == roi)]
                if not row.empty:
                    values.append(row['pct_significant'].values[0])
                    has_sig.append(row['has_significant'].values[0])
                else:
                    values.append(0)
                    has_sig.append(False)

            positions = x + i * bar_width
            bars = ax.bar(positions, values, bar_width,
                         label=layer_labels_jp[contrast], color=colors[i])

            # Mark significant ROIs
            for bar, is_sig in zip(bars, has_sig):
                if is_sig:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           '*', ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_xticks(x + bar_width * 1.5)
        ax.set_xticklabels([roi_labels_jp.get(r, r) for r in key_rois],
                          rotation=45, ha='right', fontsize=14)

        ax.set_ylabel('有意ボクセル割合 (%)', fontsize=14)
        ax.set_title(model_name, fontsize=16, fontweight='bold', loc='left')
        ax.legend(fontsize=12, loc='upper right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Hierarchical: 層グループ別 ROI効果\n(* クラスターFWE p < 0.05)',
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = output_dir / 'hierarchical_roi_barplot_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ROI Analysis for Hierarchical')
    parser.add_argument('--source', type=str, choices=['clip', 'convnext'], default=None)
    parser.add_argument('--cluster-fwe-dir', type=Path, default=None)
    parser.add_argument('--roi-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    # Use v2 (proper RFT) results
    cluster_fwe_base = root / 'results' / 'dnn_analysis' / 'cluster_fwe_v2'

    if args.roi_dir is None:
        args.roi_dir = root / 'rois' / 'HarvardOxford'

    if args.output_dir is None:
        args.output_dir = root / 'results' / 'dnn_analysis' / 'roi_analysis' / 'hierarchical'

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ROI directory: {args.roi_dir}")
    print(f"Output directory: {args.output_dir}")

    # Run analysis for both sources if not specified
    sources = [args.source] if args.source else ['clip', 'convnext']

    dfs = {}
    for source in sources:
        cluster_fwe_dir = cluster_fwe_base / f'hierarchical_{source}'

        if not cluster_fwe_dir.exists():
            print(f"\nWarning: {cluster_fwe_dir} not found, skipping {source}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {source.upper()}")
        print(f"Cluster FWE directory: {cluster_fwe_dir}")
        print(f"{'='*60}")

        df = run_roi_analysis(cluster_fwe_dir, args.roi_dir, args.output_dir, source)

        if df is not None:
            dfs[source] = df
            create_heatmap(df, args.output_dir, source)

    # Create comparison barplot if both sources available
    if 'clip' in dfs and 'convnext' in dfs:
        print("\nCreating comparison barplot...")
        create_barplot(dfs['clip'], dfs['convnext'], args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
