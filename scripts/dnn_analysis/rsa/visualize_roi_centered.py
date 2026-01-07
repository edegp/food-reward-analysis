#!/usr/bin/env python3
"""
Visualize ROI-centered RSA results.
Generates two figures:
1. rsa_roi_centered_comparison_jp.png - Bar plot of NC% for all ROIs
2. rsa_roi_centered_summary_jp.png - Summary bar plot comparing models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import argparse

ROOT = Path("/Users/yuhiaoki/dev/hit/food-brain")
OUTPUT_DIR = Path("/Volumes/Extreme Pro/hit/food-brain/results/rsa_analysis/roi_centered")
PAPER_IMAGE_DIR = ROOT / "paper" / "image"

# Japanese ROI names mapping
ROI_NAMES_JP = {
    'V1': 'V1',
    'EarlyVisual': '初期視覚野',
    'LOC': 'LOC',
    'Fusiform': '紡錘状回',
    'IT': '下側頭皮質',
    'IPL': '下頭頂小葉',
    'SPL': '上頭頂小葉',
    'AngularGyrus': '角回',
    'Precuneus': '楔前部',
    'STG': '上側頭回',
    'MTG': '中側頭回',
    'TemporalPole': '側頭極',
    'PHC': '海馬傍回',
    'IFG': '下前頭回',
    'DLPFC': '背外側前頭前野',
    'VLPFC': '腹外側前頭前野',
    'OFC': '眼窩前頭皮質',
    'FrontalPole': '前頭極',
    'Broca_L': 'ブローカ野',
    'vmPFC': '腹内側前頭前野',
    'ACC': '前帯状皮質',
    'PCC': '後帯状皮質',
    'Insula': '島皮質',
    'Hippocampus': '海馬',
    'Amygdala': '扁桃体',
    'Caudate': '尾状核',
    'Putamen': '被殻',
    'NAcc': '側坐核',
    'Thalamus': '視床',
    'Pallidum': '淡蒼球',
    'Striatum': '線条体',
}

# Visual ROIs (to exclude for non-visual analysis)
VISUAL_ROIS = ['V1', 'EarlyVisual', 'LOC', 'Fusiform', 'IT']

# ROI order for plotting (visual -> parietal -> temporal -> frontal -> cingulate -> insula -> subcortical)
ROI_ORDER = [
    'V1', 'EarlyVisual', 'LOC', 'Fusiform', 'IT',  # Visual
    'IPL', 'SPL', 'AngularGyrus', 'Precuneus',  # Parietal
    'STG', 'MTG', 'TemporalPole', 'PHC',  # Temporal
    'IFG', 'DLPFC', 'VLPFC', 'OFC', 'FrontalPole', 'Broca_L', 'vmPFC',  # Frontal
    'ACC', 'PCC',  # Cingulate
    'Insula',  # Insula
    'Hippocampus', 'Amygdala', 'Caudate', 'Putamen', 'NAcc', 'Thalamus', 'Pallidum', 'Striatum',  # Subcortical
]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load ROI centered summary CSV."""
    df = pd.read_csv(csv_path)
    return df


def create_comparison_plot(df: pd.DataFrame, output_path: Path):
    """
    Create bar plot comparing NC% for all ROIs across 3 models.
    """
    plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']

    # Sort by ROI order
    df_sorted = df.set_index('ROI').loc[ROI_ORDER].reset_index()

    # Get Japanese names
    roi_labels = [ROI_NAMES_JP.get(roi, roi) for roi in df_sorted['ROI']]

    fig, ax = plt.subplots(figsize=(20, 6))

    x = np.arange(len(roi_labels))
    width = 0.25

    colors = {
        'CLIP': '#1f77b4',      # Blue
        'ImageNet': '#ff7f0e',  # Orange
        'Food': '#2ca02c'       # Green
    }

    # Plot bars
    ax.bar(x - width, df_sorted['CLIP_NC%'], width, label='CLIP', color=colors['CLIP'])
    ax.bar(x, df_sorted['ImageNet_NC%'], width, label='ImageNet', color=colors['ImageNet'])
    ax.bar(x + width, df_sorted['Food_NC%'], width, label='Food', color=colors['Food'])

    ax.set_xlabel('ROI', fontsize=14)
    ax.set_ylabel('説明率 (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_labels, rotation=90, fontsize=10)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 75)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_plot(df: pd.DataFrame, output_path: Path):
    """
    Create summary bar plot with two panels:
    - Left: Mean NC% across all ROIs
    - Right: Mean NC% excluding visual ROIs
    """
    plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    colors = {
        'CLIP': '#1f77b4',
        'ImageNet': '#ff7f0e',
        'Food': '#2ca02c'
    }

    models = ['CLIP', 'ImageNet', 'Food']

    # Left panel: All ROIs
    ax1 = axes[0]
    means_all = []
    stds_all = []
    for model in models:
        col = f'{model}_NC%'
        means_all.append(df[col].mean())
        stds_all.append(df[col].std())

    x = np.arange(len(models))
    bars1 = ax1.bar(x, means_all, yerr=stds_all, capsize=5,
                    color=[colors[m] for m in models], alpha=0.85)

    # Add value labels
    for i, (mean, bar) in enumerate(zip(means_all, bars1)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds_all[i] + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Statistical test (CLIP vs ImageNet)
    clip_vals = df['CLIP_NC%'].values
    imagenet_vals = df['ImageNet_NC%'].values
    t_stat, p_val = stats.ttest_rel(clip_vals, imagenet_vals)

    # Add significance bracket
    y_max = max(means_all) + max(stds_all) + 8
    ax1.plot([0, 1], [y_max, y_max], 'k-', linewidth=1)
    ax1.text(0.5, y_max + 1, f'p = {p_val:.3f}', ha='center', fontsize=10)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.set_ylabel('説明率 (%)', fontsize=14)
    ax1.set_ylim(0, 65)

    # Right panel: Non-visual ROIs only
    ax2 = axes[1]
    df_nonvis = df[~df['ROI'].isin(VISUAL_ROIS)]

    means_nonvis = []
    stds_nonvis = []
    for model in models:
        col = f'{model}_NC%'
        means_nonvis.append(df_nonvis[col].mean())
        stds_nonvis.append(df_nonvis[col].std())

    bars2 = ax2.bar(x, means_nonvis, yerr=stds_nonvis, capsize=5,
                    color=[colors[m] for m in models], alpha=0.85)

    # Add value labels
    for i, (mean, bar) in enumerate(zip(means_nonvis, bars2)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds_nonvis[i] + 0.3,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Statistical tests for non-visual
    clip_nonvis = df_nonvis['CLIP_NC%'].values
    imagenet_nonvis = df_nonvis['ImageNet_NC%'].values
    food_nonvis = df_nonvis['Food_NC%'].values

    _, p_clip_img = stats.ttest_rel(clip_nonvis, imagenet_nonvis)
    _, p_img_food = stats.ttest_rel(imagenet_nonvis, food_nonvis)

    # Significance markers
    def get_sig_marker(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return f'p = {p:.3f}'

    # CLIP vs ImageNet bracket
    y_max1 = max(means_nonvis[:2]) + max(stds_nonvis[:2]) + 1.5
    ax2.plot([0, 1], [y_max1, y_max1], 'k-', linewidth=1)
    ax2.text(0.5, y_max1 + 0.2, get_sig_marker(p_clip_img), ha='center', fontsize=12)

    # ImageNet vs Food bracket
    y_max2 = max(means_nonvis[1:]) + max(stds_nonvis[1:]) + 1.5
    ax2.plot([1, 2], [y_max2, y_max2], 'k-', linewidth=1)
    ax2.text(1.5, y_max2 + 0.2, get_sig_marker(p_img_food), ha='center', fontsize=12)

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=12)
    ax2.set_ylabel('説明率 (%)', fontsize=14)
    ax2.set_ylim(0, 22)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize ROI-centered RSA results')
    parser.add_argument('--input', type=str, default=None,
                        help='Input CSV file path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for images')
    args = parser.parse_args()

    # Set paths
    csv_path = Path(args.input) if args.input else OUTPUT_DIR / "roi_centered_summary.csv"
    output_dir = Path(args.output_dir) if args.output_dir else PAPER_IMAGE_DIR

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print("Run calc_rsa_roi_centered.py first to generate the data.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} ROIs")

    # Create plots
    comparison_path = output_dir / "rsa_roi_centered_comparison_jp.png"
    summary_path = output_dir / "rsa_roi_centered_summary_jp.png"

    create_comparison_plot(df, comparison_path)
    create_summary_plot(df, summary_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
