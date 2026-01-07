#!/usr/bin/env python3
"""
Visualize Beta RMS for ROI Analysis
Create grid barplot with mean ± SEM across subjects
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

# ROI groups (same as create_grid_barplot.py)
ROI_GROUPS = {
    '視覚野': [
        ('V1_bi', 'V1'),
        ('EarlyVisual_bi', '初期視覚野'),
        ('LOC_bi', 'LOC'),
        ('Fusiform_bi', '紡錘状回'),
        ('IT_bi', 'IT'),
        ('PHC_bi', '海馬傍回'),
    ],
    '空間注意系': [
        ('SPL_bi', 'SPL'),
        ('IPL_bi', 'IPL'),
        ('Precuneus_bi', '楔前部'),
    ],
    '記憶系': [
        ('Hippocampus_bi', '海馬'),
        ('PCC_bi', 'PCC'),
    ],
    '言語野': [
        ('IFG_bi', 'IFG'),
        ('MTG_bi', 'MTG'),
        ('STG_bi', 'STG'),
        ('TemporalPole_bi', '側頭極'),
        ('AngularGyrus_L', '左角回'),
        ('AngularGyrus_R', '右角回'),
    ],
    '報酬系': [
        ('NAcc_bi', 'NAcc'),
        ('OFC_bi', 'OFC'),
    ],
    '価値判断系': [
        ('vmPFC_bi', 'vmPFC'),
        ('DLPFC_bi', 'DLPFC'),
    ],
    '習慣学習系': [
        ('Caudate_bi', '尾状核'),
        ('Putamen_bi', '被殻'),
        ('Thalamus_bi', '視床'),
    ],
    '注意選択系': [
        ('Amygdala_bi', '扁桃体'),
        ('Insula_bi', '島皮質'),
        ('ACC_bi', 'ACC'),
    ],
}

# Layer contrasts
LAYER_CONTRASTS = [
    ('Initial_only', '初期層'),
    ('Middle_only', '中間層'),
    ('Late_only', '後期層'),
    ('Final_only', '最終層'),
]

LAYER_CONTRASTS_ALL = [
    ('Initial_only', '初期層'),
    ('Middle_only', '中間層'),
    ('Late_only', '後期層'),
    ('Final_only', '最終層'),
    ('Shared_Initial_Middle', '共有(初-中)'),
    ('Shared_Middle_Late', '共有(中-後)'),
    ('Shared_Late_Final', '共有(後-最)'),
    ('Global_F', 'Global'),
]

LAYER_CONTRASTS_WITHSHARED = [
    ('Initial_withShared', '初期層+共有'),
    ('Middle_withShared', '中間層+共有'),
    ('Late_withShared', '後期層+共有'),
    ('Final_withShared', '最終層+共有'),
]

LAYER_CONTRASTS_WITHSHARED_WITHGLOBAL = [
    ('Initial_withShared_withGlobal', '初期層+共有+Global'),
    ('Middle_withShared_withGlobal', '中間層+共有+Global'),
    ('Late_withShared_withGlobal', '後期層+共有+Global'),
    ('Final_withShared_withGlobal', '最終層+共有+Global'),
]

LAYER_CONTRASTS_WITHSHARED_PLUS_GLOBAL = [
    ('Initial_withShared', '初期層+共有'),
    ('Middle_withShared', '中間層+共有'),
    ('Late_withShared', '後期層+共有'),
    ('Final_withShared', '最終層+共有'),
    ('Global_F', 'Global'),
]


def load_significance(svc_dir: Path, source: str) -> dict:
    """Load significance information from SVC results"""
    svc_file = svc_dir / f'{source}_hierarchical_svc.csv'
    if not svc_file.exists():
        return {}

    df = pd.read_csv(svc_file)
    sig_dict = {}
    for _, row in df.iterrows():
        key = (row['layer'] if 'layer' in df.columns else row['contrast'], row['roi'])
        sig_dict[key] = row['significant']
    return sig_dict


def create_grid_barplot_rms(clip_df: pd.DataFrame, convnext_df: pd.DataFrame,
                            output_dir: Path, mode: str = 'layer',
                            svc_dir: Path = None):
    """Create grid barplot with RMS values and SEM error bars"""

    # Select contrast list based on mode
    if mode == 'all':
        contrasts = LAYER_CONTRASTS_ALL
    elif mode == 'withshared':
        contrasts = LAYER_CONTRASTS_WITHSHARED
    elif mode == 'withshared_withglobal':
        contrasts = LAYER_CONTRASTS_WITHSHARED_WITHGLOBAL
    elif mode == 'withshared_plus_global':
        contrasts = LAYER_CONTRASTS_WITHSHARED_PLUS_GLOBAL
    else:
        contrasts = LAYER_CONTRASTS

    # Load significance if available
    clip_sig = {}
    convnext_sig = {}
    if svc_dir:
        clip_sig = load_significance(svc_dir, 'clip')
        convnext_sig = load_significance(svc_dir, 'convnext')

    # Find ROIs that are significant in at least one model/layer
    significant_rois = set()
    for (layer_name, _) in contrasts:
        for sig_dict in [clip_sig, convnext_sig]:
            for (layer, roi), is_sig in sig_dict.items():
                if layer == layer_name and is_sig:
                    significant_rois.add(roi)

    # Find ROIs that have data
    all_rois = set(clip_df['roi'].unique()) | set(convnext_df['roi'].unique())

    # Filter ROI_GROUPS to only include significant ROIs
    filtered_groups = {}
    for group_name, rois in ROI_GROUPS.items():
        filtered_rois = [(roi_name, roi_label) for roi_name, roi_label in rois
                         if roi_name in all_rois and roi_name in significant_rois]
        if filtered_rois:
            filtered_groups[group_name] = filtered_rois

    print(f"ROIs with data: {len(all_rois)}")
    print(f"Significant ROIs: {len(significant_rois)}")
    print(f"Groups with significant ROIs: {len(filtered_groups)}")

    n_groups = len(filtered_groups)
    width_ratios = [len(rois) for rois in filtered_groups.values()]
    total_width = sum(width_ratios)
    fig_width = total_width * 1.2 + 2

    fig, axes = plt.subplots(2, n_groups, figsize=(fig_width, 10), sharey=True,
                             gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.25, 'hspace': 0.6})

    # Colors for each contrast
    if mode == 'all':
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        bar_width = 0.09
    elif mode == 'layer_and_shared':
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bar_width = 0.13
    elif mode == 'withshared_plus_global':
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f']  # Global is gray
        bar_width = 0.13
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bar_width = 0.15

    for source_idx, (source, df, sig_dict) in enumerate([
            ('convnext', convnext_df, convnext_sig),
            ('clip', clip_df, clip_sig)]):
        for group_idx, (group_name, rois) in enumerate(filtered_groups.items()):
            ax = axes[source_idx, group_idx]

            x = np.arange(len(rois))

            for i, (layer_name, layer_label) in enumerate(contrasts):
                values = []
                errors = []
                has_sig = []

                for roi_name, roi_label in rois:
                    # Check significance first
                    sig_key = (layer_name, roi_name)
                    is_sig = sig_dict.get(sig_key, False)

                    if not is_sig:
                        # Not significant - don't show bar
                        values.append(0)
                        errors.append(0)
                        has_sig.append(False)
                        continue

                    # Get data for this layer and ROI
                    mask = (df['layer'] == layer_name) & (df['roi'] == roi_name)
                    roi_df = df[mask]

                    if not roi_df.empty:
                        # Calculate mean and SEM across subjects
                        mean_rms = roi_df['rms'].mean()
                        sem_rms = roi_df['rms'].std() / np.sqrt(len(roi_df))
                        values.append(mean_rms)
                        errors.append(sem_rms)
                        has_sig.append(True)
                    else:
                        values.append(0)
                        errors.append(0)
                        has_sig.append(False)

                positions = x + i * bar_width
                bars = ax.bar(positions, values, bar_width, color=colors[i],
                             yerr=errors, capsize=2,
                             error_kw={'elinewidth': 1, 'capthick': 1})

                # Add significance markers
                for bar, is_sig, err in zip(bars, has_sig, errors):
                    if is_sig:
                        ax.text(bar.get_x() + bar.get_width()/2,
                               bar.get_height() + err + 0.002,
                               '*', ha='center', va='bottom',
                               fontsize=21, fontweight='bold')

            n_bars = len(contrasts)
            ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
            ax.set_xticklabels([l for _, l in rois], rotation=45, ha='right', fontsize=21)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='y', labelsize=18)

            # Group title on top row only
            if source_idx == 0:
                ax.set_title(group_name, fontsize=24, fontweight='bold')

            # Y-axis label on leftmost column only
            if group_idx == 0:
                ax.set_ylabel('効果量', fontsize=21)
                ax.text(-0.12, 1.05, source.upper(), transform=ax.transAxes,
                       fontsize=24, fontweight='bold', va='bottom', ha='left')

    # Set consistent ylim across all axes (find max across both models)
    max_ylim = 0
    for ax_row in axes:
        for ax in ax_row:
            max_ylim = max(max_ylim, ax.get_ylim()[1])
    for ax_row in axes:
        for ax in ax_row:
            ax.set_ylim(0, max_ylim + 0.1)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    labels = [l for _, l in contrasts]
    ncol = len(contrasts)
    fig.legend(handles, labels, loc='upper center', ncol=ncol,
               fontsize=21, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    suffix = f'_{mode}' if mode != 'layer' else ''
    output_path = output_dir / f'roi_rms_barplot{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize Beta RMS for ROI analysis')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--mode', type=str, default='layer',
                       choices=['layer', 'layer_and_shared', 'all', 'withshared', 'withshared_withglobal', 'withshared_plus_global'],
                       help='Contrast mode')
    parser.add_argument('--all', action='store_true', help='Generate all versions')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    rms_dir = root / 'results' / 'dnn_analysis' / 'roi_analysis' / 'beta_rms'
    svc_dir = root / 'results' / 'dnn_analysis' / 'roi_analysis' / 'hierarchical_svc'
    output_dir = Path(args.output) if args.output else root / 'paper' / 'image'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load RMS data
    clip_file = rms_dir / 'clip_beta_rms.csv'
    convnext_file = rms_dir / 'convnext_beta_rms.csv'

    if not clip_file.exists() or not convnext_file.exists():
        print(f"RMS files not found. Run extract_beta_rms.m first.")
        print(f"Expected: {clip_file}")
        print(f"Expected: {convnext_file}")
        return

    clip_df = pd.read_csv(clip_file)
    convnext_df = pd.read_csv(convnext_file)

    print(f"CLIP: {len(clip_df)} rows, {clip_df['subject'].nunique()} subjects")
    print(f"ConvNeXt: {len(convnext_df)} rows, {convnext_df['subject'].nunique()} subjects")

    # Determine which versions to generate
    if args.all:
        modes = ['layer', 'all', 'withshared', 'withshared_withglobal']
    else:
        modes = [args.mode]

    for mode in modes:
        print(f"\nGenerating {mode} version...")
        create_grid_barplot_rms(clip_df, convnext_df, output_dir, mode,
                                svc_dir if svc_dir.exists() else None)

    print("\nDone!")


if __name__ == '__main__':
    main()
