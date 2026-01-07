#!/usr/bin/env python3
"""
Create simplified ROI heatmap without shared layers
"""

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
import matplotlib.pyplot as plt
import matplotlib
from scipy import io as sio
matplotlib.rcParams['font.family'] = 'Hiragino Sans'


def f_to_partial_eta_squared(f_value, df1, df2):
    """Convert F value to partial eta-squared"""
    return (df1 * f_value) / (df1 * f_value + df2)


def extract_roi_effectsize(spm_f_img, roi_mask, df1, df2):
    """Extract effect size from ROI"""
    if roi_mask.shape != spm_f_img.shape[:3]:
        roi_mask = image.resample_to_img(
            roi_mask, spm_f_img,
            interpolation='nearest',
            force_resample=True
        )

    f_data = spm_f_img.get_fdata()
    roi_data = roi_mask.get_fdata()
    roi_mask_bool = roi_data > 0
    roi_values = f_data[roi_mask_bool]
    valid_values = roi_values[~np.isnan(roi_values) & (roi_values > 0)]

    if len(valid_values) == 0:
        return {'max_eta2p': 0, 'n_voxels': 0}

    eta2p_values = f_to_partial_eta_squared(valid_values, df1, df2)
    return {
        'max_eta2p': float(np.max(eta2p_values)),
        'n_voxels': len(valid_values)
    }


def check_roi_svc_significance(cluster_img, roi_mask):
    """Check if ROI has significant voxels"""
    if roi_mask.shape != cluster_img.shape[:3]:
        roi_mask = image.resample_to_img(
            roi_mask, cluster_img,
            interpolation='nearest',
            force_resample=True
        )

    cluster_data = cluster_img.get_fdata()
    roi_data = roi_mask.get_fdata()
    roi_mask_bool = roi_data > 0
    cluster_in_roi = cluster_data[roi_mask_bool]
    return int(np.sum(cluster_in_roi > 0)) > 0


def load_spm_df(spm_mat_path):
    """Load degrees of freedom from SPM.mat"""
    mat = sio.loadmat(str(spm_mat_path), struct_as_record=False, squeeze_me=True)
    spm = mat['SPM']
    df2 = float(spm.xX.erdf)
    xCon = spm.xCon
    if not hasattr(xCon, '__len__'):
        xCon = [xCon]
    for con in xCon:
        if hasattr(con, 'STAT') and con.STAT == 'F':
            df1 = float(con.eidf)
            return df1, df2
    return 1.0, df2


def create_simple_heatmap(source: str, output_dir: Path, significant_rois: set, mode: str = 'layer'):
    """Create heatmap for a single model

    Args:
        mode: 'layer' (default), 'global' (layer + Global), 'shared' (layer + Shared)
    """

    root = Path(__file__).resolve().parents[3]

    svc_csv = root / 'results' / 'dnn_analysis' / 'roi_analysis' / 'hierarchical_svc' / f'{source}_hierarchical_svc.csv'
    if svc_csv.exists():
        df = pd.read_csv(svc_csv)
        print(f"Loaded SVC results from {svc_csv}")
    else:
        raise FileNotFoundError(f"SVC results not found: {svc_csv}")

    # ROI groups (same as barplot, with literature references)
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

    if mode == 'global':
        contrast_order = ['Global_F', 'Initial_only', 'Middle_only', 'Late_only', 'Final_only']
        contrast_labels_jp = ['Global', '初期層', '中間層', '後期層', '最終層']
    elif mode == 'shared':
        contrast_order = ['Shared_F', 'Initial_only', 'Middle_only', 'Late_only', 'Final_only']
        contrast_labels_jp = ['Shared', '初期層', '中間層', '後期層', '最終層']
    elif mode == 'withshared':
        contrast_order = ['Initial_withShared', 'Middle_withShared', 'Late_withShared', 'Final_withShared']
        contrast_labels_jp = ['初期層+Shared', '中間層+Shared', '後期層+Shared', '最終層+Shared']
    elif mode == 'global_withshared':
        contrast_order = ['Initial_withShared', 'Middle_withShared', 'Late_withShared', 'Final_withShared', 'Global_F']
        contrast_labels_jp = ['初期層+Shared', '中間層+Shared', '後期層+Shared', '最終層+Shared', 'Global']
    elif mode == 'withshared_withglobal':
        contrast_order = ['Initial_withShared_withGlobal', 'Middle_withShared_withGlobal', 'Late_withShared_withGlobal', 'Final_withShared_withGlobal']
        contrast_labels_jp = ['初期層+Shared+Global', '中間層+Shared+Global', '後期層+Shared+Global', '最終層+Shared+Global']
    elif mode == 'layer_and_shared':
        contrast_order = ['Initial_only', 'Middle_only', 'Late_only', 'Final_only', 'Shared_F']
        contrast_labels_jp = ['初期層', '中間層', '後期層', '最終層', 'Shared']
    elif mode == 'layer_and_shared_individual':
        contrast_order = ['Initial_only', 'Middle_only', 'Late_only', 'Final_only',
                         'Shared_Initial_Middle_F', 'Shared_Middle_Late_F', 'Shared_Late_Final_F']
        contrast_labels_jp = ['初期層', '中間層', '後期層', '最終層',
                             'Shared(初-中)', 'Shared(中-後)', 'Shared(後-最)']
    else:
        contrast_order = ['Initial_only', 'Middle_only', 'Late_only', 'Final_only']
        contrast_labels_jp = ['初期層', '中間層', '後期層', '最終層']

    # Flatten ROI order with group boundaries, excluding non-significant ROIs
    roi_order = []
    roi_labels = []
    group_boundaries = []  # x positions for vertical lines
    group_centers = []     # x positions for group labels
    group_names_filtered = []  # only groups with significant ROIs
    current_pos = 0

    for group_name, rois in ROI_GROUPS.items():
        group_start = current_pos
        group_has_sig = False
        for roi_name, roi_label in rois:
            if roi_name in significant_rois:
                roi_order.append(roi_name)
                roi_labels.append(roi_label)
                current_pos += 1
                group_has_sig = True
        if group_has_sig:
            group_centers.append((group_start + current_pos - 1) / 2)
            group_boundaries.append(current_pos - 0.5)
            group_names_filtered.append(group_name)

    # Remove last boundary (no line after last group)
    if group_boundaries:
        group_boundaries = group_boundaries[:-1]

    # Create pivot tables (use peak_eta2p for SVC analysis)
    pivot_eta2 = df.pivot_table(index='contrast', columns='roi', values='peak_eta2p', aggfunc='first')
    pivot_sig = df.pivot_table(index='contrast', columns='roi', values='significant', aggfunc='first')

    # Reorder
    available_rois = [r for r in roi_order if r in pivot_eta2.columns]
    available_contrasts = [c for c in contrast_order if c in pivot_eta2.index]

    pivot_eta2 = pivot_eta2.reindex(available_contrasts)[available_rois]
    pivot_sig = pivot_sig.reindex(available_contrasts)[available_rois]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 5))

    # Background gray for non-significant
    ax.imshow(np.ones_like(pivot_eta2.values) * 0.5, cmap='Greys', aspect='auto', vmin=0, vmax=1, alpha=0.3)

    # Significant cells with color (peak_eta2p)
    im = ax.imshow(pivot_eta2.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)

    # Make non-significant cells gray
    for i in range(len(available_contrasts)):
        for j in range(len(available_rois)):
            if not pivot_sig.values[i, j]:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='lightgray', alpha=0.8))

    # Add vertical lines to separate categories
    for boundary in group_boundaries:
        ax.axvline(x=boundary, color='black', linewidth=2)

    # Labels (1.5x font size)
    ax.set_xticks(range(len(available_rois)))
    ax.set_xticklabels([roi_labels[roi_order.index(r)] for r in available_rois],
                       rotation=45, ha='right', fontsize=17)

    ax.set_yticks(range(len(available_contrasts)))
    ax.set_yticklabels(contrast_labels_jp, fontsize=18)

    # Group labels at top
    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks(group_centers)
    ax2.set_xticklabels(group_names_filtered, fontsize=17, fontweight='bold')
    ax2.tick_params(length=0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('効果量', fontsize=17)
    cbar.ax.tick_params(labelsize=14)

    # Add annotations
    for i in range(len(available_contrasts)):
        for j in range(len(available_rois)):
            val = pivot_eta2.values[i, j]
            is_sig = pivot_sig.values[i, j]
            if is_sig and val > 0:
                color = 'white' if val > 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=12, color=color, fontweight='bold')

    ax.set_title(source.upper(), fontsize=24, fontweight='bold', loc='left')

    plt.tight_layout()

    suffix = f'_{mode}' if mode != 'layer' else ''
    output_path = output_dir / f'{source}_designN_v3_svc_heatmap{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create simplified ROI heatmap')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--mode', type=str, default='layer',
                       choices=['layer', 'global', 'shared', 'withshared', 'global_withshared',
                                'withshared_withglobal', 'layer_and_shared', 'layer_and_shared_individual'],
                       help='Contrast mode')
    parser.add_argument('--all', action='store_true', help='Generate all versions')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output) if args.output else root / 'paper' / 'image'
    output_dir.mkdir(parents=True, exist_ok=True)

    svc_dir = root / 'results' / 'dnn_analysis' / 'roi_analysis' / 'hierarchical_svc'

    # Load both CSVs
    clip_df = pd.read_csv(svc_dir / 'clip_hierarchical_svc.csv')
    convnext_df = pd.read_csv(svc_dir / 'convnext_hierarchical_svc.csv')

    # Determine which versions to generate
    if args.all:
        modes = ['layer', 'global_withshared', 'layer_and_shared', 'layer_and_shared_individual', 'withshared_withglobal']
    else:
        modes = [args.mode]

    for mode in modes:
        # Find ROIs significant in at least one model
        if mode == 'global':
            contrast_order = ['Global_F', 'Initial_only', 'Middle_only', 'Late_only', 'Final_only']
        elif mode == 'shared':
            contrast_order = ['Shared_F', 'Initial_only', 'Middle_only', 'Late_only', 'Final_only']
        elif mode == 'withshared':
            contrast_order = ['Initial_withShared', 'Middle_withShared', 'Late_withShared', 'Final_withShared']
        elif mode == 'global_withshared':
            contrast_order = ['Initial_withShared', 'Middle_withShared', 'Late_withShared', 'Final_withShared', 'Global_F']
        elif mode == 'withshared_withglobal':
            contrast_order = ['Initial_withShared_withGlobal', 'Middle_withShared_withGlobal', 'Late_withShared_withGlobal', 'Final_withShared_withGlobal']
        elif mode == 'layer_and_shared':
            contrast_order = ['Initial_only', 'Middle_only', 'Late_only', 'Final_only', 'Shared_F']
        elif mode == 'layer_and_shared_individual':
            contrast_order = ['Initial_only', 'Middle_only', 'Late_only', 'Final_only',
                             'Shared_Initial_Middle_F', 'Shared_Middle_Late_F', 'Shared_Late_Final_F']
        else:
            contrast_order = ['Initial_only', 'Middle_only', 'Late_only', 'Final_only']

        significant_rois = set()
        for df in [clip_df, convnext_df]:
            for contrast in contrast_order:
                sig_df = df[(df['contrast'] == contrast) & (df['significant'] == True)]
                significant_rois.update(sig_df['roi'].tolist())

        print(f"Significant ROIs ({mode}): {len(significant_rois)}")

        # Create separate heatmaps for each model
        for source in ['clip', 'convnext']:
            create_simple_heatmap(source, output_dir, significant_rois, mode)

    print("\nDone!")


if __name__ == '__main__':
    main()
