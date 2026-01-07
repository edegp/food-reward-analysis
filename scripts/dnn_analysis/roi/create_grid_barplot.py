#!/usr/bin/env python3
"""
Create grid barplot for ROI analysis results
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

# ROI groups based on literature
# Visual: Ventral visual stream hierarchy
#   - Grill-Spector & Malach (2004) Annu Rev Neurosci. doi:10.1146/annurev.neuro.27.070203.144220
#   - Kravitz et al. (2013) Trends Cogn Sci. doi:10.1016/j.tics.2012.10.011
# Parietal: Dorsal attention network
#   - Corbetta & Shulman (2002) Nat Rev Neurosci. doi:10.1038/nrn755
# Memory: Hippocampal memory system
#   - Eichenbaum (2017) Nat Rev Neurosci. doi:10.1038/nrn.2017.74
# Language: Dual stream model
#   - Hickok & Poeppel (2007) Nat Rev Neurosci. doi:10.1038/nrn2113
# Reward: Reward circuitry
#   - Haber & Knutson (2010) Neuropsychopharmacology. doi:10.1038/npp.2009.129
#   - Diekhof et al. (2012) Neuropsychologia. doi:10.1016/j.neuropsychologia.2012.02.007
# Basal Ganglia: Reward learning and habit formation
#   - Haber & Knutson (2010) Neuropsychopharmacology. doi:10.1038/npp.2009.129
# Salience: Salience network
#   - Seeley et al. (2007) J Neurosci. doi:10.1523/JNEUROSCI.5587-06.2007
#   - Menon (2015) Brain Mapping. doi:10.1016/B978-0-12-397025-1.00052-X
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
        ('Broca_L', 'Broca'),
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

LAYER_CONTRASTS = [
    ('Initial_only', '初期層'),
    ('Middle_only', '中間層'),
    ('Late_only', '後期層'),
    ('Final_only', '最終層'),
]

LAYER_CONTRASTS_GLOBAL = [
    ('Global_F', 'Global'),
    ('Initial_only', '初期層'),
    ('Middle_only', '中間層'),
    ('Late_only', '後期層'),
    ('Final_only', '最終層'),
]

LAYER_CONTRASTS_SHARED = [
    ('Shared_F', 'Shared'),
    ('Initial_only', '初期層'),
    ('Middle_only', '中間層'),
    ('Late_only', '後期層'),
    ('Final_only', '最終層'),
]

LAYER_CONTRASTS_WITHSHARED = [
    ('Initial_withShared', '初期層+Shared'),
    ('Middle_withShared', '中間層+Shared'),
    ('Late_withShared', '後期層+Shared'),
    ('Final_withShared', '最終層+Shared'),
]

LAYER_CONTRASTS_GLOBAL_WITHSHARED = [
    ('Initial_withShared', '初期層+Shared'),
    ('Middle_withShared', '中間層+Shared'),
    ('Late_withShared', '後期層+Shared'),
    ('Final_withShared', '最終層+Shared'),
    ('Global_F', 'Global'),
]

LAYER_CONTRASTS_AND_SHARED = [
    ('Initial_only', '初期層'),
    ('Middle_only', '中間層'),
    ('Late_only', '後期層'),
    ('Final_only', '最終層'),
    ('Shared_F', 'Shared'),
]

LAYER_CONTRASTS_AND_SHARED_INDIVIDUAL = [
    ('Initial_only', '初期層'),
    ('Middle_only', '中間層'),
    ('Late_only', '後期層'),
    ('Final_only', '最終層'),
    ('Shared_Initial_Middle_F', 'Shared(初-中)'),
    ('Shared_Middle_Late_F', 'Shared(中-後)'),
    ('Shared_Late_Final_F', 'Shared(後-最)'),
]

LAYER_CONTRASTS_WITHSHARED_WITHGLOBAL = [
    ('Initial_withShared_withGlobal', '初期層+Shared+Global'),
    ('Middle_withShared_withGlobal', '中間層+Shared+Global'),
    ('Late_withShared_withGlobal', '後期層+Shared+Global'),
    ('Final_withShared_withGlobal', '最終層+Shared+Global'),
]


def create_grid_barplot(clip_df: pd.DataFrame, convnext_df: pd.DataFrame, output_dir: Path, mode: str = 'layer'):
    """Create grid barplot with ROIs grouped by function

    Args:
        mode: 'layer' (default), 'global' (layer + Global), 'shared' (layer + Shared), 'withshared' (layer + associated Shared)
    """

    # Select contrast list based on mode
    if mode == 'global':
        contrasts = LAYER_CONTRASTS_GLOBAL
    elif mode == 'shared':
        contrasts = LAYER_CONTRASTS_SHARED
    elif mode == 'withshared':
        contrasts = LAYER_CONTRASTS_WITHSHARED
    elif mode == 'global_withshared':
        contrasts = LAYER_CONTRASTS_GLOBAL_WITHSHARED
    elif mode == 'withshared_withglobal':
        contrasts = LAYER_CONTRASTS_WITHSHARED_WITHGLOBAL
    elif mode == 'layer_and_shared':
        contrasts = LAYER_CONTRASTS_AND_SHARED
    elif mode == 'layer_and_shared_individual':
        contrasts = LAYER_CONTRASTS_AND_SHARED_INDIVIDUAL
    else:
        contrasts = LAYER_CONTRASTS

    # Find ROIs that are significant in at least one model for any layer contrast
    significant_rois = set()
    for df in [clip_df, convnext_df]:
        for contrast, _ in contrasts:
            sig_df = df[(df['contrast'] == contrast) & (df['significant'] == True)]
            significant_rois.update(sig_df['roi'].tolist())

    # Filter ROI_GROUPS to only include significant ROIs
    filtered_groups = {}
    for group_name, rois in ROI_GROUPS.items():
        filtered_rois = [(roi_name, roi_label) for roi_name, roi_label in rois
                         if roi_name in significant_rois]
        if filtered_rois:
            filtered_groups[group_name] = filtered_rois

    print(f"Significant ROIs: {len(significant_rois)} / {sum(len(rois) for rois in ROI_GROUPS.values())}")
    print(f"Groups with significant ROIs: {len(filtered_groups)} / {len(ROI_GROUPS)}")

    n_groups = len(filtered_groups)
    # Calculate width ratios based on number of ROIs in each group
    width_ratios = [len(rois) for rois in filtered_groups.values()]
    total_width = sum(width_ratios)
    fig_width = total_width * 1.2 + 2  # Scale factor for readable bars

    fig, axes = plt.subplots(2, n_groups, figsize=(fig_width, 10), sharey=True,
                             gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.15, 'hspace': 0.5})

    # Colors for each contrast
    if mode == 'layer_and_shared_individual':
        # 7 colors: 4 layers + 3 individual Shared
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        bar_width = 0.11
    elif mode in ['global', 'shared', 'global_withshared', 'layer_and_shared']:
        # 5 colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bar_width = 0.13
    else:
        # 4 layer colors (for both layer and withshared modes)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bar_width = 0.15

    for source_idx, (source, df) in enumerate([('convnext', convnext_df), ('clip', clip_df)]):
        for group_idx, (group_name, rois) in enumerate(filtered_groups.items()):
            ax = axes[source_idx, group_idx]

            x = np.arange(len(rois))

            for i, (contrast_name, contrast_label) in enumerate(contrasts):
                values = []
                errors = []
                has_sig = []

                # Check which columns are available
                # Use peak_eta2p with 95% CI (from noncentral F distribution) for SVC analysis
                if 'eta2p_ci_lower' in df.columns and 'eta2p_ci_upper' in df.columns:
                    use_ci = True
                    value_col = 'peak_eta2p'
                else:
                    use_ci = False
                    value_col = 'peak_eta2p'  # Always use peak for SVC analysis

                errors_lower = []
                errors_upper = []

                for roi_name, roi_label in rois:
                    row = df[(df['contrast'] == contrast_name) & (df['roi'] == roi_name)]
                    if not row.empty and row['significant'].values[0]:
                        val = row[value_col].values[0]
                        values.append(val)
                        if use_ci:
                            # Asymmetric error bars from CI
                            ci_lower = row['eta2p_ci_lower'].values[0]
                            ci_upper = row['eta2p_ci_upper'].values[0]
                            # Handle NaN CI values (numerical instability)
                            if pd.isna(ci_lower) or pd.isna(ci_upper):
                                errors_lower.append(0)
                                errors_upper.append(0)
                            else:
                                errors_lower.append(max(0, val - ci_lower))
                                errors_upper.append(max(0, ci_upper - val))
                        else:
                            # No error bars if CI not available
                            errors_lower.append(0)
                            errors_upper.append(0)
                        has_sig.append(True)
                    else:
                        values.append(0)
                        errors_lower.append(0)
                        errors_upper.append(0)
                        has_sig.append(False)

                positions = x + i * bar_width
                if use_ci:
                    bars = ax.bar(positions, values, bar_width, color=colors[i],
                                 yerr=[errors_lower, errors_upper], capsize=2,
                                 error_kw={'elinewidth': 1, 'capthick': 1})
                else:
                    bars = ax.bar(positions, values, bar_width, color=colors[i])

                for bar, is_sig, err_up in zip(bars, has_sig, errors_upper):
                    if is_sig:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err_up + 0.01,
                               '*', ha='center', va='bottom', fontsize=21, fontweight='bold')

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
                # Add model name as left-aligned title at top (outside plot area)
                ax.text(-0.12, 1.05, source.upper(), transform=ax.transAxes,
                       fontsize=24, fontweight='bold', va='bottom', ha='left')

            # η²p reference lines (Cohen's guidelines)
            ax.set_ylim(0, 0.85)
            ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)  # small
            ax.axhline(y=0.06, color='gray', linestyle='-.', alpha=0.5, linewidth=0.5)  # medium
            ax.axhline(y=0.14, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)   # large

    # Legend
    handles = [plt.Rectangle((0,0), 1, 1, color=c) for c in colors]
    labels = [l for _, l in contrasts]
    if mode == 'layer_and_shared_individual':
        ncol = 7
    elif mode in ['global', 'shared', 'global_withshared', 'layer_and_shared']:
        ncol = 5
    else:
        ncol = 4
    fig.legend(handles, labels, loc='upper center', ncol=ncol, fontsize=21, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    suffix = f'_{mode}' if mode != 'layer' else ''
    output_path = output_dir / f'roi_grid_barplot{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Create grid barplot for ROI analysis')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--mode', type=str, default='layer',
                       choices=['layer', 'global', 'shared', 'withshared', 'global_withshared',
                                'withshared_withglobal', 'layer_and_shared', 'layer_and_shared_individual'],
                       help='Contrast mode')
    parser.add_argument('--all', action='store_true', help='Generate all versions')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    svc_dir = root / 'results' / 'dnn_analysis' / 'roi_analysis' / 'hierarchical_svc'
    output_dir = Path(args.output) if args.output else root / 'paper' / 'image'
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_df = pd.read_csv(svc_dir / 'clip_hierarchical_svc.csv')
    convnext_df = pd.read_csv(svc_dir / 'convnext_hierarchical_svc.csv')

    print(f"CLIP: {len(clip_df)} rows, {clip_df['significant'].sum()} significant")
    print(f"ConvNeXt: {len(convnext_df)} rows, {convnext_df['significant'].sum()} significant")

    # Determine which versions to generate
    if args.all:
        modes = ['layer', 'global_withshared', 'layer_and_shared', 'layer_and_shared_individual', 'withshared_withglobal']
    else:
        modes = [args.mode]

    for mode in modes:
        print(f"\nGenerating {mode} version...")
        create_grid_barplot(clip_df, convnext_df, output_dir, mode)

    print("\nDone!")


if __name__ == '__main__':
    main()
