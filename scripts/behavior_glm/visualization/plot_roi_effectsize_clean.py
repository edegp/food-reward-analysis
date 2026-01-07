"""Clean ROI Effect Size Barplot for Behavior GLM
- White background
- No effect size reference lines
- No title
- Japanese labels
- Error bars (95% CI)
- Beta-based effect size calculation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project root (3 levels up from this script: scripts/behavior_glm/visualization/)
ROOT = Path(__file__).resolve().parents[3]

# Set font for Japanese text
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

# Load second-level beta results
csv_path = ROOT / 'results' / 'roi_analysis' / 'roi_secondlevel_beta.csv'
df = pd.read_csv(csv_path)

roi_names_jp = df['ROI'].tolist()
mean_beta = df['mean_beta'].values
se_beta = df['se_beta'].values

# Use first-level p-values for significance coloring
csv_firstlevel = ROOT / 'results' / 'roi_analysis' / 'roi_effectsize_beta.csv'
df_first = pd.read_csv(csv_firstlevel)
p_values = df_first['p_value'].values

# Create figure
fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
ax.set_facecolor('white')

# Colors - dark orange for significant (p<0.05), light orange for non-significant
colors = ['#E85D04' if p < 0.05 else '#FFBA08' for p in p_values]

# Bar plot with error bars (SD)
bars = ax.bar(range(len(mean_beta)), mean_beta,
              color=colors, edgecolor='none', width=0.7)

# Error bars (SE)
ax.errorbar(range(len(mean_beta)), mean_beta, yerr=se_beta,
            fmt='none', ecolor='black', capsize=4, capthick=1.5, elinewidth=1.5)

# Zero line
ax.axhline(y=0, color='black', linewidth=0.8)

# Labels (1.5x original size)
ax.set_xticks(range(len(roi_names_jp)))
ax.set_xticklabels(roi_names_jp, rotation=30, ha='right', fontsize=17)
ax.set_ylabel('効果量 (β)', fontsize=20)
ax.tick_params(axis='y', labelsize=15)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

# Y-axis limits (adjusted for SE scale)
ax.set_ylim(-0.2, 1.0)

# Add value labels on bars (1.5x original size)
for i, (beta, se) in enumerate(zip(mean_beta, se_beta)):
    y_pos = beta + se + 0.03 if beta >= 0 else beta - se - 0.05
    ax.text(i, y_pos, f'{beta:.2f}', ha='center', va='bottom' if beta >= 0 else 'top',
            fontsize=14, fontweight='bold')

plt.tight_layout()

# Save figure
output_path = ROOT / 'results' / 'roi_analysis' / 'roi_effectsize_rgb_nutri_ImagexValue.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')

plt.close()
