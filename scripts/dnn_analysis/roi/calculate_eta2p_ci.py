#!/usr/bin/env python3
"""
Calculate η²p (partial eta-squared) with 95% CI from SVC results
Uses noncentral F distribution for confidence intervals
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import brentq


def eta2p_from_f(f_value: float, df1: float, df2: float) -> float:
    """Calculate partial eta-squared from F-value"""
    return (df1 * f_value) / (df1 * f_value + df2)


def ncp_from_eta2p(eta2p: float, df2: float) -> float:
    """Calculate noncentrality parameter from η²p"""
    if eta2p >= 1:
        return np.inf
    if eta2p <= 0:
        return 0
    return (eta2p * df2) / (1 - eta2p)


def eta2p_from_ncp(ncp: float, df2: float) -> float:
    """Calculate η²p from noncentrality parameter"""
    return ncp / (ncp + df2)


def ci_eta2p(f_value: float, df1: float, df2: float, alpha: float = 0.05) -> tuple:
    """
    Calculate confidence interval for partial eta-squared
    using noncentral F distribution

    Based on Smithson (2001) and Steiger (2004)

    The CI is found by inverting the noncentral F distribution:
    - Lower bound: ncp where observed F is at the (1-α/2) quantile
    - Upper bound: ncp where observed F is at the (α/2) quantile
    """
    eta2p = eta2p_from_f(f_value, df1, df2)

    # For lower CI bound:
    # Find ncp such that F_obs is the (1-α/2) quantile of ncF(df1, df2, ncp)
    # i.e., P(F < F_obs | ncp) = 1 - α/2
    # i.e., P(F > F_obs | ncp) = α/2

    def lower_func(ncp):
        return stats.ncf.cdf(f_value, df1, df2, ncp) - (1 - alpha/2)

    def upper_func(ncp):
        return stats.ncf.cdf(f_value, df1, df2, ncp) - alpha/2

    # Find lower bound of ncp
    # Check if lower bound exists (if F is significant)
    # If CDF at ncp=0 is < 0.975, F is too small to have a positive lower bound
    if stats.ncf.cdf(f_value, df1, df2, 0) < (1 - alpha/2):
        ncp_lower = 0
    else:
        # F is significant, search for lower bound
        # Use wide search range
        ncp_lower = brentq(lower_func, 0, 10000)

    # Find upper bound of ncp
    # Upper bound: find ncp where CDF at F_obs = alpha/2
    # This means F_obs is unusually low for this ncp
    # Use adaptive search range based on observed values
    ncp_obs = ncp_from_eta2p(eta2p, df2) if eta2p > 0 else 1

    # For numerical stability, limit search range based on when ncf.cdf becomes unstable
    # Typically becomes NaN around ncp > 2000 for large df
    search_max = min(2000, max(ncp_obs * 20, 500))
    ncp_upper = brentq(upper_func, 0, search_max)

    # Convert ncp bounds to η²p bounds
    ci_lower = eta2p_from_ncp(ncp_lower, df2)
    ci_upper = eta2p_from_ncp(ncp_upper, df2)

    return eta2p, ci_lower, ci_upper


def main():
    parser = argparse.ArgumentParser(description='Calculate η²p with 95% CI')
    parser.add_argument('--input', type=str, default=None, help='Input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    svc_dir = root / 'results' / 'dnn_analysis' / 'roi_analysis' / 'hierarchical_svc'

    # Process both sources
    for source in ['clip', 'convnext']:
        input_file = Path(args.input) if args.input else svc_dir / f'{source}_hierarchical_svc.csv'
        output_file = Path(args.output) if args.output else svc_dir / f'{source}_hierarchical_svc.csv'

        if not input_file.exists():
            print(f"File not found: {input_file}")
            continue

        print(f"\nProcessing {source}...")
        df = pd.read_csv(input_file)

        # Add CI columns if not present
        if 'eta2p_ci_lower' not in df.columns:
            df['eta2p_ci_lower'] = np.nan
            df['eta2p_ci_upper'] = np.nan
            df['eta2p_se'] = np.nan

        # Calculate CI for each row
        for idx, row in df.iterrows():
            f_value = row['peak_F']

            # Read df values from CSV (saved by MATLAB)
            if 'df1' in row and 'df2' in row and pd.notna(row['df1']) and pd.notna(row['df2']):
                df1 = int(row['df1'])
                df2 = int(row['df2'])
            else:
                # Fallback to hardcoded values if not in CSV
                contrast = row['contrast']
                df_table_clip = {
                    'Initial_only': (2, 60),
                    'Initial_withShared': (4, 120),
                    'Middle_only': (3, 90),
                    'Middle_withShared': (7, 210),
                    'Late_only': (2, 60),
                    'Late_withShared': (6, 180),
                    'Final_only': (2, 60),
                    'Final_withShared': (4, 120),
                    'Global_F': (28, 840),
                    'Shared_F': (6, 180),
                }
                df_table_convnext = df_table_clip.copy()
                df_table_convnext['Global_F'] = (22, 660)
                df_table = df_table_clip if source == 'clip' else df_table_convnext
                if contrast in df_table:
                    df1, df2 = df_table[contrast]
                else:
                    df1, df2 = 2, 60

            if pd.notna(f_value) and f_value > 0:
                try:
                    eta2p, ci_lower, ci_upper = ci_eta2p(f_value, df1, df2)
                    df.at[idx, 'eta2p_ci_lower'] = ci_lower
                    df.at[idx, 'eta2p_ci_upper'] = ci_upper
                    # SE approximation from CI width
                    df.at[idx, 'eta2p_se'] = (ci_upper - ci_lower) / (2 * 1.96)

                    if row['significant']:
                        print(f"  {row['roi']}/{row['contrast']}: η²p={eta2p:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
                except Exception as e:
                    # Set CI to NaN when calculation fails (numerical instability for large F/df)
                    df.at[idx, 'eta2p_ci_lower'] = np.nan
                    df.at[idx, 'eta2p_ci_upper'] = np.nan
                    df.at[idx, 'eta2p_se'] = np.nan
                    if row['significant']:
                        print(f"  {row['roi']}/{row['contrast']}: η²p={eta2p_from_f(f_value, df1, df2):.3f} [CI計算失敗: {e}]")

        # Save updated CSV
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    print("\nDone!")


if __name__ == '__main__':
    main()
