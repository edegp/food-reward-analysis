#!/usr/bin/env python3
"""
Create 3-Level Hierarchical PCs from existing orthogonalized PCs.

Hierarchy:
1. Global PCs: Common across ALL layers (from clip_global_common_pcs.csv)
2. Layer-Shared PCs: Shared between ADJACENT layer pairs
   - Initial-Middle shared
   - Middle-Late shared
   - Late-Final shared
3. Layer-Specific PCs: Unique to each layer (after removing Global + Shared)

Layer-Shared extraction:
  For each adjacent pair (e.g., Initial & Middle):
  - Concatenate their PCs horizontally
  - Apply PCA to find shared variation
  - This captures what's common between the two layers
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def orthogonalize(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Orthogonalize X with respect to basis using projection."""
    if basis.shape[1] == 0:
        return X
    proj = basis @ np.linalg.lstsq(basis, X, rcond=None)[0]
    return X - proj


def extract_shared_pcs(X1: np.ndarray, X2: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Extract shared PCs between two sets of features using CCA.

    CCA finds directions that maximize correlation between X1 and X2,
    which represents the shared information between the two layers.

    Returns the shared scores (average of both projections).
    """
    n_comp = min(n_components, X1.shape[1], X2.shape[1])
    cca = CCA(n_components=n_comp)
    X1_c, X2_c = cca.fit_transform(X1, X2)
    # Return average of both projections as the shared representation
    return (X1_c + X2_c) / 2


def create_3level_pcs(
    global_pcs_path: Path,
    layer_orth_pcs_path: Path,
    output_dir: Path,
    source: str,
    n_shared_per_pair: int = 2,
    specific_variance: float = 0.5,
):
    """
    Create 3-level hierarchical PCs from existing data.
    """
    print(f"Loading Global PCs from {global_pcs_path}")
    global_df = pd.read_csv(global_pcs_path)

    print(f"Loading Layer-Orth PCs from {layer_orth_pcs_path}")
    layer_df = pd.read_csv(layer_orth_pcs_path)

    # Ensure image_id is string for consistent matching
    global_df['image_id'] = global_df['image_id'].astype(str)
    layer_df['image_id'] = layer_df['image_id'].astype(str)

    # Get unique image IDs
    image_ids = global_df['image_id'].unique()
    print(f"Number of images: {len(image_ids)}")

    # ========== Level 1: Global PCs (already extracted) ==========
    global_wide = global_df[global_df['module_name'].str.contains('Global', case=False)].copy()
    pc_cols = [c for c in global_wide.columns if c.startswith('pc')]
    n_global = len(pc_cols)
    print(f"\n=== Level 1: Global PCs ===")
    print(f"Global PCs: {n_global} components")

    # Create image_id to Global PCs mapping
    global_pcs_dict = {}
    for _, row in global_wide.iterrows():
        img_id = row['image_id']
        global_pcs_dict[img_id] = [row[f'pc{i+1}'] for i in range(n_global)]

    # ========== Get Layer Groups ==========
    layer_groups = layer_df['module_name'].unique().tolist()
    layer_groups = [lg for lg in layer_groups if 'Global' not in lg and lg != 'global']
    # Sort to ensure order: Initial, Middle, Late, Final
    order = ['Initial', 'Middle', 'Late', 'Final']
    layer_groups = sorted(layer_groups, key=lambda x: next((i for i, o in enumerate(order) if o in x), 99))
    print(f"Layer groups (ordered): {layer_groups}")

    # Collect layer-orth PCs for all images
    layer_orth_data = {}
    for group in layer_groups:
        group_df = layer_df[layer_df['module_name'] == group].copy()
        pc_cols = [c for c in group_df.columns if c.startswith('pc')]
        n_pcs = len(pc_cols)

        group_data = {}
        for _, row in group_df.iterrows():
            img_id = row['image_id']
            group_data[img_id] = [row[f'pc{i+1}'] for i in range(n_pcs)]

        layer_orth_data[group] = {'n_pcs': n_pcs, 'data': group_data}
        print(f"  {group}: {n_pcs} PCs")

    # Get valid image IDs (present in all layers)
    valid_image_ids = []
    for img_id in image_ids:
        if img_id not in global_pcs_dict:
            continue
        valid = all(img_id in layer_orth_data[g]['data'] for g in layer_groups)
        if valid:
            valid_image_ids.append(img_id)
    print(f"Valid images: {len(valid_image_ids)}")

    # ========== Level 2: Layer-Shared PCs (between adjacent pairs) ==========
    print(f"\n=== Level 2: Layer-Shared PCs (CCA between adjacent pairs) ===")

    # Define adjacent pairs
    pairs = []
    for i in range(len(layer_groups) - 1):
        pairs.append((layer_groups[i], layer_groups[i + 1]))
    print(f"Adjacent pairs: {pairs}")

    shared_pcs_dict = {}  # pair_name -> {img_id: [pc values]}
    all_shared_pcs = []  # For combining into basis

    for g1, g2 in pairs:
        pair_name = f"{g1.split('.')[-1]}-{g2.split('.')[-1]}"

        # Get PCs for both groups
        X1 = np.array([layer_orth_data[g1]['data'][img_id] for img_id in valid_image_ids])
        X2 = np.array([layer_orth_data[g2]['data'][img_id] for img_id in valid_image_ids])

        # Extract shared PCs using CCA
        shared = extract_shared_pcs(X1, X2, n_components=n_shared_per_pair)
        n_shared = shared.shape[1]

        print(f"  {pair_name}: {n_shared} shared PCs")

        # Store results
        shared_pcs_dict[pair_name] = {}
        for i, img_id in enumerate(valid_image_ids):
            shared_pcs_dict[pair_name][img_id] = shared[i, :].tolist()

        all_shared_pcs.append(shared)

    # Combine all shared PCs
    all_shared_mat_raw = np.hstack(all_shared_pcs)
    n_total_shared = all_shared_mat_raw.shape[1]
    print(f"Total shared PCs (before orthogonalization): {n_total_shared}")

    # ========== Orthogonalize Shared PCs ==========
    # Step 1: Orthogonalize Shared PCs with respect to Global PCs
    global_mat = np.array([global_pcs_dict[img_id] for img_id in valid_image_ids])
    all_shared_mat_orth_global = orthogonalize(all_shared_mat_raw, global_mat)

    # Step 2: Apply QR decomposition to orthogonalize Shared PCs among themselves
    # QR decomposition: A = QR where Q is orthogonal
    Q, R = np.linalg.qr(all_shared_mat_orth_global)
    all_shared_mat = Q[:, :n_total_shared]  # Keep same number of components

    # Verify orthogonality
    corr_after = np.corrcoef(all_shared_mat.T)
    off_diag = corr_after[np.triu_indices_from(corr_after, k=1)]
    print(f"After orthogonalization: max|correlation| between Shared PCs = {np.max(np.abs(off_diag)):.6f}")

    # Verify orthogonality with Global
    corr_global_shared = np.corrcoef(global_mat.T, all_shared_mat.T)[:global_mat.shape[1], global_mat.shape[1]:]
    print(f"After orthogonalization: max|correlation| Global vs Shared = {np.max(np.abs(corr_global_shared)):.6f}")

    # Update shared_pcs_dict with orthogonalized values
    # Map back to pair structure (for saving)
    col_idx = 0
    for pair_idx, (g1, g2) in enumerate(pairs):
        pair_name = f"{g1.split('.')[-1]}-{g2.split('.')[-1]}"
        n_pcs = all_shared_pcs[pair_idx].shape[1]
        shared_pcs_dict[pair_name] = {}
        for i, img_id in enumerate(valid_image_ids):
            shared_pcs_dict[pair_name][img_id] = all_shared_mat[i, col_idx:col_idx+n_pcs].tolist()
        col_idx += n_pcs

    # ========== Level 3: Layer-Specific PCs ==========
    print(f"\n=== Level 3: Layer-Specific PCs (variance threshold: {specific_variance}) ===")

    # Combined basis: Global + all Shared
    combined_basis = np.hstack([global_mat, all_shared_mat])
    print(f"Combined basis (Global + Shared): {combined_basis.shape}")

    # First pass: Extract Layer-Specific PCs for each group
    specific_pcs_raw = {}
    specific_info = {}

    for group in layer_groups:
        group_short = group.split('.')[-1]

        # Get layer PCs
        X_group = np.array([layer_orth_data[group]['data'][img_id] for img_id in valid_image_ids])

        # Orthogonalize with respect to Global + Shared
        X_specific = orthogonalize(X_group, combined_basis)

        # Check remaining variance
        var_remaining = np.var(X_specific)
        if var_remaining < 1e-10:
            print(f"  {group_short}: No unique variance remaining, using 0 PCs")
            specific_pcs_raw[group] = np.zeros((len(valid_image_ids), 0))
            specific_info[group_short] = {'n_pcs': 0, 'explained_variance': 0.0}
            continue

        # Extract Layer-Specific PCs
        try:
            pca_specific = PCA(n_components=specific_variance, svd_solver='full')
            group_pcs = pca_specific.fit_transform(X_specific)
            n_specific = group_pcs.shape[1]
            exp_var = pca_specific.explained_variance_ratio_.sum()
        except Exception as e:
            print(f"  {group_short}: PCA failed ({e}), using 0 PCs")
            specific_pcs_raw[group] = np.zeros((len(valid_image_ids), 0))
            specific_info[group_short] = {'n_pcs': 0, 'explained_variance': 0.0}
            continue

        specific_pcs_raw[group] = group_pcs
        specific_info[group_short] = {'n_pcs': n_specific, 'explained_variance': exp_var}
        print(f"  {group_short}: {n_specific} PCs (before cross-layer orthogonalization)")

    # Second pass: Orthogonalize Layer-Specific PCs across all layers
    print("\n=== Orthogonalizing Layer-Specific PCs across layers ===")

    # Combine all Layer-Specific PCs
    all_specific_list = [specific_pcs_raw[g] for g in layer_groups if specific_pcs_raw[g].shape[1] > 0]
    if all_specific_list:
        all_specific_mat_raw = np.hstack(all_specific_list)

        # Orthogonalize with respect to Global + Shared first
        all_specific_orth = orthogonalize(all_specific_mat_raw, combined_basis)

        # QR decomposition for mutual orthogonality
        Q_specific, _ = np.linalg.qr(all_specific_orth)
        all_specific_mat = Q_specific[:, :all_specific_mat_raw.shape[1]]

        # Verify orthogonality
        corr_specific = np.corrcoef(all_specific_mat.T)
        off_diag_specific = corr_specific[np.triu_indices_from(corr_specific, k=1)]
        print(f"After orthogonalization: max|correlation| between Layer-Specific PCs = {np.max(np.abs(off_diag_specific)):.6f}")

        # Verify orthogonality with Global + Shared
        corr_basis_specific = np.corrcoef(combined_basis.T, all_specific_mat.T)[:combined_basis.shape[1], combined_basis.shape[1]:]
        print(f"After orthogonalization: max|correlation| (Global+Shared) vs Specific = {np.max(np.abs(corr_basis_specific)):.6f}")

        # Map back to layer structure
        col_idx = 0
        specific_pcs_dict = {}
        for group in layer_groups:
            group_short = group.split('.')[-1]
            n_pcs = specific_info[group_short]['n_pcs']
            if n_pcs > 0:
                specific_pcs_dict[group] = {}
                for i, img_id in enumerate(valid_image_ids):
                    specific_pcs_dict[group][img_id] = all_specific_mat[i, col_idx:col_idx+n_pcs].tolist()
                col_idx += n_pcs
                print(f"  {group_short}: {n_pcs} PCs (orthogonalized)")
            else:
                specific_pcs_dict[group] = {img_id: [] for img_id in valid_image_ids}
    else:
        specific_pcs_dict = {group: {img_id: [] for img_id in valid_image_ids} for group in layer_groups}

    # ========== Save Results ==========
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    for img_id in valid_image_ids:
        row = {'image_id': img_id}

        # Global PCs
        for i, val in enumerate(global_pcs_dict[img_id]):
            row[f'Global_pc{i+1}'] = val

        # Shared PCs (by pair)
        for pair_name, pair_data in shared_pcs_dict.items():
            for i, val in enumerate(pair_data[img_id]):
                row[f'Shared_{pair_name}_pc{i+1}'] = val

        # Specific PCs
        for group in layer_groups:
            group_short = group.split('.')[-1]
            if img_id in specific_pcs_dict[group]:
                for i, val in enumerate(specific_pcs_dict[group][img_id]):
                    row[f'{group_short}_pc{i+1}'] = val

        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)

    output_path = output_dir / f'{source}_3level_pcs.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Summary
    total_pcs = n_global + n_total_shared + sum(info['n_pcs'] for info in specific_info.values())
    print(f"\n=== Summary ===")
    print(f"Global PCs: {n_global}")
    print(f"Shared PCs: {n_total_shared} ({n_shared_per_pair} per pair × {len(pairs)} pairs)")
    for group_short, info in specific_info.items():
        print(f"{group_short}-Specific PCs: {info['n_pcs']}")
    print(f"Total PCs: {total_pcs}")

    # Verify orthogonality
    print("\n=== Verifying Orthogonality ===")

    # Global vs Shared
    corr = np.corrcoef(global_mat.T, all_shared_mat.T)[:n_global, n_global:]
    max_corr = np.max(np.abs(corr))
    print(f"Max |correlation| Global vs Shared: {max_corr:.4f}")

    # Shared pairs vs each other
    offset = 0
    for i, (g1, g2) in enumerate(pairs):
        pair_name = f"{g1.split('.')[-1]}-{g2.split('.')[-1]}"
        pair_pcs = all_shared_pcs[i]
        # Check vs Global
        corr_g = np.corrcoef(global_mat.T, pair_pcs.T)[:n_global, n_global:]
        print(f"  {pair_name} vs Global: max|r| = {np.max(np.abs(corr_g)):.4f}")

    # Save metadata
    metadata = {
        'source': source,
        'global': {'n_pcs': n_global},
        'shared': {
            'n_pcs_per_pair': n_shared_per_pair,
            'pairs': [f"{g1.split('.')[-1]}-{g2.split('.')[-1]}" for g1, g2 in pairs],
            'total_pcs': n_total_shared
        },
        'specific': specific_info,
        'total_pcs': total_pcs,
        'n_images': len(valid_image_ids)
    }

    meta_path = output_dir / f'{source}_3level_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Create 3-level hierarchical PCs')
    parser.add_argument('--source', type=str, required=True, choices=['clip', 'convnext'],
                        help='DNN source (clip or convnext)')
    parser.add_argument('--global-pcs', type=str,
                        help='Path to global common PCs CSV')
    parser.add_argument('--layer-orth-pcs', type=str,
                        help='Path to layer-orth-to-global PCs CSV')
    parser.add_argument('--output', type=str, default='data_images/dnn_pmods/3level',
                        help='Output directory')
    parser.add_argument('--n-shared-per-pair', type=int, default=2,
                        help='Number of shared PCs per adjacent pair')
    parser.add_argument('--specific-variance', type=float, default=0.5,
                        help='Variance threshold for Layer-Specific PCs')

    args = parser.parse_args()

    if args.global_pcs is None:
        args.global_pcs = f'data_images/dnn_pmods/orthogonalized/{args.source}_global_common_pcs.csv'
    if args.layer_orth_pcs is None:
        args.layer_orth_pcs = f'data_images/dnn_pmods/orthogonalized/{args.source}_layer_orth_to_global_pcs.csv'

    create_3level_pcs(
        global_pcs_path=Path(args.global_pcs),
        layer_orth_pcs_path=Path(args.layer_orth_pcs),
        output_dir=Path(args.output),
        source=args.source,
        n_shared_per_pair=args.n_shared_per_pair,
        specific_variance=args.specific_variance,
    )


if __name__ == '__main__':
    main()
