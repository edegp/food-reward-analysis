#!/usr/bin/env python3
"""
Extract beta information from LSS GLM SPM.mat files.
Creates beta_info.csv mapping image IDs to beta indices.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io
import warnings
warnings.filterwarnings('ignore')

def extract_beta_info_from_spm(spm_file: Path) -> pd.DataFrame:
    """Extract image ID to beta index mapping from SPM.mat."""
    print(f"  Loading {spm_file}")

    try:
        # Load SPM.mat
        spm_data = scipy.io.loadmat(str(spm_file), simplify_cells=True)
        spm = spm_data['SPM']

        image_ids = []
        beta_indices = []

        # Go through all sessions and regressors
        beta_idx = 1

        for sess_idx, sess in enumerate(spm['Sess']):
            # Process U (stimulus functions)
            if 'U' in sess and sess['U'] is not None:
                U_list = sess['U'] if isinstance(sess['U'], list) else [sess['U']]

                for reg in U_list:
                    reg_name = reg['name']
                    if isinstance(reg_name, list):
                        reg_name = reg_name[0]
                    if isinstance(reg_name, np.ndarray):
                        reg_name = str(reg_name)

                    # Check if this is an image regressor
                    if 'Image_' in reg_name:
                        # Extract image ID
                        img_id = reg_name.replace('Image_', '')
                        image_ids.append(img_id)
                        beta_indices.append(beta_idx)

                    beta_idx += 1

            # Skip confound regressors
            if 'C' in sess and sess['C'] is not None:
                if 'C' in sess['C']:
                    C_matrix = sess['C']['C']
                    if C_matrix is not None:
                        n_confounds = C_matrix.shape[1] if len(C_matrix.shape) > 1 else 1
                        beta_idx += n_confounds

        print(f"  Found {len(image_ids)} image regressors")

        # Create DataFrame
        beta_info = pd.DataFrame({
            'image_id': image_ids,
            'beta_index': beta_indices
        })

        return beta_info

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    root_dir = Path('/Users/yuhiaoki/dev/hit/food-brain')
    result_dir = root_dir / 'results' / 'first_level_analysis'

    # Subject list
    subjects = [f'{i:03d}' for i in range(1, 21)]

    print(f"Extracting beta info for {len(subjects)} subjects...")

    for sub_id in subjects:
        print(f"Processing subject {sub_id}...")

        # Find latest LSS GLM directory
        sub_glm_dir = result_dir / f'sub-{sub_id}' / 'glm_model' / 'lss_glm'
        if not sub_glm_dir.exists():
            print(f"  WARNING: LSS GLM directory not found")
            continue

        # Get latest timestamp directory
        timestamp_dirs = sorted([d for d in sub_glm_dir.iterdir() if d.is_dir()])
        if not timestamp_dirs:
            print(f"  WARNING: No LSS GLM results found")
            continue

        latest_dir = timestamp_dirs[-1]

        # Load SPM.mat
        spm_file = latest_dir / 'SPM.mat'
        if not spm_file.exists():
            print(f"  WARNING: SPM.mat not found")
            continue

        # Extract beta info
        beta_info = extract_beta_info_from_spm(spm_file)
        if beta_info is None:
            continue

        # Save to CSV
        output_dir = latest_dir / 'beta_values'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'beta_info.csv'
        beta_info.to_csv(output_file, index=False)

        print(f"  Saved: {output_file}")

    print("Done!")


if __name__ == '__main__':
    main()
