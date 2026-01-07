#!/usr/bin/env python3
"""
ROI-based RSA with Double-Centering (equivalent to CKA)
- All layers from each model
- Harvard-Oxford ROIs
- Leave-One-Out Noise Ceiling
- 3 model comparison (CLIP, ImageNet, Food)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from nilearn.image import resample_to_img
import json
from collections import Counter
import pandas as pd
from tqdm import tqdm

ROOT = Path("/Users/yuhiaoki/dev/hit/food-brain")
LSS_BASE = Path("/Volumes/Extreme Pro/hit/food-brain/results/first_level_analysis")
ROI_DIR = ROOT / "rois" / "HarvardOxford"
OUTPUT_DIR = Path("/Volumes/Extreme Pro/hit/food-brain/results/rsa_analysis/roi_centered")

# All Harvard-Oxford ROIs
ALL_ROIS = [
    # Visual
    "V1_bi", "EarlyVisual_bi", "LOC_bi", "Fusiform_bi", "IT_bi",
    # Parietal
    "IPL_bi", "SPL_bi", "AngularGyrus_bi", "Precuneus_bi",
    # Temporal
    "STG_bi", "MTG_bi", "TemporalPole_bi", "PHC_bi",
    # Frontal
    "IFG_bi", "DLPFC_bi", "VLPFC_bi", "OFC_bi", "FrontalPole_bi", "Broca_L", "vmPFC_bi",
    # Cingulate
    "ACC_bi", "PCC_bi",
    # Insula
    "Insula_bi",
    # Subcortical
    "Hippocampus_bi", "Amygdala_bi", "Caudate_bi", "Putamen_bi",
    "NAcc_bi", "Thalamus_bi", "Pallidum_bi", "Striatum_bi",
]


def double_center(rdm_vec, n):
    """
    Double-center a vectorized RDM (upper triangle).
    This makes Pearson correlation equivalent to CKA.
    """
    # Convert to square matrix
    rdm = squareform(rdm_vec)

    # Double centering: rdm - row_mean - col_mean + grand_mean
    row_mean = rdm.mean(axis=1, keepdims=True)
    col_mean = rdm.mean(axis=0, keepdims=True)
    grand_mean = rdm.mean()

    centered = rdm - row_mean - col_mean + grand_mean

    # Return upper triangle
    return squareform(centered, checks=False)


def load_roi_mask(roi_name, reference_img):
    """Load ROI mask and resample to reference space"""
    mask_path = ROI_DIR / f"{roi_name}_mask.nii"
    if not mask_path.exists():
        print(f"  Warning: {mask_path} not found")
        return None

    mask_img = nib.load(mask_path)
    resampled = resample_to_img(mask_img, reference_img, interpolation='nearest')
    return resampled.get_fdata() > 0.5


def prepare_dnn_rdms_per_layer(model_file, selected_images):
    """Load DNN RDMs for each layer separately"""
    dnn_data = np.load(model_file, allow_pickle=True)
    layer_names = [k for k in dnn_data.keys() if k not in ['image_ids', 'layer_names']]

    dnn_img_ids = [str(x) for x in dnn_data["image_ids"]]
    img_indices = [dnn_img_ids.index(img_id.zfill(4)) for img_id in selected_images
                   if img_id.zfill(4) in dnn_img_ids]

    n_images = len(img_indices)

    # Return RDM for each layer (already centered)
    layer_rdms = {}
    layer_rdms_centered = {}
    for layer in layer_names:
        rdm = dnn_data[layer][np.ix_(img_indices, img_indices)]
        rdm_vec = squareform(rdm, checks=False)
        layer_rdms[layer] = rdm_vec
        layer_rdms_centered[layer] = double_center(rdm_vec, n_images)

    return layer_rdms, layer_rdms_centered, layer_names, n_images


def compute_rdm(patterns):
    """Compute RDM from pattern matrix (n_stimuli x n_voxels)"""
    return pdist(patterns, metric='correlation')


def compute_nc_loo_centered(subject_rdms, n_images):
    """
    Compute Leave-One-Out Noise Ceiling with centered RDMs

    Returns: (lower_nc, upper_nc, correlations_lower, correlations_upper)
    """
    n_subjects = len(subject_rdms)

    # Center all subject RDMs
    subject_rdms_centered = [double_center(rdm, n_images) for rdm in subject_rdms]

    # Mean centered RDM across all subjects
    mean_rdm_all = np.mean(np.vstack(subject_rdms_centered), axis=0)

    correlations_lower = []
    correlations_upper = []

    for i in range(n_subjects):
        subj_rdm = subject_rdms_centered[i]

        # Lower bound: correlate with mean of OTHER subjects
        other_rdms = [subject_rdms_centered[j] for j in range(n_subjects) if j != i]
        mean_rdm_others = np.mean(np.vstack(other_rdms), axis=0)
        r_lower, _ = spearmanr(subj_rdm, mean_rdm_others)
        correlations_lower.append(r_lower if not np.isnan(r_lower) else 0)

        # Upper bound: correlate with mean of ALL subjects (including self)
        r_upper, _ = spearmanr(subj_rdm, mean_rdm_all)
        correlations_upper.append(r_upper if not np.isnan(r_upper) else 0)

    lower_nc = np.mean(correlations_lower)
    upper_nc = np.mean(correlations_upper)

    return lower_nc, upper_nc, correlations_lower, correlations_upper, subject_rdms_centered


def main():
    print("=" * 60)
    print("ROI-based RSA with Double-Centering")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "CLIP": ROOT / "data_images" / "dnn_rdms" / "clip_rdms.npz",
        "ImageNet": ROOT / "data_images" / "dnn_rdms" / "convnext_pretrained_rdms.npz",
        "Food": ROOT / "data_images" / "dnn_rdms" / "convnext_rdms.npz"
    }

    # Collect subject directories
    print("\n[1/5] Collecting subject data...")
    subj_dirs = {}
    for subj_dir in sorted(LSS_BASE.iterdir()):
        if subj_dir.is_dir() and subj_dir.name.startswith("sub-") and "sub-sub" not in subj_dir.name:
            lss_dirs = sorted(subj_dir.glob("glm_model/lss_glm/2*"))
            if lss_dirs:
                subj_dirs[subj_dir.name] = lss_dirs[-1]

    subjects = sorted(subj_dirs.keys())
    n_subj = len(subjects)
    print(f"  Subjects: {n_subj}")

    # Get common images
    dnn_data = np.load(models["CLIP"], allow_pickle=True)
    dnn_image_ids = [str(img_id) for img_id in dnn_data["image_ids"] if int(img_id) <= 568]

    img_counter = Counter()
    subj_beta_info = {}
    for subj in subjects:
        beta_info_file = subj_dirs[subj] / "beta_values" / "beta_info.csv"
        if beta_info_file.exists():
            beta_info = pd.read_csv(beta_info_file)
            beta_info["image_id_str"] = beta_info["image_id"].apply(lambda x: f"{int(x):04d}")
            subj_beta_info[subj] = dict(zip(beta_info["image_id_str"], beta_info["beta_index"]))
            for img_id in dnn_image_ids:
                if img_id.zfill(4) in subj_beta_info[subj]:
                    img_counter[img_id] += 1

    # Use images present in at least 25 subjects
    selected_images = [img_id for img_id in dnn_image_ids if img_counter[img_id] >= 25]
    n_img = len(selected_images)
    print(f"  Images: {n_img}")

    # Get reference image for ROI resampling
    print("\n[2/5] Loading reference image...")
    ref_subj = subjects[0]
    ref_beta_idx = list(subj_beta_info[ref_subj].values())[0]
    ref_beta_file = subj_dirs[ref_subj] / f"beta_{ref_beta_idx:04d}.nii"
    ref_img = nib.load(ref_beta_file)
    print(f"  Reference shape: {ref_img.shape}")

    # Load ROI masks
    print("\n[3/5] Loading ROI masks...")
    roi_masks = {}
    for roi_name in ALL_ROIS:
        mask = load_roi_mask(roi_name, ref_img)
        if mask is not None:
            n_voxels = np.sum(mask)
            print(f"  {roi_name}: {int(n_voxels)} voxels")
            roi_masks[roi_name] = mask

    # Load betas for each subject and ROI
    print("\n[4/5] Loading betas and computing subject RDMs...")

    # Store per-subject RDMs for each ROI
    subject_rdms_per_roi = {roi: [] for roi in roi_masks.keys()}

    # First pass: find available images per subject
    print("  Finding available images per subject...")
    subj_available_images = {}
    for subj in subjects:
        if subj not in subj_beta_info:
            continue
        subj_images = set()
        for img_id in selected_images:
            img_id_str = img_id.zfill(4)
            if img_id_str in subj_beta_info[subj]:
                beta_file = subj_dirs[subj] / f"beta_{subj_beta_info[subj][img_id_str]:04d}.nii"
                if beta_file.exists():
                    subj_images.add(img_id)
        subj_available_images[subj] = subj_images

    # Find optimal subset of subjects that maximizes common images
    sorted_subjects = sorted(subj_available_images.items(), key=lambda x: len(x[1]), reverse=True)

    best_score = 0
    best_n_subj = 0
    best_common = set()
    best_subjects = []

    for n in range(10, len(sorted_subjects) + 1):
        test_subjects = [s for s, _ in sorted_subjects[:n]]
        test_common = set(selected_images)
        for subj in test_subjects:
            test_common = test_common & subj_available_images[subj]
        score = n * len(test_common)
        if score > best_score:
            best_score = score
            best_n_subj = n
            best_common = test_common
            best_subjects = test_subjects

    print(f"  Optimal configuration: {best_n_subj} subjects x {len(best_common)} images")

    common_images = sorted(list(best_common))
    subjects = best_subjects
    n_images = len(common_images)

    # Second pass: load betas for common images only
    valid_subjects = []
    for si, subj in enumerate(tqdm(subjects, desc="Subjects")):
        if subj not in subj_beta_info:
            continue

        betas = []
        for img_id in common_images:
            img_id_str = img_id.zfill(4)
            beta_file = subj_dirs[subj] / f"beta_{subj_beta_info[subj][img_id_str]:04d}.nii"
            beta_data = nib.load(beta_file).get_fdata()
            betas.append(beta_data)

        if len(betas) < 10:
            continue

        betas = np.array(betas)
        valid_subjects.append(subj)

        # Compute RDM for each ROI
        for roi_name, mask in roi_masks.items():
            patterns = betas[:, mask]
            rdm = compute_rdm(patterns)
            subject_rdms_per_roi[roi_name].append(rdm)

    print(f"  Valid subjects: {len(valid_subjects)}")

    # Pre-load all model RDMs per layer (with centering)
    print("\n[5/5] Loading model RDMs and computing correlations...")
    model_layer_rdms = {}
    model_layer_rdms_centered = {}
    model_layer_names = {}
    for model_name, model_file in models.items():
        layer_rdms, layer_rdms_centered, layer_names, _ = prepare_dnn_rdms_per_layer(model_file, common_images)
        model_layer_rdms[model_name] = layer_rdms
        model_layer_rdms_centered[model_name] = layer_rdms_centered
        model_layer_names[model_name] = layer_names
        print(f"  {model_name}: {len(layer_names)} layers")

    results = {}

    for roi_name in roi_masks.keys():
        print(f"\n  {roi_name}:")
        subject_rdms = subject_rdms_per_roi[roi_name]

        if len(subject_rdms) < 5:
            print(f"    Skipping (only {len(subject_rdms)} subjects)")
            continue

        # Compute NC with centered RDMs
        lower_nc, upper_nc, corrs_lower, corrs_upper, subject_rdms_centered = compute_nc_loo_centered(
            subject_rdms, n_images
        )
        print(f"    NC (centered): Lower={lower_nc:.3f}, Upper={upper_nc:.3f}")

        # Mean centered brain RDM
        mean_brain_rdm_centered = np.mean(np.vstack(subject_rdms_centered), axis=0)

        roi_results = {
            "n_subjects": len(subject_rdms),
            "n_voxels": int(np.sum(roi_masks[roi_name])),
            "n_images": n_images,
            "nc_lower": float(lower_nc),
            "nc_upper": float(upper_nc),
            "nc_lower_std": float(np.std(corrs_lower)),
            "nc_upper_std": float(np.std(corrs_upper)),
            "models": {}
        }

        for model_name in models.keys():
            layer_rdms_centered = model_layer_rdms_centered[model_name]
            layer_names = model_layer_names[model_name]

            layer_results = {}
            best_r = -1
            best_layer = None

            for layer_name in layer_names:
                dnn_rdm_centered = layer_rdms_centered[layer_name]

                # LOO cross-validation for model correlation
                loo_correlations = []
                n_subj = len(subject_rdms_centered)
                for i in range(n_subj):
                    other_rdms = [subject_rdms_centered[j] for j in range(n_subj) if j != i]
                    mean_rdm_others = np.mean(np.vstack(other_rdms), axis=0)
                    r_loo, _ = spearmanr(dnn_rdm_centered[:len(mean_rdm_others)], mean_rdm_others)
                    loo_correlations.append(r_loo if not np.isnan(r_loo) else 0)

                r_loo_mean = np.mean(loo_correlations)
                r_loo_std = np.std(loo_correlations)

                # Full correlation
                r_full, p = spearmanr(mean_brain_rdm_centered, dnn_rdm_centered[:len(mean_brain_rdm_centered)])
                r_full = r_full if not np.isnan(r_full) else 0

                # NC ratios
                nc_ratio_lower = (r_loo_mean / lower_nc * 100) if lower_nc > 0 else 0
                nc_ratio_upper = (r_full / upper_nc * 100) if upper_nc > 0 else 0

                layer_results[layer_name] = {
                    "r_loo": float(r_loo_mean),
                    "r_loo_std": float(r_loo_std),
                    "r_full": float(r_full),
                    "p": float(p),
                    "nc_ratio_lower": float(nc_ratio_lower),
                    "nc_ratio_upper": float(nc_ratio_upper)
                }

                if r_full > best_r:
                    best_r = r_full
                    best_layer = layer_name

            # Find best layer
            roi_results["models"][model_name] = {
                "layers": layer_results,
                "best_layer": best_layer,
                "best_r": float(best_r),
                "best_nc_ratio": float(best_r / upper_nc * 100) if upper_nc > 0 else 0
            }
            print(f"    {model_name}: best_r={best_r:.3f} ({best_layer}), NC%={best_r/upper_nc*100:.1f}%")

        results[roi_name] = roi_results

    # Save results
    output_file = OUTPUT_DIR / "roi_centered_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Create summary CSV
    summary_rows = []
    for roi_name, roi_data in results.items():
        row = {
            "ROI": roi_name.replace("_bi", ""),
            "n_subjects": roi_data["n_subjects"],
            "n_voxels": roi_data["n_voxels"],
            "n_images": roi_data["n_images"],
            "NC_Lower": roi_data["nc_lower"],
            "NC_Upper": roi_data["nc_upper"],
        }
        for model_name in ["CLIP", "ImageNet", "Food"]:
            if model_name in roi_data["models"]:
                model_data = roi_data["models"][model_name]
                row[f"{model_name}_best_r"] = model_data["best_r"]
                row[f"{model_name}_NC%"] = model_data["best_nc_ratio"]
                row[f"{model_name}_best_layer"] = model_data["best_layer"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / "roi_centered_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary: Best layer correlation per ROI (with centering)")
    print("=" * 80)
    print(f"{'ROI':<20} {'NC_U':<7} {'CLIP':<12} {'ImageNet':<12} {'Food':<12}")
    print("-" * 65)
    for roi_name, roi_data in sorted(results.items(), key=lambda x: -x[1]["nc_upper"]):
        clip_r = roi_data['models'].get('CLIP', {}).get('best_r', 0)
        inet_r = roi_data['models'].get('ImageNet', {}).get('best_r', 0)
        food_r = roi_data['models'].get('Food', {}).get('best_r', 0)
        clip_nc = roi_data['models'].get('CLIP', {}).get('best_nc_ratio', 0)
        inet_nc = roi_data['models'].get('ImageNet', {}).get('best_nc_ratio', 0)
        food_nc = roi_data['models'].get('Food', {}).get('best_nc_ratio', 0)
        print(f"{roi_name.replace('_bi', ''):<20} {roi_data['nc_upper']:.3f}   "
              f"{clip_r:.3f}({clip_nc:4.0f}%) "
              f"{inet_r:.3f}({inet_nc:4.0f}%) "
              f"{food_r:.3f}({food_nc:4.0f}%)")

    print("\nDone!")


if __name__ == '__main__':
    main()
