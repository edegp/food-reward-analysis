#!/usr/bin/env python3
"""
Create ROI masks from Harvard-Oxford atlas for food-brain DNN analysis

Uses Harvard-Oxford cortical and subcortical probabilistic atlases
Threshold: 25% probability for inclusion
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from nilearn import datasets, image

# ROI definitions using Harvard-Oxford labels
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases

CORTICAL_ROIS = {
    # Visual cortex
    'V1_bi': ['Occipital Pole'],
    'LOC_bi': ['Lateral Occipital Cortex, inferior division',
               'Lateral Occipital Cortex, superior division'],
    'EarlyVisual_bi': ['Occipital Pole', 'Intracalcarine Cortex',
                       'Cuneal Cortex', 'Supracalcarine Cortex',
                       'Lingual Gyrus', 'Occipital Fusiform Gyrus'],

    # IT / Ventral stream
    'Fusiform_bi': ['Temporal Occipital Fusiform Cortex',
                    'Temporal Fusiform Cortex, anterior division',
                    'Temporal Fusiform Cortex, posterior division'],
    'IT_bi': ['Inferior Temporal Gyrus, temporooccipital part',
              'Inferior Temporal Gyrus, posterior division',
              'Inferior Temporal Gyrus, anterior division'],

    # Parietal
    'SPL_bi': ['Superior Parietal Lobule'],
    'IPL_bi': ['Angular Gyrus', 'Supramarginal Gyrus, anterior division',
               'Supramarginal Gyrus, posterior division'],
    'AngularGyrus_bi': ['Angular Gyrus'],
    'AngularGyrus_L': ['Angular Gyrus'],
    'AngularGyrus_R': ['Angular Gyrus'],
    'Precuneus_bi': ['Precuneous Cortex'],

    # Temporal / Language
    'STG_bi': ['Superior Temporal Gyrus, posterior division',
               'Superior Temporal Gyrus, anterior division'],
    'MTG_bi': ['Middle Temporal Gyrus, posterior division',
               'Middle Temporal Gyrus, anterior division',
               'Middle Temporal Gyrus, temporooccipital part'],
    'TemporalPole_bi': ['Temporal Pole'],

    # Frontal
    'IFG_bi': ['Inferior Frontal Gyrus, pars opercularis',
               'Inferior Frontal Gyrus, pars triangularis'],
    'Broca_L': ['Inferior Frontal Gyrus, pars opercularis',
                'Inferior Frontal Gyrus, pars triangularis'],  # Will filter for left only
    'DLPFC_bi': ['Middle Frontal Gyrus'],  # Dorsolateral prefrontal cortex
    'VLPFC_bi': ['Inferior Frontal Gyrus, pars opercularis',
                 'Inferior Frontal Gyrus, pars triangularis'],  # Ventrolateral prefrontal cortex
    'OFC_bi': ['Frontal Orbital Cortex'],
    'FrontalPole_bi': ['Frontal Pole'],

    # Cingulate
    'ACC_bi': ['Cingulate Gyrus, anterior division'],
    'PCC_bi': ['Cingulate Gyrus, posterior division'],

    # Insula
    'Insula_bi': ['Insular Cortex'],

    # Parahippocampal
    'PHC_bi': ['Parahippocampal Gyrus, anterior division',
               'Parahippocampal Gyrus, posterior division'],
}

SUBCORTICAL_ROIS = {
    'Hippocampus_bi': ['Left Hippocampus', 'Right Hippocampus'],
    'Hippocampus_L': ['Left Hippocampus'],
    'Hippocampus_R': ['Right Hippocampus'],
    'Amygdala_bi': ['Left Amygdala', 'Right Amygdala'],
    'Caudate_bi': ['Left Caudate', 'Right Caudate'],
    'Putamen_bi': ['Left Putamen', 'Right Putamen'],
    'NAcc_bi': ['Left Accumbens', 'Right Accumbens'],
    'Thalamus_bi': ['Left Thalamus', 'Right Thalamus'],
    'Pallidum_bi': ['Left Pallidum', 'Right Pallidum'],
}


def create_roi_from_atlas(atlas_img, atlas_labels, region_names, threshold=25):
    """Create a binary mask from probabilistic atlas regions"""
    atlas_data = atlas_img.get_fdata()
    mask_data = np.zeros(atlas_data.shape[:3])

    # Note: atlas_labels[0] is usually "Background" which we skip
    # The 4D atlas has shape (x, y, z, n_regions) where n_regions = len(labels) - 1
    for region_name in region_names:
        # Find matching labels (partial match)
        for i, label in enumerate(atlas_labels):
            if label.lower() == 'background':
                continue
            if region_name.lower() in label.lower() or label.lower() in region_name.lower():
                # For 4D probabilistic atlas
                if len(atlas_data.shape) == 4:
                    # Index is (label position - 1) because background is excluded from data
                    atlas_idx = i - 1 if atlas_labels[0].lower() == 'background' else i
                    if atlas_idx >= 0 and atlas_idx < atlas_data.shape[3]:
                        region_prob = atlas_data[:, :, :, atlas_idx]
                        mask_data[region_prob > threshold] = 1
                # For 3D labeled atlas
                else:
                    mask_data[atlas_data == i] = 1

    return mask_data


def create_roi_masks(output_dir: Path, threshold: int = 25):
    """Create ROI masks from Harvard-Oxford atlases"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch Harvard-Oxford atlases
    print("Loading Harvard-Oxford cortical atlas...")
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
    # Handle both old and new nilearn API
    if isinstance(ho_cort.maps, str):
        cort_img = nib.load(ho_cort.maps)
    else:
        cort_img = ho_cort.maps  # Already a Nifti1Image
    cort_labels = ho_cort.labels

    print("Loading Harvard-Oxford subcortical atlas...")
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm')
    if isinstance(ho_sub.maps, str):
        sub_img = nib.load(ho_sub.maps)
    else:
        sub_img = ho_sub.maps
    sub_labels = ho_sub.labels

    print(f"Cortical atlas shape: {cort_img.shape}")
    print(f"Subcortical atlas shape: {sub_img.shape}")

    # Reference for output
    ref_affine = cort_img.affine
    ref_header = cort_img.header

    created_rois = []

    # Create cortical ROIs
    print("\nCreating cortical ROIs...")
    for roi_name, region_names in CORTICAL_ROIS.items():
        print(f"  {roi_name}...")
        mask_data = create_roi_from_atlas(cort_img, cort_labels, region_names, threshold)

        # For left-lateralized ROIs, mask out right hemisphere
        # Harvard-Oxford uses RAS orientation: x < center = Right, x >= center = Left
        if roi_name.endswith('_L'):
            center_x = mask_data.shape[0] // 2
            mask_data[:center_x, :, :] = 0  # Remove right hemisphere (low x)
        # For right-lateralized ROIs, mask out left hemisphere
        elif roi_name.endswith('_R'):
            center_x = mask_data.shape[0] // 2
            mask_data[center_x:, :, :] = 0  # Remove left hemisphere (high x)

        n_voxels = int(np.sum(mask_data > 0))
        if n_voxels > 0:
            mask_img = nib.Nifti1Image(mask_data.astype(np.float32), ref_affine, ref_header)
            output_path = output_dir / f'{roi_name}_mask.nii'
            nib.save(mask_img, output_path)
            print(f"    Saved: {output_path} ({n_voxels} voxels)")
            created_rois.append((roi_name, n_voxels, region_names))
        else:
            print(f"    Warning: {roi_name} has 0 voxels!")

    # Create subcortical ROIs
    print("\nCreating subcortical ROIs...")
    for roi_name, region_names in SUBCORTICAL_ROIS.items():
        print(f"  {roi_name}...")
        mask_data = create_roi_from_atlas(sub_img, sub_labels, region_names, threshold)

        n_voxels = int(np.sum(mask_data > 0))
        if n_voxels > 0:
            mask_img = nib.Nifti1Image(mask_data.astype(np.float32), ref_affine, ref_header)
            output_path = output_dir / f'{roi_name}_mask.nii'
            nib.save(mask_img, output_path)
            print(f"    Saved: {output_path} ({n_voxels} voxels)")
            created_rois.append((roi_name, n_voxels, region_names))
        else:
            print(f"    Warning: {roi_name} has 0 voxels!")

    # Create combined ROIs
    print("\nCreating combined ROIs...")

    # vmPFC = medial OFC + subcallosal cortex (approximation)
    vmPFC_regions = ['Subcallosal Cortex', 'Frontal Medial Cortex', 'Paracingulate Gyrus']
    mask_data = create_roi_from_atlas(cort_img, cort_labels, vmPFC_regions, threshold)
    n_voxels = int(np.sum(mask_data > 0))
    if n_voxels > 0:
        mask_img = nib.Nifti1Image(mask_data.astype(np.float32), ref_affine, ref_header)
        output_path = output_dir / 'vmPFC_bi_mask.nii'
        nib.save(mask_img, output_path)
        print(f"    Saved: {output_path} ({n_voxels} voxels)")
        created_rois.append(('vmPFC_bi', n_voxels, vmPFC_regions))

    # Striatum = Caudate + Putamen
    caud_data = create_roi_from_atlas(sub_img, sub_labels, ['Left Caudate', 'Right Caudate'], threshold)
    put_data = create_roi_from_atlas(sub_img, sub_labels, ['Left Putamen', 'Right Putamen'], threshold)
    striatum_data = np.maximum(caud_data, put_data)
    n_voxels = int(np.sum(striatum_data > 0))
    if n_voxels > 0:
        mask_img = nib.Nifti1Image(striatum_data.astype(np.float32), ref_affine, ref_header)
        output_path = output_dir / 'Striatum_bi_mask.nii'
        nib.save(mask_img, output_path)
        print(f"    Saved: {output_path} ({n_voxels} voxels)")
        created_rois.append(('Striatum_bi', n_voxels, ['Caudate', 'Putamen']))

    # Save summary
    summary_path = output_dir / 'roi_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("ROI Summary (Harvard-Oxford Atlas)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Threshold: 25% probability\n\n")
        f.write("References:\n")
        f.write("- Harvard-Oxford Atlas: Desikan et al., 2006, NeuroImage\n")
        f.write("- Visual hierarchy: Grill-Spector & Malach, 2004, Annu Rev Neurosci\n")
        f.write("- DNN-brain: Güçlü & van Gerven, 2015, J Neurosci\n")
        f.write("- Food processing: van der Laan et al., 2011, Obesity Reviews\n\n")
        f.write(f"{'ROI':<25} {'Voxels':>8}  Regions\n")
        f.write("-" * 60 + "\n")
        for roi_name, n_voxels, regions in created_rois:
            regions_str = ', '.join(regions[:2])
            if len(regions) > 2:
                regions_str += f'... (+{len(regions)-2})'
            f.write(f"{roi_name:<25} {n_voxels:>8}  {regions_str}\n")

    print(f"\nSummary saved to: {summary_path}")
    print(f"Total ROIs created: {len(created_rois)}")

    return created_rois


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create ROI masks from Harvard-Oxford atlas')
    parser.add_argument('--output-dir', type=Path,
                       default=Path(__file__).resolve().parents[3] / 'rois' / 'HarvardOxford')
    parser.add_argument('--threshold', type=int, default=25,
                       help='Probability threshold for inclusion (default: 25)')

    args = parser.parse_args()

    create_roi_masks(args.output_dir, args.threshold)
