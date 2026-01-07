function create_aal_rois()
%% Create ROI masks from AAL2 atlas for food valuation study
% Based on Suzuki et al. (2017) and Bartra et al. (2013)
%
% ROIs:
%   - vmPFC: Frontal_Med_Orb + Rectus
%   - Medial OFC: OFCmed
%   - Lateral OFC: OFClat
%   - L-Striatum: Caudate_L + Putamen_L + Pallidum_L
%   - R-Striatum: Caudate_R + Putamen_R + Pallidum_R

clearvars; close all; clc;

% Paths (3 levels up from this script: scripts/behavior_glm/roi/)
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
% Note: SPM path is assumed to be on MATLAB path
% Atlas file location depends on SPM installation
spm_dir = fileparts(which('spm'));
atlas_file = fullfile(spm_dir, 'toolbox', 'bspmview', 'supportfiles', 'AAL2_Atlas_Map.nii');
output_dir = fullfile(root_dir, 'rois', 'AAL2');

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Load atlas
V_atlas = spm_vol(atlas_file);
Y_atlas = spm_read_vols(V_atlas);

% AAL2 region indices (from AAL2_Atlas_Labels.csv)
roi_definitions = {
    % ROI name, indices, description
    'vmPFC',           [21, 22, 23, 24],        'Frontal_Med_Orb_L/R + Rectus_L/R';
    'OFC_medial_L',    [25],                    'OFCmed_L';
    'OFC_medial_R',    [26],                    'OFCmed_R';
    'OFC_medial_bi',   [25, 26],                'OFCmed bilateral';
    'OFC_lateral_L',   [31],                    'OFClat_L';
    'OFC_lateral_R',   [32],                    'OFClat_R';
    'OFC_lateral_bi',  [31, 32],                'OFClat bilateral';
    'OFC_anterior_L',  [27],                    'OFCant_L';
    'OFC_anterior_R',  [28],                    'OFCant_R';
    'OFC_anterior_bi', [27, 28],                'OFCant bilateral';
    'Striatum_L',      [75, 77, 79],            'Caudate_L + Putamen_L + Pallidum_L';
    'Striatum_R',      [76, 78, 80],            'Caudate_R + Putamen_R + Pallidum_R';
    'Striatum_bi',     [75, 76, 77, 78, 79, 80],'Striatum bilateral';
    'Caudate_L',       [75],                    'Caudate_L';
    'Caudate_R',       [76],                    'Caudate_R';
    'Putamen_L',       [77],                    'Putamen_L';
    'Putamen_R',       [78],                    'Putamen_R';
    'Insula_L',        [33],                    'Insula_L';
    'Insula_R',        [34],                    'Insula_R';
    'Insula_bi',       [33, 34],                'Insula bilateral';
    'Amygdala_L',      [45],                    'Amygdala_L';
    'Amygdala_R',      [46],                    'Amygdala_R';
    'Amygdala_bi',     [45, 46],                'Amygdala bilateral';
};

fprintf('Creating ROI masks from AAL2 atlas...\n');
fprintf('Atlas: %s\n', atlas_file);
fprintf('Output: %s\n\n', output_dir);

% Create each ROI mask
for i = 1:size(roi_definitions, 1)
    roi_name = roi_definitions{i, 1};
    roi_indices = roi_definitions{i, 2};
    roi_desc = roi_definitions{i, 3};

    % Create binary mask
    Y_roi = zeros(size(Y_atlas));
    for idx = roi_indices
        Y_roi = Y_roi | (Y_atlas == idx);
    end

    % Count voxels
    n_voxels = sum(Y_roi(:));

    % Calculate volume in mm^3
    voxel_size = abs(det(V_atlas.mat(1:3, 1:3)));
    volume_mm3 = n_voxels * voxel_size;
    volume_cm3 = volume_mm3 / 1000;

    % Save mask
    V_out = V_atlas;
    V_out.fname = fullfile(output_dir, [roi_name, '_mask.nii']);
    V_out.dt = [spm_type('uint8'), 0];
    spm_write_vol(V_out, uint8(Y_roi));

    fprintf('  %s: %d voxels (%.2f cm³) - %s\n', roi_name, n_voxels, volume_cm3, roi_desc);
end

fprintf('\nROI masks created successfully!\n');
fprintf('Output directory: %s\n', output_dir);

% Create summary table
summary_file = fullfile(output_dir, 'roi_summary.txt');
fid = fopen(summary_file, 'w');
fprintf(fid, 'AAL2 ROI Summary for Food Valuation Study\n');
fprintf(fid, '==========================================\n\n');
fprintf(fid, 'References:\n');
fprintf(fid, '  - Suzuki et al. (2017) Nature Communications\n');
fprintf(fid, '  - Bartra et al. (2013) NeuroImage\n\n');
fprintf(fid, 'Atlas: AAL2 (Rolls et al., 2015)\n\n');
fprintf(fid, 'ROI Definitions:\n');
fprintf(fid, '%-20s %-30s %s\n', 'ROI Name', 'AAL2 Regions', 'Description');
fprintf(fid, '%-20s %-30s %s\n', repmat('-', 1, 20), repmat('-', 1, 30), repmat('-', 1, 30));
for i = 1:size(roi_definitions, 1)
    fprintf(fid, '%-20s %-30s %s\n', roi_definitions{i, 1}, mat2str(roi_definitions{i, 2}), roi_definitions{i, 3});
end
fclose(fid);

fprintf('Summary saved: %s\n', summary_file);

%% Create MarsBar ROI files (.mat format)
fprintf('\nCreating MarsBar ROI files...\n');
marsbar_dir = fullfile(output_dir, 'marsbar');
if ~exist(marsbar_dir, 'dir')
    mkdir(marsbar_dir);
end

% Add MarsBar to path if not already
marsbar_path = fullfile(spm_dir, 'toolbox', 'marsbar');
if exist(marsbar_path, 'dir')
    addpath(marsbar_path);

    for i = 1:size(roi_definitions, 1)
        roi_name = roi_definitions{i, 1};
        nii_file = fullfile(output_dir, [roi_name, '_mask.nii']);
        mat_file = fullfile(marsbar_dir, [roi_name, '_roi.mat']);

        try
            % Create MarsBar ROI from NIfTI mask
            roi = maroi_image(nii_file);
            roi = label(roi, roi_name);
            saveroi(roi, mat_file);
            fprintf('  Created: %s\n', mat_file);
        catch ME
            fprintf('  Warning: Could not create MarsBar ROI for %s: %s\n', roi_name, ME.message);
        end
    end
    fprintf('\nMarsBar ROIs saved to: %s\n', marsbar_dir);
else
    fprintf('MarsBar not found at %s. Skipping MarsBar ROI creation.\n', marsbar_path);
end

end
