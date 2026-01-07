%% Small Volume Correction (SVC) Analysis for Hierarchical DNN GLM
% Apply voxel-level FWE correction within each ROI
%
% Usage: run from MATLAB with SPM12 in path

clear; clc;

%% Setup paths
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');

% ROI directory
roi_dir = fullfile(root_dir, 'rois', 'HarvardOxford');

% Output directory
output_dir = fullfile(root_dir, 'results', 'dnn_analysis', 'roi_analysis', 'hierarchical_svc');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Define sources and contrasts
sources = {'clip', 'convnext'};

contrast_info = {
    'Initial_only', '初期層のみ';
    'Initial_withShared', '初期層+共有';
    'Initial_withShared_withGlobal', '初期層+共有+全体';
    'Middle_only', '中間層のみ';
    'Middle_withShared', '中間層+共有';
    'Middle_withShared_withGlobal', '中間層+共有+全体';
    'Late_only', '後期層のみ';
    'Late_withShared', '後期層+共有';
    'Late_withShared_withGlobal', '後期層+共有+全体';
    'Final_only', '最終層のみ';
    'Final_withShared', '最終層+共有';
    'Final_withShared_withGlobal', '最終層+共有+全体';
    'Shared_F', '共有PC（統合）';
    'Shared_Initial_Middle_F', '共有（初期-中間）';
    'Shared_Middle_Late_F', '共有（中間-後期）';
    'Shared_Late_Final_F', '共有（後期-最終）';
    'Global_F', '全体';
};

%% Find ROI masks
roi_files = dir(fullfile(roi_dir, '*_mask.nii'));
if isempty(roi_files)
    roi_files = dir(fullfile(roi_dir, '*_mask.nii.gz'));
end
fprintf('Found %d ROI masks\n', length(roi_files));

%% Run SVC for each source
for s = 1:length(sources)
    source = sources{s};
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('Processing %s\n', upper(source));
    fprintf('%s\n', repmat('=', 1, 60));

    % Find second-level directory
    second_level_base = fullfile(root_dir, 'results', 'dnn_analysis', 'second_level');
    % Use daywise results (updated 2024-12-29)
    design_dir = fullfile(second_level_base, sprintf('hierarchical_%s_v3_daywise', source));

    runs = dir(design_dir);
    runs = runs([runs.isdir] & ~startsWith({runs.name}, '.'));
    if isempty(runs)
        warning('No runs found in %s', design_dir);
        continue;
    end
    spm_dir = fullfile(design_dir, runs(end).name);
    fprintf('Second-level directory: %s\n', spm_dir);

    % Results table
    results = {};

    %% Process each contrast
    for c = 1:size(contrast_info, 1)
        contrast_name = contrast_info{c, 1};
        contrast_label = contrast_info{c, 2};

        contrast_dir = fullfile(spm_dir, contrast_name);
        spm_mat_file = fullfile(contrast_dir, 'SPM.mat');

        if ~exist(spm_mat_file, 'file')
            fprintf('Warning: %s not found\n', spm_mat_file);
            continue;
        end

        fprintf('\nContrast: %s\n', contrast_name);

        % Load SPM.mat
        load(spm_mat_file, 'SPM');

        %% Process each ROI
        for r = 1:length(roi_files)
            roi_file = fullfile(roi_files(r).folder, roi_files(r).name);
            roi_name = strrep(roi_files(r).name, '_mask.nii', '');
            roi_name = strrep(roi_name, '.gz', '');

            try
                % Apply SVC
                [p_fwe, peak_F, peak_xyz, n_voxels, mean_eta2p_val, std_eta2p_val, se_eta2p_val] = apply_svc(SPM, roi_file, spm_mat_file);

                % Calculate effect size
                df1 = SPM.xCon(1).eidf;
                df2 = SPM.xX.erdf;
                if ~isnan(peak_F)
                    eta2p = (df1 * peak_F) / (df1 * peak_F + df2);
                    mean_eta2p = mean_eta2p_val;
                    std_eta2p = std_eta2p_val;
                    se_eta2p = se_eta2p_val;
                else
                    eta2p = NaN;
                    mean_eta2p = NaN;
                    std_eta2p = NaN;
                    se_eta2p = NaN;
                end

                % Store results
                results{end+1, 1} = source;
                results{end, 2} = contrast_name;
                results{end, 3} = contrast_label;
                results{end, 4} = roi_name;
                results{end, 5} = n_voxels;
                results{end, 6} = peak_F;
                results{end, 7} = p_fwe;
                results{end, 8} = p_fwe < 0.05;  % FWE threshold
                results{end, 9} = eta2p;
                results{end, 10} = mean_eta2p;
                results{end, 11} = std_eta2p;
                results{end, 12} = se_eta2p;
                results{end, 13} = df1;
                results{end, 14} = df2;
                results{end, 15} = peak_xyz(1);
                results{end, 16} = peak_xyz(2);
                results{end, 17} = peak_xyz(3);

                if p_fwe < 0.05
                    fprintf('  %s: F=%.2f, p_FWE=%.4f*, eta2p=%.3f, mean=%.3f, SE=%.4f\n', ...
                        roi_name, peak_F, p_fwe, eta2p, mean_eta2p, se_eta2p);
                end

            catch ME
                fprintf('  Error processing %s: %s\n', roi_name, ME.message);
            end
        end
    end

    %% Save results
    if ~isempty(results)
        % Convert to table
        T = cell2table(results, 'VariableNames', ...
            {'source', 'contrast', 'contrast_label', 'roi', 'n_voxels', ...
             'peak_F', 'p_fwe', 'significant', 'peak_eta2p', 'mean_eta2p', ...
             'std_eta2p', 'se_eta2p', 'df1', 'df2', 'x', 'y', 'z'});

        csv_file = fullfile(output_dir, sprintf('%s_hierarchical_svc.csv', source));
        writetable(T, csv_file);
        fprintf('\nSaved: %s\n', csv_file);
    end
end

fprintf('\nDone!\n');


%% Helper function: Apply SVC
function [p_fwe, peak_F, peak_xyz, n_voxels, mean_eta2p, std_eta2p, se_eta2p] = apply_svc(SPM, roi_file, spm_mat_file)
    % Apply Small Volume Correction using ROI mask (voxel-level FWE)

    % Load ROI mask
    V_roi = spm_vol(roi_file);
    Y_roi = spm_read_vols(V_roi);

    % Get ROI voxel coordinates
    [i, j, k] = ind2sub(size(Y_roi), find(Y_roi > 0));
    n_voxels = length(i);

    if n_voxels == 0
        p_fwe = NaN;
        peak_F = NaN;
        peak_xyz = [NaN NaN NaN];
        mean_eta2p = NaN;
        std_eta2p = NaN;
        se_eta2p = NaN;
        return;
    end

    % Convert to mm coordinates
    XYZ_roi_mm = V_roi.mat * [i'; j'; k'; ones(1, n_voxels)];
    XYZ_roi_mm = XYZ_roi_mm(1:3, :);

    % Load spmF image (use current contrast_dir instead of SPM.swd which may be outdated)
    spmF_file = fullfile(fileparts(spm_mat_file), 'spmF_0001.nii');
    if ~exist(spmF_file, 'file')
        error('spmF_0001.nii not found in %s', fileparts(spm_mat_file));
    end

    V_F = spm_vol(spmF_file);

    % Convert ROI mm to voxel indices in F-map space
    XYZ_F_vox = V_F.mat \ [XYZ_roi_mm; ones(1, n_voxels)];
    XYZ_F_vox = round(XYZ_F_vox(1:3, :));

    % Get F values within ROI
    Y_F = spm_read_vols(V_F);

    % Filter valid voxels
    valid = XYZ_F_vox(1,:) >= 1 & XYZ_F_vox(1,:) <= size(Y_F,1) & ...
            XYZ_F_vox(2,:) >= 1 & XYZ_F_vox(2,:) <= size(Y_F,2) & ...
            XYZ_F_vox(3,:) >= 1 & XYZ_F_vox(3,:) <= size(Y_F,3);

    XYZ_F_vox = XYZ_F_vox(:, valid);
    XYZ_roi_mm = XYZ_roi_mm(:, valid);
    n_voxels = sum(valid);

    if n_voxels == 0
        p_fwe = NaN;
        peak_F = NaN;
        peak_xyz = [NaN NaN NaN];
        mean_eta2p = NaN;
        std_eta2p = NaN;
        se_eta2p = NaN;
        return;
    end

    % Get F values
    ind = sub2ind(size(Y_F), XYZ_F_vox(1,:), XYZ_F_vox(2,:), XYZ_F_vox(3,:));
    F_vals = Y_F(ind);

    % Remove NaN
    valid_F = ~isnan(F_vals) & F_vals > 0;
    F_vals = F_vals(valid_F);
    XYZ_roi_mm = XYZ_roi_mm(:, valid_F);
    n_voxels = sum(valid_F);

    if n_voxels == 0
        p_fwe = NaN;
        peak_F = NaN;
        peak_xyz = [NaN NaN NaN];
        mean_eta2p = NaN;
        std_eta2p = NaN;
        se_eta2p = NaN;
        return;
    end

    % Find peak
    [peak_F, peak_idx] = max(F_vals);
    peak_xyz = XYZ_roi_mm(:, peak_idx)';

    % Calculate effect sizes for all voxels in ROI
    df1 = SPM.xCon(1).eidf;
    df2 = SPM.xX.erdf;
    eta2p_vals = (df1 .* F_vals) ./ (df1 .* F_vals + df2);
    mean_eta2p = mean(eta2p_vals);
    std_eta2p = std(eta2p_vals);
    se_eta2p = std_eta2p / sqrt(n_voxels);

    % Calculate SVC p-value using Random Field Theory
    % Get smoothness estimate (FWHM in voxels)
    if isfield(SPM.xVol, 'FWHM')
        FWHM = SPM.xVol.FWHM;
    else
        FWHM = [3 3 3];  % Default
    end

    % Calculate RESEL count for the ROI
    voxel_size = sqrt(sum(V_F.mat(1:3,1:3).^2));
    FWHM_mm = FWHM .* voxel_size;

    % Volume in RESELs
    roi_vol_mm3 = n_voxels * prod(voxel_size);
    resel_vol = prod(FWHM_mm);
    S = roi_vol_mm3 / resel_vol;  % Number of RESELs

    % Approximate RESEL counts [0D, 1D, 2D, 3D]
    R = [1, 0, 0, S];

    % Get degrees of freedom
    df = [df1, df2];

    % Calculate p-value using spm_P (voxel-level FWE)
    % k=0 means voxel-level inference (no cluster extent requirement)
    [p_fwe, ~] = spm_P(1, 0, peak_F, df, 'F', R, 1, n_voxels);
end
