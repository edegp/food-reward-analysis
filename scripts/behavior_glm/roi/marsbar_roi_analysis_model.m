function results = marsbar_roi_analysis_model(model_name)
%% MarsBar ROI Analysis for specified behavior GLM model
% Extract mean beta/contrast values from second-level analysis
%
% Usage: marsbar_roi_analysis_model('glm_rgb_nutri')

if nargin < 1
    model_name = 'glm_001p_6';
end

% Initialize MarsBar (requires SPM and MarsBar on MATLAB path)
marsbar('on');

%% Define paths (3 levels up from this script: scripts/behavior_glm/roi/)
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');

% Find latest second-level results
second_level_dir = fullfile(root_dir, 'results', 'second_level_analysis', model_name);
spm_dirs = dir(fullfile(second_level_dir, 'ImagexValue_*'));
if isempty(spm_dirs)
    error('No second-level results found for model: %s', model_name);
end
[~, idx] = max([spm_dirs.datenum]);
spm_file = fullfile(spm_dirs(idx).folder, spm_dirs(idx).name, 'SPM.mat');
fprintf('Using SPM.mat: %s\n', spm_file);

roi_dir = fullfile(root_dir, 'rois', 'AAL2');
output_dir = fullfile(root_dir, 'results', 'behavior_glm', 'roi', model_name);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Define ROIs to analyze
roi_names = {
    'vmPFC_mask.nii'
    'OFC_medial_bi_mask.nii'
    'OFC_lateral_bi_mask.nii'
    'Striatum_L_mask.nii'
    'Striatum_R_mask.nii'
    'Insula_bi_mask.nii'
    'Amygdala_bi_mask.nii'
};

%% Extract ROI statistics
fprintf('\n=== ROI Analysis Results for %s ===\n\n', model_name);
fprintf('%-25s %10s %10s %10s %10s\n', 'ROI', 'Mean T', 'Max T', 'Min T', 'Voxels');
fprintf('%s\n', repmat('-', 1, 70));

results = struct();

for i = 1:length(roi_names)
    roi_file = fullfile(roi_dir, roi_names{i});

    if ~exist(roi_file, 'file')
        fprintf('%-25s %s\n', roi_names{i}, 'FILE NOT FOUND');
        continue;
    end

    % Load spmT image
    spmT_file = fullfile(fileparts(spm_file), 'spmT_0001.nii');
    V_t = spm_vol(spmT_file);
    Y_t = spm_read_vols(V_t);

    % Load ROI mask
    V_roi = spm_vol(roi_file);
    Y_roi = spm_read_vols(V_roi);

    % Resample ROI to spmT space if needed
    if ~isequal(V_t.dim, V_roi.dim)
        Y_roi_resampled = zeros(V_t.dim);
        for z = 1:V_t.dim(3)
            for y = 1:V_t.dim(2)
                for x = 1:V_t.dim(1)
                    mni = V_t.mat * [x; y; z; 1];
                    roi_vox = V_roi.mat \ mni;
                    rx = round(roi_vox(1));
                    ry = round(roi_vox(2));
                    rz = round(roi_vox(3));
                    if rx >= 1 && rx <= V_roi.dim(1) && ...
                       ry >= 1 && ry <= V_roi.dim(2) && ...
                       rz >= 1 && rz <= V_roi.dim(3)
                        Y_roi_resampled(x,y,z) = Y_roi(rx, ry, rz);
                    end
                end
            end
        end
        Y_roi = Y_roi_resampled;
    end

    % Get T values within ROI
    roi_mask = Y_roi > 0;
    t_values = Y_t(roi_mask & ~isnan(Y_t));

    if isempty(t_values)
        fprintf('%-25s %s\n', roi_names{i}, 'NO OVERLAP WITH MASK');
        continue;
    end

    % Calculate statistics
    mean_t = mean(t_values);
    max_t = max(t_values);
    min_t = min(t_values);
    n_voxels = length(t_values);

    fprintf('%-25s %10.3f %10.3f %10.3f %10d\n', ...
        strrep(roi_names{i}, '_mask.nii', ''), mean_t, max_t, min_t, n_voxels);

    % Store results
    results(i).name = roi_names{i};
    results(i).mean_t = mean_t;
    results(i).max_t = max_t;
    results(i).min_t = min_t;
    results(i).n_voxels = n_voxels;
    results(i).t_values = t_values;
end

fprintf('\n');

%% Statistical significance (manual t-test to avoid Statistics Toolbox)
fprintf('=== Significance Testing (one-sample t-test, H0: T=0) ===\n\n');
fprintf('%-25s %10s %10s %10s\n', 'ROI', 'Mean T', 'p-value', 'Sig');
fprintf('%s\n', repmat('-', 1, 55));

for i = 1:length(results)
    if isfield(results(i), 't_values') && ~isempty(results(i).t_values)
        % Manual one-sample t-test
        x = results(i).t_values;
        n = length(x);
        m = mean(x);
        s = std(x);
        se = s / sqrt(n);
        t_stat = m / se;
        df = n - 1;
        % Two-tailed p-value using incomplete beta function
        p = 2 * betainc(df / (df + t_stat^2), df/2, 0.5);

        if p < 0.001
            sig = '***';
        elseif p < 0.01
            sig = '**';
        elseif p < 0.05
            sig = '*';
        else
            sig = '';
        end
        fprintf('%-25s %10.3f %10.4f %10s\n', ...
            strrep(results(i).name, '_mask.nii', ''), results(i).mean_t, p, sig);
        results(i).p_value = p;
    end
end

fprintf('\n* p<0.05, ** p<0.01, *** p<0.001\n');

%% Save results
save(fullfile(output_dir, 'roi_analysis_results.mat'), 'results');
fprintf('\nResults saved to: %s\n', fullfile(output_dir, 'roi_analysis_results.mat'));

end
