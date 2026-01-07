%% ROI Analysis - Second-level Beta Values
% Extract beta values from second-level analysis

addpath('/Users/yuhiaoki/spm');

%% Define paths
second_level_dir = '/Users/yuhiaoki/dev/hit/food-brain/results/second_level_analysis/glm_rgb_nutri/ImagexValue_20251221_145043';
roi_dir = '/Users/yuhiaoki/dev/hit/food-brain/rois/AAL2';
output_dir = '/Users/yuhiaoki/dev/hit/food-brain/results/roi_analysis';

%% Load second-level beta image
beta_file = fullfile(second_level_dir, 'beta_0001.nii');
V_beta = spm_vol(beta_file);
Y_beta = spm_read_vols(V_beta);

% Load ResMS for SE calculation
resms_file = fullfile(second_level_dir, 'ResMS.nii');
V_resms = spm_vol(resms_file);
Y_resms = spm_read_vols(V_resms);

%% Define ROIs
roi_files = {
    'vmPFC_mask.nii'
    'OFC_medial_bi_mask.nii'
    'Striatum_L_mask.nii'
    'Striatum_R_mask.nii'
    'Insula_bi_mask.nii'
    'Amygdala_bi_mask.nii'
};

roi_labels_jp = {'vmPFC', '内側OFC', '左線条体', '右線条体', '島皮質', '扁桃体'};
n_rois = length(roi_files);

%% Extract ROI values
mean_beta = zeros(n_rois, 1);
mean_resms = zeros(n_rois, 1);
n_voxels = zeros(n_rois, 1);

fprintf('\n=== Second-level Beta Values ===\n');

for r = 1:n_rois
    roi_file = fullfile(roi_dir, roi_files{r});
    if ~exist(roi_file, 'file')
        fprintf('Warning: %s not found\n', roi_file);
        continue;
    end

    V_roi = spm_vol(roi_file);
    Y_roi = spm_read_vols(V_roi);

    % Resample ROI to beta space if needed
    if ~isequal(V_beta.dim, V_roi.dim)
        Y_roi_resampled = zeros(V_beta.dim);
        for z = 1:V_beta.dim(3)
            for y = 1:V_beta.dim(2)
                for x = 1:V_beta.dim(1)
                    mni = V_beta.mat * [x; y; z; 1];
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

    % Get values within ROI
    roi_mask = Y_roi > 0 & ~isnan(Y_beta);
    beta_values = Y_beta(roi_mask);
    resms_values = Y_resms(roi_mask);

    mean_beta(r) = mean(beta_values);
    mean_resms(r) = mean(resms_values);
    n_voxels(r) = length(beta_values);

    fprintf('%-15s: beta = %.4f (n=%d voxels)\n', roi_labels_jp{r}, mean_beta(r), n_voxels(r));
end

%% Calculate SE from ResMS
% SE = sqrt(ResMS / n_subjects)
% For one-sample t-test at second level
n_subjects = 31;
se_beta = sqrt(mean_resms / n_subjects);

% T-values
t_values = mean_beta ./ sqrt(mean_resms);

% P-values (custom tcdf)
df = n_subjects - 1;
tcdf_custom = @(t, v) 0.5 * (1 + sign(t) .* (1 - betainc(v./(v + t.^2), v/2, 0.5)));
p_values = 2 * (1 - tcdf_custom(abs(t_values), df));

%% Print results
fprintf('\n=== Results Summary ===\n');
fprintf('%-15s %10s %10s %10s %10s\n', 'ROI', 'Beta', 'SE', 'T', 'p-value');
fprintf('-----------------------------------------------------------\n');
for r = 1:n_rois
    sig_str = '';
    if p_values(r) < 0.001
        sig_str = '***';
    elseif p_values(r) < 0.01
        sig_str = '**';
    elseif p_values(r) < 0.05
        sig_str = '*';
    end
    fprintf('%-15s %10.4f %10.4f %10.2f %10.4f %s\n', ...
        roi_labels_jp{r}, mean_beta(r), se_beta(r), t_values(r), p_values(r), sig_str);
end

%% Save results
results_2nd = struct();
results_2nd.roi_labels_jp = roi_labels_jp;
results_2nd.mean_beta = mean_beta';
results_2nd.se_beta = se_beta';
results_2nd.t_values = t_values';
results_2nd.p_values = p_values';
results_2nd.n_voxels = n_voxels';
results_2nd.n_subjects = n_subjects;

save(fullfile(output_dir, 'roi_secondlevel_beta.mat'), 'results_2nd');

% Save as CSV
T = table(roi_labels_jp', mean_beta, se_beta, t_values, p_values, n_voxels, ...
    'VariableNames', {'ROI', 'mean_beta', 'se_beta', 't_value', 'p_value', 'n_voxels'});
writetable(T, fullfile(output_dir, 'roi_secondlevel_beta.csv'));
fprintf('\nSaved: %s\n', fullfile(output_dir, 'roi_secondlevel_beta.csv'));
