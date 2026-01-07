%% ROI Effect Size Analysis - Beta-based
% Extract contrast values from each subject's first-level analysis
% Calculate effect size (Cohen's d) from beta values
% (Requires SPM on MATLAB path)

%% Define paths (3 levels up from this script: scripts/behavior_glm/roi/)
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
first_level_base = fullfile(root_dir, 'results', 'first_level_analysis');
roi_dir = fullfile(root_dir, 'rois', 'AAL2');
output_dir = fullfile(root_dir, 'results', 'roi_analysis');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Define subjects
subjects = dir(fullfile(first_level_base, 'sub-*'));
n_subjects = length(subjects);
fprintf('Found %d subjects\n', n_subjects);

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

%% Extract contrast values for each subject and ROI
contrast_values = zeros(n_subjects, n_rois);

for s = 1:n_subjects
    sub_id = subjects(s).name;

    % Find latest run
    glm_dir = fullfile(first_level_base, sub_id, 'glm_model', 'glm_rgb_nutri');
    runs = dir(fullfile(glm_dir, '202*'));
    if isempty(runs)
        fprintf('Warning: No runs found for %s\n', sub_id);
        continue;
    end
    latest_run = runs(end).name;

    % Load contrast image
    con_file = fullfile(glm_dir, latest_run, 'con_0001.nii');
    if ~exist(con_file, 'file')
        fprintf('Warning: %s not found\n', con_file);
        continue;
    end

    V_con = spm_vol(con_file);
    Y_con = spm_read_vols(V_con);

    % Extract ROI values
    for r = 1:n_rois
        roi_file = fullfile(roi_dir, roi_files{r});
        if ~exist(roi_file, 'file')
            continue;
        end

        V_roi = spm_vol(roi_file);
        Y_roi = spm_read_vols(V_roi);

        % Resample ROI to contrast space if needed
        if ~isequal(V_con.dim, V_roi.dim)
            Y_roi_resampled = zeros(V_con.dim);
            for z = 1:V_con.dim(3)
                for y = 1:V_con.dim(2)
                    for x = 1:V_con.dim(1)
                        mni = V_con.mat * [x; y; z; 1];
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

        % Get mean contrast value within ROI
        roi_mask = Y_roi > 0;
        con_values = Y_con(roi_mask & ~isnan(Y_con));

        if ~isempty(con_values)
            contrast_values(s, r) = mean(con_values);
        end
    end

    if mod(s, 5) == 0
        fprintf('Processed %d/%d subjects\n', s, n_subjects);
    end
end

%% Calculate statistics
mean_beta = mean(contrast_values, 1);
std_beta = std(contrast_values, 0, 1);
se_beta = std_beta / sqrt(n_subjects);

% Cohen's d = mean / SD
cohens_d = mean_beta ./ std_beta;

% 95% CI for d (approximation)
se_d = sqrt(1/n_subjects + cohens_d.^2 / (2*n_subjects));
ci_95 = 1.96 * se_d;

% T-test for significance (custom implementation - no Statistics Toolbox)
% One-sample t-test: t = mean / (SD / sqrt(n))
t_values = mean_beta ./ (std_beta / sqrt(n_subjects));

% Custom tcdf using incomplete beta function
tcdf_custom = @(t, v) 0.5 * (1 + sign(t) .* (1 - betainc(v./(v + t.^2), v/2, 0.5)));

df = n_subjects - 1;
p_values = 2 * (1 - tcdf_custom(abs(t_values), df));  % two-tailed

%% Print results
fprintf('\n=== Beta-based Effect Size Results ===\n');
fprintf('%-15s %10s %10s %10s %10s %10s\n', 'ROI', 'Mean Beta', 'SD', 'Cohen''s d', '95% CI', 'p-value');
fprintf('----------------------------------------------------------------------\n');
for r = 1:n_rois
    sig_str = '';
    if p_values(r) < 0.001
        sig_str = '***';
    elseif p_values(r) < 0.01
        sig_str = '**';
    elseif p_values(r) < 0.05
        sig_str = '*';
    end
    fprintf('%-15s %10.3f %10.3f %10.2f %10.2f %10.4f %s\n', ...
        roi_labels_jp{r}, mean_beta(r), std_beta(r), cohens_d(r), ci_95(r), p_values(r), sig_str);
end

%% Save results for Python visualization
results_beta = struct();
results_beta.roi_labels_jp = roi_labels_jp;
results_beta.mean_beta = mean_beta;
results_beta.std_beta = std_beta;
results_beta.se_beta = se_beta;
results_beta.cohens_d = cohens_d;
results_beta.ci_95 = ci_95;
results_beta.p_values = p_values;
results_beta.n_subjects = n_subjects;
results_beta.contrast_values = contrast_values;

save(fullfile(output_dir, 'roi_effectsize_beta.mat'), 'results_beta');
fprintf('\nResults saved to: %s\n', fullfile(output_dir, 'roi_effectsize_beta.mat'));

%% Also save as CSV for easy access
T = table(roi_labels_jp', mean_beta', std_beta', se_beta', cohens_d', ci_95', p_values', ...
    'VariableNames', {'ROI', 'mean_beta', 'std_beta', 'se_beta', 'cohens_d', 'ci_95', 'p_value'});
writetable(T, fullfile(output_dir, 'roi_effectsize_beta.csv'));
fprintf('CSV saved to: %s\n', fullfile(output_dir, 'roi_effectsize_beta.csv'));
