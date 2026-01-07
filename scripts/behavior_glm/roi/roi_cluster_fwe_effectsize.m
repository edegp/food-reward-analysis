%% ROI Effect Size with Cluster-level FWE mask
% Use same ROIs but only include voxels that survive cluster-level FWE
% (Requires SPM on MATLAB path)

%% Define paths (3 levels up from this script: scripts/behavior_glm/roi/)
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
cluster_file = fullfile(root_dir, 'results', 'behavior_glm', 'cluster_fwe', 'glm_001p_6', 'ImagexValue_clusterFWE.nii');
first_level_base = fullfile(root_dir, 'results', 'first_level_analysis');
roi_dir = fullfile(root_dir, 'rois', 'AAL2');
output_dir = fullfile(root_dir, 'results', 'roi_analysis');

%% Load cluster FWE mask
V_cluster = spm_vol(cluster_file);
Y_cluster = spm_read_vols(V_cluster);
cluster_mask = Y_cluster > 0;  % All significant voxels

fprintf('Cluster FWE significant voxels: %d\n', sum(cluster_mask(:)));

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

%% Find subjects
subjects = dir(fullfile(first_level_base, 'sub-*'));
n_subjects = length(subjects);

%% Extract contrast values for each subject and ROI (cluster FWE masked)
contrast_values = zeros(n_subjects, n_rois);
n_voxels_roi = zeros(n_rois, 1);
n_voxels_sig = zeros(n_rois, 1);  % Voxels within ROI AND cluster FWE significant

for s = 1:n_subjects
    sub_id = subjects(s).name;

    % Find latest run
    glm_dir = fullfile(first_level_base, sub_id, 'glm_model', 'glm_rgb_nutri');
    runs = dir(fullfile(glm_dir, '202*'));
    if isempty(runs), continue; end
    latest_run = runs(end).name;

    % Load contrast image
    con_file = fullfile(glm_dir, latest_run, 'con_0001.nii');
    if ~exist(con_file, 'file'), continue; end

    V_con = spm_vol(con_file);
    Y_con = spm_read_vols(V_con);

    for r = 1:n_rois
        roi_file = fullfile(roi_dir, roi_files{r});
        if ~exist(roi_file, 'file'), continue; end

        V_roi = spm_vol(roi_file);
        Y_roi = spm_read_vols(V_roi);

        % Resample ROI and cluster mask to contrast space
        Y_roi_resampled = zeros(V_con.dim);
        Y_cluster_resampled = zeros(V_con.dim);

        for z = 1:V_con.dim(3)
            for y = 1:V_con.dim(2)
                for x = 1:V_con.dim(1)
                    mni = V_con.mat * [x; y; z; 1];

                    % ROI
                    roi_vox = V_roi.mat \ mni;
                    rx = round(roi_vox(1)); ry = round(roi_vox(2)); rz = round(roi_vox(3));
                    if rx >= 1 && rx <= V_roi.dim(1) && ry >= 1 && ry <= V_roi.dim(2) && rz >= 1 && rz <= V_roi.dim(3)
                        Y_roi_resampled(x,y,z) = Y_roi(rx, ry, rz);
                    end

                    % Cluster mask
                    cluster_vox = V_cluster.mat \ mni;
                    cx = round(cluster_vox(1)); cy = round(cluster_vox(2)); cz = round(cluster_vox(3));
                    if cx >= 1 && cx <= V_cluster.dim(1) && cy >= 1 && cy <= V_cluster.dim(2) && cz >= 1 && cz <= V_cluster.dim(3)
                        Y_cluster_resampled(x,y,z) = Y_cluster(cx, cy, cz);
                    end
                end
            end
        end

        % Combined mask: ROI AND cluster FWE significant
        combined_mask = (Y_roi_resampled > 0) & (Y_cluster_resampled > 0) & ~isnan(Y_con);

        if s == 1
            n_voxels_roi(r) = sum(Y_roi_resampled(:) > 0);
            n_voxels_sig(r) = sum(combined_mask(:));
        end

        con_vals = Y_con(combined_mask);
        if ~isempty(con_vals)
            contrast_values(s, r) = mean(con_vals);
        else
            contrast_values(s, r) = NaN;
        end
    end

    if mod(s, 10) == 0
        fprintf('Processed %d/%d subjects\n', s, n_subjects);
    end
end

%% Calculate effect sizes (only for ROIs with significant voxels)
fprintf('\n========================================\n');
fprintf('ROI Effect Size (Cluster FWE masked)\n');
fprintf('========================================\n');
fprintf('%-15s %8s %8s %8s %8s %8s %8s\n', 'ROI', 'Total', 'Sig', 'Beta', 'SD', 'd', 'p');
fprintf('------------------------------------------------------------------------\n');

results = struct();
results.roi_labels_jp = roi_labels_jp;
results.n_voxels_roi = n_voxels_roi;
results.n_voxels_sig = n_voxels_sig;

mean_beta = zeros(n_rois, 1);
std_beta = zeros(n_rois, 1);
cohens_d = zeros(n_rois, 1);
p_values = ones(n_rois, 1);
significant = zeros(n_rois, 1);

for r = 1:n_rois
    if n_voxels_sig(r) > 0
        vals = contrast_values(~isnan(contrast_values(:, r)), r);
        n = length(vals);

        mean_beta(r) = mean(vals);
        std_beta(r) = std(vals);
        cohens_d(r) = mean_beta(r) / std_beta(r);

        % T-test
        t_val = mean_beta(r) / (std_beta(r) / sqrt(n));
        df = n - 1;
        tcdf_custom = @(t, v) 0.5 * (1 + sign(t) .* (1 - betainc(v./(v + t.^2), v/2, 0.5)));
        p_values(r) = 2 * (1 - tcdf_custom(abs(t_val), df));
        significant(r) = 1;

        sig_str = '';
        if p_values(r) < 0.001, sig_str = '***';
        elseif p_values(r) < 0.01, sig_str = '**';
        elseif p_values(r) < 0.05, sig_str = '*'; end

        fprintf('%-15s %8d %8d %8.3f %8.3f %8.2f %8.4f %s\n', ...
            roi_labels_jp{r}, n_voxels_roi(r), n_voxels_sig(r), ...
            mean_beta(r), std_beta(r), cohens_d(r), p_values(r), sig_str);
    else
        fprintf('%-15s %8d %8d %8s %8s %8s %8s (no sig voxels)\n', ...
            roi_labels_jp{r}, n_voxels_roi(r), n_voxels_sig(r), '-', '-', '-', '-');
    end
end

results.mean_beta = mean_beta;
results.std_beta = std_beta;
results.se_beta = std_beta / sqrt(n_subjects);
results.cohens_d = cohens_d;
results.p_values = p_values;
results.significant = significant;
results.n_subjects = n_subjects;

%% Save results
save(fullfile(output_dir, 'roi_cluster_fwe_effectsize.mat'), 'results');

T = table(roi_labels_jp', n_voxels_roi, n_voxels_sig, mean_beta, std_beta, results.se_beta, cohens_d, p_values, significant, ...
    'VariableNames', {'ROI', 'n_voxels_total', 'n_voxels_sig', 'mean_beta', 'std_beta', 'se_beta', 'cohens_d', 'p_value', 'significant'});
writetable(T, fullfile(output_dir, 'roi_cluster_fwe_effectsize.csv'));
fprintf('\nSaved: %s\n', fullfile(output_dir, 'roi_cluster_fwe_effectsize.csv'));
