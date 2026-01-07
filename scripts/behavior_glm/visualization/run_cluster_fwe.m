function run_cluster_fwe(model_name)
%% Run cluster-level FWE analysis using proper SPM RFT calculation
%
% Usage:
%   run_cluster_fwe('glm_001p_6')
%   run_cluster_fwe('glm_rgb_nutri')
%
% This function calculates the cluster extent threshold based on
% the actual smoothness (RESEL) of the data using SPM's RFT functions.

if nargin < 1
    model_name = 'glm_001p_6';
end

% Setup paths
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
results_base = fullfile(root_dir, 'results', 'second_level_analysis');

% Find model directory
model_dir = fullfile(results_base, model_name);
if ~exist(model_dir, 'dir')
    error('Model directory not found: %s', model_dir);
end

% Output directory
output_dir = fullfile(root_dir, 'results', 'behavior_glm', 'cluster_fwe', model_name);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Parameters
cluster_p = 0.05;   % Cluster-level FWE threshold
voxel_p = 0.001;    % Cluster-forming threshold (uncorrected)

fprintf('\n========================================\n');
fprintf('Cluster-level FWE Analysis (proper RFT)\n');
fprintf('========================================\n');
fprintf('Model: %s\n', model_name);
fprintf('Cluster FWE: p < %.3f\n', cluster_p);
fprintf('Cluster-forming: p < %.4f (uncorrected)\n', voxel_p);
fprintf('Output: %s\n', output_dir);
fprintf('========================================\n\n');

% Initialize SPM
spm('defaults', 'fmri');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);

% Find all contrast directories
subdirs = dir(model_dir);
subdirs = subdirs([subdirs.isdir] & ~startsWith({subdirs.name}, '.'));

% Group by contrast name and find latest
contrast_dirs = struct();
for i = 1:length(subdirs)
    name = subdirs(i).name;
    % Parse: ContrastName_YYYYMMDD_HHMMSS
    parts = regexp(name, '(.+)_(\d{8}_\d{6})$', 'tokens');
    if ~isempty(parts)
        contrast_name = parts{1}{1};
        timestamp = parts{1}{2};

        % Check if this directory has SPM.mat (skip incomplete runs)
        if ~exist(fullfile(model_dir, name, 'SPM.mat'), 'file')
            continue;
        end

        % Compare timestamps lexicographically (YYYYMMDD_HHMMSS format)
        if ~isfield(contrast_dirs, contrast_name)
            contrast_dirs.(contrast_name).timestamp = timestamp;
            contrast_dirs.(contrast_name).path = fullfile(model_dir, name);
        elseif str2double(strrep(timestamp, '_', '')) > str2double(strrep(contrast_dirs.(contrast_name).timestamp, '_', ''))
            contrast_dirs.(contrast_name).timestamp = timestamp;
            contrast_dirs.(contrast_name).path = fullfile(model_dir, name);
        end
    end
end

% Process each contrast
contrast_names = fieldnames(contrast_dirs);
results = struct('name', {}, 'n_clusters', {}, 'n_voxels', {}, ...
                'peak_val', {}, 'k_threshold', {});

for i = 1:length(contrast_names)
    contrast_name = contrast_names{i};
    spm_dir = contrast_dirs.(contrast_name).path;
    spm_path = fullfile(spm_dir, 'SPM.mat');

    if ~exist(spm_path, 'file')
        fprintf('SPM.mat not found: %s\n', spm_path);
        continue;
    end

    fprintf('\nProcessing: %s\n', contrast_name);
    fprintf('  Directory: %s\n', spm_dir);

    % Load SPM
    load(spm_path, 'SPM');

    % Get stat file (T-contrast, con 1)
    stat_file = fullfile(spm_dir, 'spmT_0001.nii');
    if ~exist(stat_file, 'file')
        fprintf('  Stat file not found, skipping\n');
        continue;
    end

    % Run cluster analysis with proper RFT
    [n_clusters, n_voxels, peak_val, k_thresh, thresh_img] = ...
        analyze_clusters_rft(SPM, spm_dir, 1, cluster_p, voxel_p);

    % Store results
    results(end+1).name = contrast_name;
    results(end).n_clusters = n_clusters;
    results(end).n_voxels = n_voxels;
    results(end).peak_val = peak_val;
    results(end).k_threshold = k_thresh;

    fprintf('  Extent threshold (k): %d voxels\n', k_thresh);
    fprintf('  Significant clusters: %d, Total voxels: %d\n', n_clusters, n_voxels);

    % Save thresholded image
    if n_clusters > 0 && ~isempty(thresh_img)
        safe_name = matlab.lang.makeValidName(contrast_name);
        out_file = fullfile(output_dir, [safe_name, '_clusterFWE.nii']);

        V = spm_vol(stat_file);
        V.fname = out_file;
        V.dt = [spm_type('float32') spm_platform('bigend')];
        spm_write_vol(V, thresh_img);
        fprintf('  Saved: %s\n', out_file);
    end
end

% Save summary
save_summary(output_dir, results, model_name, cluster_p, voxel_p);

fprintf('\n========================================\n');
fprintf('Done!\n');
fprintf('========================================\n');

end


function [n_clusters, n_voxels, peak_val, k_thresh, thresh_img] = ...
    analyze_clusters_rft(SPM, spm_dir, con_idx, cluster_p, voxel_p)
%% Analyze clusters using SPM's proper RFT calculation

n_clusters = 0;
n_voxels = 0;
peak_val = NaN;
k_thresh = 0;
thresh_img = [];

% Get stat file (T-statistic)
stat_file = fullfile(spm_dir, sprintf('spmT_%04d.nii', con_idx));
if ~exist(stat_file, 'file')
    return;
end

% Load stat image
V = spm_vol(stat_file);
Y = spm_read_vols(V);

% Get degrees of freedom for T-statistic
df = [1, SPM.xX.erdf];

% Calculate cluster-forming threshold
u = spm_invTcdf(1 - voxel_p, df(2));
fprintf('  Cluster-forming threshold: T = %.3f (p < %.4f unc.)\n', u, voxel_p);

% Get RESEL information from SPM
R = SPM.xVol.R;       % RESEL counts [0D, 1D, 2D, 3D]
S = SPM.xVol.S;       % Search volume in voxels
FWHM = SPM.xVol.FWHM; % Smoothness in voxels

fprintf('  Smoothness (FWHM): [%.2f, %.2f, %.2f] voxels\n', FWHM(1), FWHM(2), FWHM(3));
fprintf('  Search volume: %d voxels\n', S);

% Calculate cluster extent threshold using SPM's RFT
% Note: spm_P expects k in RESELS, so we need to pass FWHM for conversion
k_thresh = calculate_cluster_extent_threshold(u, df, 'T', R, cluster_p, S, FWHM);

if isinf(k_thresh)
    fprintf('  No cluster size would be significant at FWE p < %.3f\n', cluster_p);
    return;
end

fprintf('  RFT cluster extent threshold (k): %d voxels\n', k_thresh);

% Apply cluster-forming threshold
mask = Y > u;

if ~any(mask(:))
    fprintf('  No voxels survive cluster-forming threshold\n');
    return;
end

% Find clusters
[L, num_clusters] = spm_bwlabel(double(mask), 18);

if num_clusters == 0
    return;
end

% Calculate cluster sizes
cluster_sizes = zeros(num_clusters, 1);
for c = 1:num_clusters
    cluster_sizes(c) = sum(L(:) == c);
end

% Find significant clusters
sig_idx = cluster_sizes >= k_thresh;
n_clusters = sum(sig_idx);

fprintf('  Total clusters: %d, Significant (k >= %d): %d\n', ...
        num_clusters, k_thresh, n_clusters);

if n_clusters == 0
    return;
end

% Create thresholded image with only significant clusters
thresh_img = zeros(size(Y));
sig_cluster_ids = find(sig_idx);
for i = 1:length(sig_cluster_ids)
    c = sig_cluster_ids(i);
    cluster_mask = (L == c);
    thresh_img(cluster_mask) = Y(cluster_mask);
end

n_voxels = sum(thresh_img(:) > 0);
peak_val = max(thresh_img(:));

end


function k = calculate_cluster_extent_threshold(u, df, STAT, R, alpha, S, FWHM)
%% Calculate cluster extent threshold for FWE correction using SPM's spm_P
%
% Uses binary search with spm_P to find the minimum cluster size k (in voxels)
% such that P_FWE(cluster >= k) < alpha
%
% IMPORTANT: spm_P expects k in RESELS, not voxels!
% Conversion: k_resels = k_voxels / prod(FWHM)
%
% Parameters:
%   u     - cluster-forming threshold (T or F value)
%   df    - degrees of freedom [df1, df2]
%   STAT  - 'T' or 'F'
%   R     - RESEL counts [R0, R1, R2, R3]
%   alpha - FWE threshold (e.g., 0.05)
%   S     - search volume in voxels
%   FWHM  - smoothness in voxels [x, y, z]
%
% Returns:
%   k     - minimum cluster size in VOXELS for FWE significance

% Voxels to RESELs conversion factor
V2R = 1 / prod(FWHM);
fprintf('  Voxel-to-RESEL factor: %.6f (1 voxel = %.4f resels)\n', V2R, V2R);

% Binary search for k in voxels
k_min = 1;
k_max = 5000;  % Maximum reasonable cluster size in voxels

% First check if even the largest cluster is not significant
k_resels = k_max * V2R;
[P_max, ~] = spm_P(1, k_resels, u, df, STAT, R, 1, S);
if isnan(P_max) || P_max >= alpha
    % Even 5000 voxels not significant, no reasonable threshold exists
    k = Inf;
    return;
end

% Check if k=1 is already significant (unlikely with correct units)
k_resels = k_min * V2R;
[P_min, ~] = spm_P(1, k_resels, u, df, STAT, R, 1, S);
if ~isnan(P_min) && P_min < alpha
    k = 1;
    return;
end

% Binary search: find minimum k (in voxels) where P < alpha
for iter = 1:50
    k = round((k_min + k_max) / 2);
    k_resels = k * V2R;
    [P, ~] = spm_P(1, k_resels, u, df, STAT, R, 1, S);

    if isnan(P)
        % Handle NaN - try smaller k
        k_max = k;
    elseif P < alpha
        % k is significant, try smaller
        k_max = k;
    else
        % k is not significant, try larger
        k_min = k;
    end

    if k_max - k_min <= 1
        break;
    end
end

% Return the minimum significant k (in voxels)
k = k_max;

% Verify the result
k_resels = k * V2R;
[P_final, ~] = spm_P(1, k_resels, u, df, STAT, R, 1, S);
fprintf('  Verified: k=%d voxels (%.4f resels), P_FWE=%.4f\n', k, k_resels, P_final);

end


function save_summary(output_dir, results, model_name, cluster_p, voxel_p)
%% Save summary files

% Text summary
summary_file = fullfile(output_dir, 'cluster_fwe_summary.txt');
fid = fopen(summary_file, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'Cluster-level FWE Results (proper RFT)\n');
fprintf(fid, '========================================\n');
fprintf(fid, 'Model: %s\n', model_name);
fprintf(fid, 'Cluster FWE: p < %.3f\n', cluster_p);
fprintf(fid, 'Cluster-forming: p < %.4f (uncorrected)\n', voxel_p);
fprintf(fid, '========================================\n\n');

fprintf(fid, '%-30s %8s %10s %10s %10s\n', 'Contrast', 'k_thresh', 'Clusters', 'Voxels', 'Peak T');
fprintf(fid, '%s\n', repmat('-', 1, 70));

for i = 1:length(results)
    r = results(i);
    if isnan(r.peak_val)
        fprintf(fid, '%-30s %8d %10d %10d %10s\n', r.name, r.k_threshold, r.n_clusters, r.n_voxels, 'N/A');
    else
        fprintf(fid, '%-30s %8d %10d %10d %10.2f\n', r.name, r.k_threshold, r.n_clusters, r.n_voxels, r.peak_val);
    end
end

fclose(fid);
fprintf('\nSummary saved: %s\n', summary_file);

% CSV
csv_file = fullfile(output_dir, 'cluster_fwe_results.csv');
T = table();
for i = 1:length(results)
    row = results(i);
    T = [T; table({row.name}, row.n_clusters, row.n_voxels, row.peak_val, row.k_threshold, ...
                  'VariableNames', {'name', 'n_clusters', 'n_voxels', 'peak_val', 'k_threshold'})];
end
writetable(T, csv_file);
fprintf('CSV saved: %s\n', csv_file);

end
