function run_cluster_fwe(design_name, source_label)
%% Run cluster-level FWE analysis using proper SPM RFT calculation
%
% This version correctly calculates the minimum cluster size (extent threshold)
% based on data smoothness (RESEL) using SPM's RFT functions.
%
% Usage:
%   run_cluster_fwe_batch_v2('designO', 'clip')
%   run_cluster_fwe_batch_v2('hierarchical', 'convnext')

% Setup paths
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..', '..');
results_base = fullfile(root_dir, 'results', 'dnn_analysis', 'second_level');

% Find SPM directory
if contains(design_name, 'v3')
    spm_parent = fullfile(results_base, [strrep(design_name, '_v3', ''), '_', source_label, '_v3']);
else
    spm_parent = fullfile(results_base, [design_name, '_', source_label]);
end

% Find latest run
d = dir(spm_parent);
d = d([d.isdir] & ~startsWith({d.name}, '.'));
if isempty(d)
    error('No results found in: %s', spm_parent);
end
[~, idx] = max([d.datenum]);
spm_dir = fullfile(spm_parent, d(idx).name);

% Output directory
output_dir = fullfile(root_dir, 'results', 'dnn_analysis', 'cluster_fwe_v2', ...
                     [design_name, '_', source_label]);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('\n========================================\n');
fprintf('Cluster-level FWE Analysis (v2 - proper RFT)\n');
fprintf('========================================\n');
fprintf('Design: %s\n', design_name);
fprintf('Source: %s\n', source_label);
fprintf('SPM dir: %s\n', spm_dir);
fprintf('Output: %s\n', output_dir);
fprintf('========================================\n\n');

% Initialize SPM
spm('defaults', 'fmri');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);

% Check if this is a design with subdirectories (like Design N v3)
subdirs = dir(fullfile(spm_dir, '*_only'));
if ~isempty(subdirs)
    % Design N v3 style - each contrast in subdirectory
    process_subdirectory_design(spm_dir, output_dir, design_name, source_label);
else
    % Standard design - all contrasts in one SPM.mat
    process_standard_design(spm_dir, output_dir, design_name, source_label);
end

end


function process_standard_design(spm_dir, output_dir, design_name, source_label)
%% Process standard design (Design O, etc.)

spm_path = fullfile(spm_dir, 'SPM.mat');
load(spm_path, 'SPM');

% Parameters
cluster_p = 0.05;   % Cluster-level FWE threshold
voxel_p = 0.001;    % Cluster-forming threshold (uncorrected)

% Results storage
results = struct('name', {}, 'type', {}, 'n_clusters', {}, 'n_voxels', {}, ...
                'peak_val', {}, 'k_threshold', {}, 'cluster_sizes', {});

n_contrasts = length(SPM.xCon);

for ic = 1:n_contrasts
    con_name = SPM.xCon(ic).name;
    con_type = SPM.xCon(ic).STAT;

    fprintf('Processing: %s (%s)\n', con_name, con_type);

    % Get stat file
    if strcmp(con_type, 'T')
        stat_file = fullfile(spm_dir, sprintf('spmT_%04d.nii', ic));
    else
        stat_file = fullfile(spm_dir, sprintf('spmF_%04d.nii', ic));
    end

    if ~exist(stat_file, 'file')
        fprintf('  Skipped (file not found)\n');
        continue;
    end

    % Run cluster analysis with proper RFT
    [n_clusters, n_voxels, peak_val, k_thresh, cluster_sizes, thresh_img] = ...
        analyze_clusters_rft(SPM, spm_dir, ic, cluster_p, voxel_p);

    % Save result
    results(end+1).name = con_name;
    results(end).type = con_type;
    results(end).n_clusters = n_clusters;
    results(end).n_voxels = n_voxels;
    results(end).peak_val = peak_val;
    results(end).k_threshold = k_thresh;
    results(end).cluster_sizes = cluster_sizes;

    fprintf('  Extent threshold (k): %d voxels\n', k_thresh);
    fprintf('  Significant clusters: %d, Total voxels: %d\n', n_clusters, n_voxels);

    % Save thresholded image if significant
    if n_clusters > 0 && ~isempty(thresh_img)
        safe_name = matlab.lang.makeValidName(con_name);
        out_file = fullfile(output_dir, [safe_name, '_clusterFWE.nii']);

        V = spm_vol(stat_file);
        V.fname = out_file;
        V.dt = [spm_type('float32') spm_platform('bigend')];
        spm_write_vol(V, thresh_img);
        fprintf('  Saved: %s\n', out_file);
    end
end

% Save summary
save_summary(output_dir, results, design_name, source_label, cluster_p, voxel_p, SPM);

end


function process_subdirectory_design(spm_dir, output_dir, design_name, source_label)
%% Process design with subdirectories (Design N v3)

% Parameters
cluster_p = 0.05;
voxel_p = 0.001;

% Find all contrast subdirectories
subdirs = dir(spm_dir);
subdirs = subdirs([subdirs.isdir] & ~startsWith({subdirs.name}, '.'));

results = struct('name', {}, 'type', {}, 'n_clusters', {}, 'n_voxels', {}, ...
                'peak_val', {}, 'k_threshold', {}, 'cluster_sizes', {});

% For storing first SPM for summary
first_SPM = [];

for i = 1:length(subdirs)
    subdir_name = subdirs(i).name;
    subdir_path = fullfile(spm_dir, subdir_name);
    sub_spm_path = fullfile(subdir_path, 'SPM.mat');

    if ~exist(sub_spm_path, 'file')
        continue;
    end

    fprintf('Processing: %s\n', subdir_name);

    load(sub_spm_path, 'SPM');
    if isempty(first_SPM)
        first_SPM = SPM;
    end

    % Run cluster analysis with proper RFT (contrast 1 in each subdirectory)
    [n_clusters, n_voxels, peak_val, k_thresh, cluster_sizes, thresh_img] = ...
        analyze_clusters_rft(SPM, subdir_path, 1, cluster_p, voxel_p);

    results(end+1).name = subdir_name;
    results(end).type = 'F';
    results(end).n_clusters = n_clusters;
    results(end).n_voxels = n_voxels;
    results(end).peak_val = peak_val;
    results(end).k_threshold = k_thresh;
    results(end).cluster_sizes = cluster_sizes;

    fprintf('  Extent threshold (k): %d voxels\n', k_thresh);
    fprintf('  Significant clusters: %d, Total voxels: %d\n', n_clusters, n_voxels);

    % Save thresholded image
    if n_clusters > 0 && ~isempty(thresh_img)
        stat_file = fullfile(subdir_path, 'spmF_0001.nii');
        out_file = fullfile(output_dir, [subdir_name, '_clusterFWE.nii']);

        V = spm_vol(stat_file);
        V.fname = out_file;
        V.dt = [spm_type('float32') spm_platform('bigend')];
        spm_write_vol(V, thresh_img);
        fprintf('  Saved: %s\n', out_file);
    end
end

save_summary(output_dir, results, design_name, source_label, cluster_p, voxel_p, first_SPM);

end


function [n_clusters, n_voxels, peak_val, k_thresh, cluster_sizes, thresh_img] = ...
    analyze_clusters_rft(SPM, spm_dir, con_idx, cluster_p, voxel_p)
%% Analyze clusters using SPM's proper RFT calculation
%
% This function calculates the cluster extent threshold (k) based on
% the actual smoothness of the data estimated by SPM.

n_clusters = 0;
n_voxels = 0;
peak_val = NaN;
k_thresh = 0;
cluster_sizes = [];
thresh_img = [];

con_type = SPM.xCon(con_idx).STAT;

% Get stat file
if strcmp(con_type, 'T')
    stat_file = fullfile(spm_dir, sprintf('spmT_%04d.nii', con_idx));
else
    stat_file = fullfile(spm_dir, sprintf('spmF_%04d.nii', con_idx));
end

if ~exist(stat_file, 'file')
    return;
end

% Load stat image
V = spm_vol(stat_file);
Y = spm_read_vols(V);

% Get degrees of freedom
if strcmp(con_type, 'T')
    df = [1, SPM.xX.erdf];
    % Calculate cluster-forming threshold for T
    u = spm_invTcdf(1 - voxel_p, df(2));
else
    % F-statistic
    df = [SPM.xCon(con_idx).eidf, SPM.xX.erdf];
    % Calculate cluster-forming threshold for F
    u = spm_invFcdf(1 - voxel_p, df);
end

fprintf('  Cluster-forming threshold: %.3f (p < %.4f unc.)\n', u, voxel_p);

% Get RESEL information from SPM (based on estimated smoothness)
R = SPM.xVol.R;      % RESEL counts [0D, 1D, 2D, 3D]
S = SPM.xVol.S;      % Search volume in voxels
FWHM = SPM.xVol.FWHM; % Smoothness in voxels

fprintf('  Smoothness (FWHM): [%.2f, %.2f, %.2f] voxels\n', FWHM(1), FWHM(2), FWHM(3));
fprintf('  RESEL counts: [%.2f, %.2f, %.2f, %.2f]\n', R(1), R(2), R(3), R(4));
fprintf('  Search volume: %d voxels\n', S);

% Calculate the minimum cluster extent threshold (k) for FWE correction
% Using SPM's spm_P function with binary search
% Note: spm_P expects k in RESELS, so we need to pass FWHM for conversion
k_thresh = calculate_cluster_extent_threshold(u, df, con_type, R, cluster_p, S, FWHM);

if isinf(k_thresh)
    fprintf('  No cluster size would be significant at FWE p < %.3f\n', cluster_p);
    return;
end

fprintf('  Calculated extent threshold (k): %d voxels\n', k_thresh);

% Apply cluster-forming threshold
mask = Y > u;

if ~any(mask(:))
    fprintf('  No voxels survive cluster-forming threshold\n');
    return;
end

% Find clusters using spm_bwlabel
[L, num_clusters] = spm_bwlabel(double(mask), 18);

if num_clusters == 0
    return;
end

% Calculate cluster sizes
cluster_sizes_all = zeros(num_clusters, 1);
for c = 1:num_clusters
    cluster_sizes_all(c) = sum(L(:) == c);
end

% Find clusters that exceed the extent threshold
sig_idx = cluster_sizes_all >= k_thresh;
n_clusters = sum(sig_idx);

fprintf('  Total clusters found: %d, Significant (k >= %d): %d\n', ...
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
cluster_sizes = cluster_sizes_all(sig_idx);

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


function save_summary(output_dir, results, design_name, source_label, cluster_p, voxel_p, SPM)
%% Save summary file

summary_file = fullfile(output_dir, 'cluster_fwe_summary.txt');
fid = fopen(summary_file, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'Cluster-level FWE Results (v2 - proper RFT)\n');
fprintf(fid, '========================================\n');
fprintf(fid, 'Design: %s\n', design_name);
fprintf(fid, 'Source: %s\n', source_label);
fprintf(fid, 'Cluster FWE: p < %.3f\n', cluster_p);
fprintf(fid, 'Cluster-forming: p < %.3f (uncorrected)\n', voxel_p);
fprintf(fid, '========================================\n\n');

% Print smoothness info
if isfield(SPM, 'xVol') && isfield(SPM.xVol, 'FWHM')
    FWHM = SPM.xVol.FWHM;
    fprintf(fid, 'Smoothness (FWHM): [%.2f, %.2f, %.2f] voxels\n', FWHM(1), FWHM(2), FWHM(3));
    fprintf(fid, 'RESEL counts: [%.2f, %.2f, %.2f, %.2f]\n', ...
            SPM.xVol.R(1), SPM.xVol.R(2), SPM.xVol.R(3), SPM.xVol.R(4));
    fprintf(fid, 'Search volume: %d voxels\n\n', SPM.xVol.S);
end

fprintf(fid, '%-30s %6s %8s %10s %10s %10s\n', 'Contrast', 'Type', 'k_thresh', 'Clusters', 'Voxels', 'Peak');
fprintf(fid, '%s\n', repmat('-', 1, 80));

for i = 1:length(results)
    r = results(i);
    if isnan(r.peak_val)
        fprintf(fid, '%-30s %6s %8d %10d %10d %10s\n', r.name, r.type, r.k_threshold, r.n_clusters, r.n_voxels, 'N/A');
    else
        fprintf(fid, '%-30s %6s %8d %10d %10d %10.2f\n', r.name, r.type, r.k_threshold, r.n_clusters, r.n_voxels, r.peak_val);
    end
end

fclose(fid);
fprintf('\nSummary saved: %s\n', summary_file);

% Also save as CSV
csv_file = fullfile(output_dir, 'cluster_fwe_results.csv');

% Convert struct to table (excluding cluster_sizes array)
T = table();
for i = 1:length(results)
    row = results(i);
    T = [T; table({row.name}, {row.type}, row.n_clusters, row.n_voxels, ...
                  row.peak_val, row.k_threshold, ...
                  'VariableNames', {'name', 'type', 'n_clusters', 'n_voxels', 'peak_val', 'k_threshold'})];
end
writetable(T, csv_file);
fprintf('CSV saved: %s\n', csv_file);

end
