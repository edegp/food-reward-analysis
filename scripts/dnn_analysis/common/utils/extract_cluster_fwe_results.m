function extract_cluster_fwe_results(spm_dir, output_dir, varargin)
%% Extract cluster-level FWE corrected results from SPM second-level analysis
%
% Usage:
%   extract_cluster_fwe_results(spm_dir, output_dir)
%   extract_cluster_fwe_results(spm_dir, output_dir, 'cluster_p', 0.05, 'voxel_p', 0.001)
%
% Parameters:
%   spm_dir: Directory containing SPM.mat
%   output_dir: Output directory for thresholded images
%   'cluster_p': Cluster-level FWE p-value threshold (default: 0.05)
%   'voxel_p': Cluster-forming voxel-level p-value (default: 0.001 uncorrected)

% Parse inputs
p = inputParser;
addRequired(p, 'spm_dir');
addRequired(p, 'output_dir');
addParameter(p, 'cluster_p', 0.05);
addParameter(p, 'voxel_p', 0.001);
parse(p, spm_dir, output_dir, varargin{:});

cluster_p_threshold = p.Results.cluster_p;
voxel_p_threshold = p.Results.voxel_p;

% Initialize SPM
spm('defaults', 'fmri');
spm_jobman('initcfg');

% Load SPM.mat
spm_path = fullfile(spm_dir, 'SPM.mat');
if ~exist(spm_path, 'file')
    error('SPM.mat not found: %s', spm_path);
end

load(spm_path, 'SPM');

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Results table
results_table = {};

fprintf('\n========================================\n');
fprintf('Cluster-level FWE Extraction\n');
fprintf('========================================\n');
fprintf('SPM directory: %s\n', spm_dir);
fprintf('Cluster FWE p < %.3f\n', cluster_p_threshold);
fprintf('Cluster-forming voxel p < %.3f (uncorrected)\n', voxel_p_threshold);
fprintf('========================================\n\n');

% Process each contrast
n_contrasts = length(SPM.xCon);

for ic = 1:n_contrasts
    con_name = SPM.xCon(ic).name;
    con_type = SPM.xCon(ic).STAT;  % 'T' or 'F'

    fprintf('Processing: %s (%s-contrast)\n', con_name, con_type);

    % Get the statistical image
    if strcmp(con_type, 'T')
        stat_file = fullfile(spm_dir, sprintf('spmT_%04d.nii', ic));
    else
        stat_file = fullfile(spm_dir, sprintf('spmF_%04d.nii', ic));
    end

    if ~exist(stat_file, 'file')
        fprintf('  Warning: %s not found\n', stat_file);
        continue;
    end

    % Set up xSPM structure for spm_getSPM
    xSPM = [];
    xSPM.swd = spm_dir;
    xSPM.Ic = ic;
    xSPM.Im = [];  % No masking contrast
    xSPM.pm = [];
    xSPM.Ex = [];
    xSPM.title = con_name;

    % Thresholds
    xSPM.u = voxel_p_threshold;  % Cluster-forming threshold (p-value)
    xSPM.thresDesc = 'none';     % Uncorrected for cluster-forming
    xSPM.k = 0;                   % No extent threshold yet

    % Get results
    try
        [SPM_result, xSPM] = spm_getSPM(xSPM);
    catch ME
        fprintf('  Error getting SPM results: %s\n', ME.message);
        continue;
    end

    if isempty(xSPM.Z)
        fprintf('  No suprathreshold voxels at cluster-forming threshold\n');
        results_table{end+1} = {con_name, con_type, 0, 0, NaN, NaN};
        continue;
    end

    % Get cluster information with FWE correction
    % xSPM.XYZ contains voxel coordinates
    % xSPM.Z contains statistical values

    % Calculate cluster-level statistics
    [x, y, z] = ind2sub(SPM.xVol.DIM', find(xSPM.Z > 0));

    % Use SPM's cluster analysis
    % Get cluster table
    TabDat = spm_list('List', xSPM, SPM.xVol, 1);  % 1 = print table

    % Find clusters surviving FWE correction
    if isfield(xSPM, 'Pc') && ~isempty(xSPM.Pc)
        % Cluster p-values (FWE corrected)
        cluster_pFWE = xSPM.Pc;
        sig_clusters = find(cluster_pFWE < cluster_p_threshold);
        n_sig_clusters = length(sig_clusters);
    else
        n_sig_clusters = 0;
        sig_clusters = [];
    end

    fprintf('  Clusters surviving FWE p < %.3f: %d\n', cluster_p_threshold, n_sig_clusters);

    % Create thresholded image with only significant clusters
    if n_sig_clusters > 0
        % Get cluster assignments
        A = xSPM.A;  % Cluster assignment for each voxel

        % Create mask for significant clusters
        sig_mask = ismember(A, sig_clusters);

        % Load original stat image
        V = spm_vol(stat_file);
        Y = spm_read_vols(V);

        % Apply mask
        Y_thresh = zeros(size(Y));

        % Map XYZ coordinates back to image
        XYZ = xSPM.XYZ;
        Z_vals = xSPM.Z;

        for iv = 1:size(XYZ, 2)
            if sig_mask(iv)
                Y_thresh(XYZ(1,iv), XYZ(2,iv), XYZ(3,iv)) = Z_vals(iv);
            end
        end

        % Count significant voxels
        n_sig_voxels = sum(sig_mask);

        % Get peak value
        peak_val = max(Z_vals(sig_mask));

        % Save thresholded image
        safe_name = strrep(strrep(con_name, ' ', '_'), ':', '');
        safe_name = strrep(safe_name, '>', 'gt');
        out_file = fullfile(output_dir, sprintf('%s_clusterFWE.nii', safe_name));

        V_out = V;
        V_out.fname = out_file;
        V_out.dt = [spm_type('float32') spm_platform('bigend')];
        spm_write_vol(V_out, Y_thresh);

        fprintf('  Significant voxels: %d\n', n_sig_voxels);
        fprintf('  Peak value: %.2f\n', peak_val);
        fprintf('  Saved: %s\n', out_file);

        results_table{end+1} = {con_name, con_type, n_sig_clusters, n_sig_voxels, peak_val, cluster_p_threshold};
    else
        fprintf('  No significant clusters\n');
        results_table{end+1} = {con_name, con_type, 0, 0, NaN, NaN};
    end
end

% Save results summary
summary_file = fullfile(output_dir, 'cluster_fwe_summary.txt');
fid = fopen(summary_file, 'w');
fprintf(fid, 'Cluster-level FWE Results Summary\n');
fprintf(fid, '==================================\n');
fprintf(fid, 'SPM directory: %s\n', spm_dir);
fprintf(fid, 'Cluster FWE p < %.3f\n', cluster_p_threshold);
fprintf(fid, 'Cluster-forming voxel p < %.3f (uncorrected)\n\n', voxel_p_threshold);
fprintf(fid, '%-30s %5s %8s %10s %10s\n', 'Contrast', 'Type', 'Clusters', 'Voxels', 'Peak');
fprintf(fid, '%s\n', repmat('-', 1, 70));

for i = 1:length(results_table)
    r = results_table{i};
    if isnan(r{5})
        fprintf(fid, '%-30s %5s %8d %10d %10s\n', r{1}, r{2}, r{3}, r{4}, 'N/A');
    else
        fprintf(fid, '%-30s %5s %8d %10d %10.2f\n', r{1}, r{2}, r{3}, r{4}, r{5});
    end
end

fclose(fid);
fprintf('\nSummary saved: %s\n', summary_file);

end
