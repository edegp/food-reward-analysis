%% Extract Beta RMS for ROI Analysis
% Extract beta values from first-level results, calculate RMS per layer,
% and compute mean ± SEM across subjects for visualization
%
% Output: CSV with subject-level RMS values for each ROI and layer

clear; clc;

%% Setup paths
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');

% ROI directory
roi_dir = fullfile(root_dir, 'rois', 'HarvardOxford');

% Output directory
output_dir = fullfile(root_dir, 'results', 'dnn_analysis', 'roi_analysis', 'beta_rms');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% First-level results base
first_level_base = '/Volumes/Extreme Pro/hit/food-brain/results/first_level_analysis';

%% Define sources and layer structure
sources = {'clip', 'convnext'};

% Layer definitions: name, pattern to match in regressor names
layer_defs = {
    'Initial_only', {'Initial_pc'};
    'Middle_only', {'Middle_pc'};
    'Late_only', {'Late_pc'};
    'Final_only', {'Final_pc'};
    'Shared_F', {'Shared_Initial_Middle_pc', 'Shared_Middle_Late_pc', 'Shared_Late_Final_pc'};
    'Shared_Initial_Middle', {'Shared_Initial_Middle_pc'};
    'Shared_Middle_Late', {'Shared_Middle_Late_pc'};
    'Shared_Late_Final', {'Shared_Late_Final_pc'};
    'Global_F', {'Global_pc'};
    % Combined layers with their associated shared PCs
    'Initial_withShared', {'Initial_pc', 'Shared_Initial_Middle_pc'};
    'Middle_withShared', {'Middle_pc', 'Shared_Initial_Middle_pc', 'Shared_Middle_Late_pc'};
    'Late_withShared', {'Late_pc', 'Shared_Middle_Late_pc', 'Shared_Late_Final_pc'};
    'Final_withShared', {'Final_pc', 'Shared_Late_Final_pc'};
    % Combined layers with shared PCs and Global
    'Initial_withShared_withGlobal', {'Initial_pc', 'Shared_Initial_Middle_pc', 'Global_pc'};
    'Middle_withShared_withGlobal', {'Middle_pc', 'Shared_Initial_Middle_pc', 'Shared_Middle_Late_pc', 'Global_pc'};
    'Late_withShared_withGlobal', {'Late_pc', 'Shared_Middle_Late_pc', 'Shared_Late_Final_pc', 'Global_pc'};
    'Final_withShared_withGlobal', {'Final_pc', 'Shared_Late_Final_pc', 'Global_pc'};
};

%% Find subjects
subject_dirs = dir(fullfile(first_level_base, 'sub-*'));
n_subjects = length(subject_dirs);
fprintf('Found %d subjects\n', n_subjects);

%% Find ROI masks
roi_files = dir(fullfile(roi_dir, '*_mask.nii'));
if isempty(roi_files)
    roi_files = dir(fullfile(roi_dir, '*_mask.nii.gz'));
end
n_rois = length(roi_files);
fprintf('Found %d ROI masks\n', n_rois);

% Load ROI masks once and store voxel coordinates in mm
roi_xyz_mm = cell(n_rois, 1);
roi_names = cell(n_rois, 1);
for r = 1:n_rois
    roi_file = fullfile(roi_files(r).folder, roi_files(r).name);
    V_roi = spm_vol(roi_file);
    Y_roi = spm_read_vols(V_roi);
    [i, j, k] = ind2sub(size(Y_roi), find(Y_roi > 0));
    % Convert to mm coordinates
    xyz_mm = V_roi.mat * [i'; j'; k'; ones(1, length(i))];
    roi_xyz_mm{r} = xyz_mm(1:3, :);
    roi_names{r} = strrep(roi_files(r).name, '_mask.nii', '');
    roi_names{r} = strrep(roi_names{r}, '.gz', '');
    fprintf('  %s: %d voxels\n', roi_names{r}, length(i));
end

%% Process each source
for s = 1:length(sources)
    source = sources{s};
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('Processing %s\n', upper(source));
    fprintf('%s\n', repmat('=', 1, 60));

    glm_name = sprintf('glm_dnn_pmods_hierarchical_%s_daywise', source);

    % Results storage
    all_results = {};

    %% Process each subject
    for subj = 1:n_subjects
        subject_id = subject_dirs(subj).name;
        fprintf('\n%s: ', subject_id);

        % Find GLM directory
        glm_dir = fullfile(first_level_base, subject_id, 'glm_model', glm_name);
        if ~exist(glm_dir, 'dir')
            fprintf('GLM not found\n');
            continue;
        end

        % Find latest run
        runs = dir(glm_dir);
        runs = runs([runs.isdir] & ~startsWith({runs.name}, '.'));
        if isempty(runs)
            fprintf('No runs found\n');
            continue;
        end
        run_dir = fullfile(glm_dir, runs(end).name);

        % Load SPM.mat
        spm_mat_file = fullfile(run_dir, 'SPM.mat');
        if ~exist(spm_mat_file, 'file')
            fprintf('SPM.mat not found\n');
            continue;
        end
        load(spm_mat_file, 'SPM');

        % Get regressor names
        reg_names = SPM.xX.name;
        n_regs = length(reg_names);

        %% Process each layer
        for L = 1:size(layer_defs, 1)
            layer_name = layer_defs{L, 1};
            patterns = layer_defs{L, 2};

            % Find matching beta indices
            beta_indices = [];
            for p = 1:length(patterns)
                pattern = patterns{p};
                for i = 1:n_regs
                    if contains(reg_names{i}, pattern)
                        beta_indices = [beta_indices, i];
                    end
                end
            end

            if isempty(beta_indices)
                fprintf('No betas found for %s\n', layer_name);
                continue;
            end

            n_pcs = length(beta_indices);

            % Get reference volume for coordinate transformation
            first_beta_file = fullfile(run_dir, sprintf('beta_%04d.nii', beta_indices(1)));
            if ~exist(first_beta_file, 'file')
                fprintf('Beta file not found: %s\n', first_beta_file);
                continue;
            end
            V_first = spm_vol(first_beta_file);

            %% Extract ROI values using spm_sample_vol (fast)
            for r = 1:n_rois
                xyz_mm = roi_xyz_mm{r};
                n_vox = size(xyz_mm, 2);

                if n_vox == 0
                    mean_rms = NaN;
                else
                    % Convert mm to voxel coordinates in beta space
                    xyz_vox = V_first.mat \ [xyz_mm; ones(1, n_vox)];

                    % Sample beta values at ROI voxels
                    beta_vals = zeros(n_vox, n_pcs);
                    for b = 1:n_pcs
                        beta_file = fullfile(run_dir, sprintf('beta_%04d.nii', beta_indices(b)));
                        V_beta = spm_vol(beta_file);
                        beta_vals(:, b) = spm_sample_vol(V_beta, xyz_vox(1,:), xyz_vox(2,:), xyz_vox(3,:), 1);
                    end

                    % Calculate RMS per voxel, then mean across ROI
                    rms_per_voxel = sqrt(mean(beta_vals.^2, 2));
                    valid = ~isnan(rms_per_voxel) & rms_per_voxel > 0;
                    if any(valid)
                        mean_rms = mean(rms_per_voxel(valid));
                    else
                        mean_rms = NaN;
                    end
                end

                % Store result
                all_results{end+1, 1} = source;
                all_results{end, 2} = subject_id;
                all_results{end, 3} = layer_name;
                all_results{end, 4} = roi_names{r};
                all_results{end, 5} = mean_rms;
                all_results{end, 6} = n_pcs;
            end

            fprintf('.');
        end
        fprintf(' done\n');
    end

    %% Save results
    if ~isempty(all_results)
        T = cell2table(all_results, 'VariableNames', ...
            {'source', 'subject', 'layer', 'roi', 'rms', 'n_pcs'});

        csv_file = fullfile(output_dir, sprintf('%s_beta_rms.csv', source));
        writetable(T, csv_file);
        fprintf('\nSaved: %s\n', csv_file);
    end
end

fprintf('\nDone!\n');
