function run_glm(sub_id, source_label)
%% 3-Level Hierarchical DNN GLM with Separate Sessions
%
% Structure:
%   - 12 separate SPM sessions (3 days × 4 runs)
%   - Each session has:
%     - ImageOnset condition with PC pmods (Global + Shared + Layer-Specific)
%     - Question, Response, Miss conditions
%     - Motion regressors (6 parameters)
%
% Features:
%   - Each session estimates its own beta for each PC
%   - Contrasts average across sessions
%   - No rank deficiency issues
%
% Usage:
%   run_glm('001', 'clip')
%   run_glm('001', 'convnext')

clearvars -except sub_id source_label; close all; clc;

maxNumCompThreads(1);

if nargin < 1 || isempty(sub_id)
    error('Subject ID required. Usage: run_glm(''001'', ''clip'')');
end
if nargin < 2 || isempty(source_label)
    error('Source label required (''clip'' or ''convnext'')');
end

if isnumeric(sub_id)
    sub_id = sprintf('%03d', sub_id);
end

project_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(project_dir, '..', '..', '..', '..');

addpath(fullfile(root_dir, 'scripts', 'common', 'first_level'));
addpath(project_dir);

% Load 3-level PCs
pc3_path = fullfile(root_dir, 'data_images', 'dnn_pmods', '3level', ...
    [source_label, '_3level_pcs.csv']);
if ~exist(pc3_path, 'file')
    error('3-level PC file not found: %s', pc3_path);
end
pc3_table = readtable(pc3_path, 'VariableNamingRule', 'preserve');
fprintf('Loaded %d rows from 3-level PCs\n', height(pc3_table));

% Prepare PC data (indexed by image_id)
pc3_table.image_id_num = str2double(string(pc3_table.image_id));

% Identify PC columns by type
all_cols = pc3_table.Properties.VariableNames;

global_cols = all_cols(startsWith(all_cols, 'Global_pc'));
shared_cols = all_cols(startsWith(all_cols, 'Shared_'));

% Find layer-specific columns
layer_groups = {};
specific_cols = struct();
for c = 1:numel(all_cols)
    col = all_cols{c};
    if contains(col, '_pc') && ~startsWith(col, 'Global') && ~startsWith(col, 'Shared')
        parts = strsplit(col, '_pc');
        layer_name = parts{1};
        if startsWith(layer_name, [source_label, '.'])
            layer_name = strrep(layer_name, [source_label, '.'], '');
        end
        if ~ismember(layer_name, layer_groups)
            layer_groups{end+1} = layer_name;
            specific_cols.(layer_name) = {};
        end
        specific_cols.(layer_name){end+1} = col;
    end
end

num_global = numel(global_cols);
num_shared = numel(shared_cols);

fprintf('\n========================================\n');
fprintf('Hierarchical DNN GLM: Subject %s - %s\n', sub_id, upper(source_label));
fprintf('========================================\n');
fprintf('Global PCs: %d\n', num_global);
fprintf('Layer-Shared PCs: %d\n', num_shared);
fprintf('Layer-Specific PCs:\n');
total_specific = 0;
for g = 1:length(layer_groups)
    n = numel(specific_cols.(layer_groups{g}));
    fprintf('  %s: %d PCs\n', layer_groups{g}, n);
    total_specific = total_specific + n;
end
total_pmods = num_global + num_shared + total_specific;
fprintf('Total pmods per session: %d\n\n', total_pmods);

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% Setup paths
sub_02 = sprintf('%02d', str2double(sub_id));
sub_dir = fullfile(root_dir, 'fMRIprep', 'need_info', ['sub-', sub_02]);
image_dir = fullfile(root_dir, 'fMRIprep', 'smoothed', ['sub-', sub_02]);
model_dir = fullfile(root_dir, 'results', 'first_level_analysis', ['sub-', sub_id], 'glm_model');

% fMRIprep brain mask
deriv_dir = fullfile(root_dir, 'fMRIprep', 'derivatives', ['sub-', sub_02], 'anat');
fname_mask_gz = fullfile(deriv_dir, ['sub-', sub_02, '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz']);
fname_mask_nii = fullfile(deriv_dir, ['sub-', sub_02, '_space-MNI152NLin2009cAsym_desc-brain_mask.nii']);

if exist(fname_mask_nii, 'file')
    fname_mask = fname_mask_nii;
elseif exist(fname_mask_gz, 'file')
    fprintf('Gunzipping brain mask: %s\n', fname_mask_gz);
    gunzip(fname_mask_gz, deriv_dir);
    fname_mask = fname_mask_nii;
else
    warning('Brain mask not found: %s. Using default (no mask).', fname_mask_gz);
    fname_mask = '';
end

model_name = ['glm_dnn_pmods_hierarchical_', source_label, '_sessions'];
nowstr = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
outdir = fullfile(model_dir, model_name, nowstr);
if ~exist(outdir, 'dir'); mkdir(outdir); end

% Session/run configuration
sess_list = {'01', '02', '03'};
runs_default = {'1', '2', '3', '4'};
TR = 0.8;

% Behavior data directory
candidate_dirs = {fullfile(root_dir, 'Food_Behavior', ['sub-', sub_id]), ...
                  fullfile(root_dir, 'Food_Behavior', sub_id)};
beh_dir = '';
for cdix = 1:numel(candidate_dirs)
    if exist(candidate_dirs{cdix}, 'dir')
        beh_dir = candidate_dirs{cdix};
        break;
    end
end
if isempty(beh_dir)
    error('Behavior directory not found for subject %s', sub_id);
end

files = dir(fullfile(beh_dir, 'rating_data*.csv'));
[~, si] = sort([files.datenum]); files = files(si);
files_Rate_list = {files.name};

files = dir(fullfile(beh_dir, 'GLM_all*.csv'));
[~, si] = sort([files.datenum]); files = files(si);
files_Time_list = {files.name};

% Build SPM design matrix with separate sessions
fprintf('Building SPM design matrix with separate sessions...\n');

matlabbatch{1}.spm.stats.fmri_spec.dir = {outdir};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 60;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 30;

session_cnt = 0;
total_trials = 0;

for s = 1:length(sess_list)
    if strcmp(sub_id, '016') && strcmp(sess_list{s}, '03')
        runs = {'1', '2'};
    else
        runs = runs_default;
    end

    for r = 1:length(runs)
        session_cnt = session_cnt + 1;
        fprintf('\n--- Session %d (ses-%s, run-%s) ---\n', session_cnt, sess_list{s}, runs{r});

        % Get scans for this session
        scan_pattern = ['^ssub-', sub_02, '_ses-', sess_list{s}, '_task-pt_run-0', runs{r}, '_space-MNI152NLin2009cAsym_desc-preproc_bold\.nii$'];
        data_fMRI = cellstr(spm_select('ExtFPList', fullfile(image_dir, ['ses-', sess_list{s}], 'func'), scan_pattern, Inf));

        if isempty(data_fMRI) || isempty(data_fMRI{1})
            error('No scans found for session %s run %s', sess_list{s}, runs{r});
        end

        num_scans = length(data_fMRI);
        fprintf('  Scans: %d\n', num_scans);

        % Assign scans to this session
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).scans = data_fMRI;

        % Load behavior data
        time_filepath = fullfile(beh_dir, files_Time_list{session_cnt});
        rate_filepath = fullfile(beh_dir, files_Rate_list{session_cnt});
        data_tmp = make_behav_csv_run(sub_id, rate_filepath, time_filepath);

        idx_nomiss = ~(data_tmp.RatingValue == 0);
        idx_miss = ~idx_nomiss;

        % Onsets are relative to session start (not cumulative)
        image_onsets = data_tmp.image(idx_nomiss);
        question_onsets = data_tmp.question(idx_nomiss);
        question_durations = data_tmp.rating(idx_nomiss) - data_tmp.question(idx_nomiss);
        response_onsets = data_tmp.rating(idx_nomiss);

        img_names = data_tmp.ImageName(idx_nomiss);
        if ~isnumeric(img_names)
            img_names = str2double(img_names);
        end
        img_names = img_names(:);

        num_trials = length(img_names);
        total_trials = total_trials + num_trials;
        fprintf('  Trials: %d\n', num_trials);

        % Build pmod structure for this session
        pmod_struct = struct('name', {}, 'param', {}, 'poly', {});
        pmod_count = 0;

        % Global PCs
        for pc = 1:num_global
            col_name = global_cols{pc};
            pc_values = zeros(num_trials, 1);

            for trial = 1:num_trials
                img_id = img_names(trial);
                row_match = (pc3_table.image_id_num == img_id);
                if any(row_match)
                    pc_values(trial) = pc3_table.(col_name)(row_match);
                end
            end

            % Mean-center within session (SPM also does this, but explicit is clearer)
            pc_centered = pc_values - mean(pc_values, 'omitnan');
            if any(~isfinite(pc_centered))
                pc_centered(~isfinite(pc_centered)) = 0;
            end

            pmod_count = pmod_count + 1;
            pmod_struct(pmod_count).name = sprintf('Global_pc%d', pc);
            pmod_struct(pmod_count).param = pc_centered(:)';
            pmod_struct(pmod_count).poly = 1;
        end

        % Shared PCs
        for pc = 1:num_shared
            col_name = shared_cols{pc};
            pc_values = zeros(num_trials, 1);

            for trial = 1:num_trials
                img_id = img_names(trial);
                row_match = (pc3_table.image_id_num == img_id);
                if any(row_match)
                    pc_values(trial) = pc3_table.(col_name)(row_match);
                end
            end

            pc_centered = pc_values - mean(pc_values, 'omitnan');
            if any(~isfinite(pc_centered))
                pc_centered(~isfinite(pc_centered)) = 0;
            end

            pmod_count = pmod_count + 1;
            pmod_name = strrep(col_name, '-', '_');
            pmod_struct(pmod_count).name = pmod_name;
            pmod_struct(pmod_count).param = pc_centered(:)';
            pmod_struct(pmod_count).poly = 1;
        end

        % Layer-Specific PCs
        for g = 1:length(layer_groups)
            group_name = layer_groups{g};
            cols = specific_cols.(group_name);

            for pc = 1:length(cols)
                col_name = cols{pc};
                pc_values = zeros(num_trials, 1);

                for trial = 1:num_trials
                    img_id = img_names(trial);
                    row_match = (pc3_table.image_id_num == img_id);
                    if any(row_match)
                        pc_values(trial) = pc3_table.(col_name)(row_match);
                    end
                end

                pc_centered = pc_values - mean(pc_values, 'omitnan');
                if any(~isfinite(pc_centered))
                    pc_centered(~isfinite(pc_centered)) = 0;
                end

                pmod_count = pmod_count + 1;
                pmod_struct(pmod_count).name = sprintf('%s_pc%d', group_name, pc);
                pmod_struct(pmod_count).param = pc_centered(:)';
                pmod_struct(pmod_count).poly = 1;
            end
        end

        fprintf('  Pmods: %d\n', pmod_count);

        % Condition 1: ImageOnset with pmods
        cond_idx = 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).name = 'ImageOnset';
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).onset = image_onsets;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).duration = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).pmod = pmod_struct;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).orth = 0;

        % Condition 2: Question
        cond_idx = 2;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).name = 'Question';
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).onset = question_onsets;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).duration = question_durations;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).orth = 1;

        % Condition 3: Response
        cond_idx = 3;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).name = 'Response';
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).onset = response_onsets;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).duration = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).orth = 1;

        % Condition 4: Miss (if any)
        if any(idx_miss)
            miss_onsets = data_tmp.image(idx_miss);
            miss_durations = data_tmp.rating(idx_miss) - data_tmp.image(idx_miss);

            cond_idx = 4;
            matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).name = 'Miss';
            matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).onset = miss_onsets;
            matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).duration = miss_durations;
            matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).tmod = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).pmod = struct('name', {}, 'param', {}, 'poly', {});
            matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).cond(cond_idx).orth = 1;
        end

        % Motion regressors
        confounds = readtable(fullfile(sub_dir, ['ses-', sess_list{s}], 'func', ...
            ['sub-', sub_02, '_ses-', sess_list{s}, '_task-pt_run-0', runs{r}, '_desc-confounds_timeseries.tsv']), ...
            'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
        motion = [confounds.trans_x, confounds.trans_y, confounds.trans_z, ...
                  confounds.rot_x, confounds.rot_y, confounds.rot_z];

        mot_names = {'tx', 'ty', 'tz', 'rx', 'ry', 'rz'};
        for m = 1:6
            matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).regress(m).name = mot_names{m};
            matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).regress(m).val = motion(:, m);
        end

        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).multi = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).multi_reg = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_cnt).hpf = 128;
    end
end

total_sessions = session_cnt;
fprintf('\n========================================\n');
fprintf('Total sessions: %d\n', total_sessions);
fprintf('Total trials: %d\n', total_trials);
fprintf('========================================\n');

% Factorial design
matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0;
matlabbatch{1}.spm.stats.fmri_spec.mask = {fname_mask};
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

% Run model specification
fprintf('\nRunning model specification...\n');
spm_jobman('run', matlabbatch);
fprintf('SPM.mat saved to: %s\n', outdir);

% Estimation
fprintf('\nEstimating model...\n');
spm_mat = fullfile(outdir, 'SPM.mat');

matlabbatch = {};
matlabbatch{1}.spm.stats.fmri_est.spmmat = {spm_mat};
matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;

spm_jobman('run', matlabbatch);
fprintf('Estimation complete!\n');

% Build contrasts (averaging across sessions)
fprintf('\nBuilding contrasts (averaging across sessions)...\n');
build_contrasts_sessions(spm_mat, source_label, num_global, shared_cols, layer_groups, specific_cols, total_sessions);

fprintf('\n========================================\n');
fprintf('Hierarchical DNN GLM Complete: %s - %s\n', sub_id, upper(source_label));
fprintf('Output: %s\n', outdir);
fprintf('========================================\n');

end
