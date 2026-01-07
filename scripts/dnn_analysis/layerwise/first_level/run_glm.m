function run_model_dnn_glm_designI(sub_id, source_label)
%% Design I: Low-Correlation 3-Layer Analysis
%
% Uses 3 layers with low inter-layer correlation (|r| < 0.5):
% - CLIP: stage1_1 (Early), stage2_22 (Middle), stage3_2 (Late)
% - ConvNeXt: features_1_0 (Early), features_5_22 (Middle), features_7_0 (Late)
%
% Structure:
% - ImageOnset_S01-S12: Session-specific with 9 pmods (3 layers × 3 PCs)
% - Question_S01-S12: Session-specific nuisance
% - Response_S01-S12: Session-specific nuisance
% - Miss_S##: Where applicable
% - SessionConst_S02-S12: Session constants
% - Motion regressors per session
%
% Usage:
%   run_model_dnn_glm_designI('031', 'clip')
%   run_model_dnn_glm_designI('031', 'convnext')

clearvars -except sub_id source_label; close all; clc;

maxNumCompThreads(1);

if nargin < 1 || isempty(sub_id)
    error('Subject ID required. Usage: run_model_dnn_glm_designI(''031'', ''clip'')');
end
if nargin < 2 || isempty(source_label)
    error('Source label required (''clip'' or ''convnext'')');
end

if isnumeric(sub_id)
    sub_id = sprintf('%03d', sub_id);
end

project_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(project_dir, '..', '..', '..', '..');

addpath(fullfile(root_dir, 'scripts', 'suzuki'));
addpath(fullfile(root_dir, 'scripts', 'common', 'first_level'));
addpath(project_dir);
addpath(fullfile(project_dir, '..', 'first_level_contrasts'));

% Define low-correlation layers for each model
if strcmp(source_label, 'clip')
    selected_layers = {
        struct('name', 'Early', 'module', 'clip.stage1_1'),
        struct('name', 'Middle', 'module', 'clip.stage2_22'),
        struct('name', 'Late', 'module', 'clip.stage3_2')
    };
elseif strcmp(source_label, 'convnext')
    selected_layers = {
        struct('name', 'Early', 'module', 'convnext.features_1_0_block_4'),
        struct('name', 'Middle', 'module', 'convnext.features_5_22_block_4'),
        struct('name', 'Late', 'module', 'convnext.features_7_0_block_4')
    };
else
    error('Unknown source_label: %s', source_label);
end

num_pcs = 3;  % Use PC1-3 for each layer

% Load per-layer DNN PCs
pmod_csv_path = fullfile(root_dir, 'data_images', 'dnn_pmods', 'per_layer', ...
    [source_label, '_pcs.csv']);
if ~exist(pmod_csv_path, 'file')
    error('Per-layer PC file not found: %s', pmod_csv_path);
end

pc_table = readtable(pmod_csv_path);
fprintf('Loaded %d rows from %s\n', height(pc_table), pmod_csv_path);

fprintf('\n========================================\n');
fprintf('Design I: Subject %s - %s\n', sub_id, upper(source_label));
fprintf('========================================\n');
fprintf('Low-Correlation 3-Layer Analysis:\n');
for i = 1:length(selected_layers)
    fprintf('  %s: %s\n', selected_layers{i}.name, selected_layers{i}.module);
end
fprintf('PCs per layer: %d\n\n', num_pcs);

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% Setup paths
sub_02 = sprintf('%02d', str2double(sub_id));
sub_dir = fullfile(root_dir, 'fMRIprep', 'need_info', ['sub-', sub_02]);
image_dir = fullfile(root_dir, 'fMRIprep', 'smoothed', ['sub-', sub_02]);
model_dir = fullfile(root_dir, 'results', 'first_level_analysis', ['sub-', sub_id], 'glm_model');

% fMRIprep derivatives directory for brain mask (same as Design E)
deriv_dir = fullfile(root_dir, 'fMRIprep', 'derivatives', ['sub-', sub_02], 'anat');
fname_mask_gz = fullfile(deriv_dir, ['sub-', sub_02, '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz']);
fname_mask_nii = fullfile(deriv_dir, ['sub-', sub_02, '_space-MNI152NLin2009cAsym_desc-brain_mask.nii']);

% Check if .nii exists, otherwise gunzip from .nii.gz
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

model_name = ['glm_dnn_pmods_designI_', source_label];
nowstr = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
outdir = fullfile(model_dir, model_name, nowstr);
if ~exist(outdir, 'dir'); mkdir(outdir); end

% Session/run configuration
sess_list = {'01', '02', '03'};
runs_default = {'1', '2', '3', '4'};
TR = 0.8;

% Collect all data
fprintf('Step 1: Collecting data from all sessions...\n');

all_scans = {};
session_data = struct();
session_scan_counts = [];

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

session_cnt = 0;
cumulative_time = 0;

for s = 1:length(sess_list)
    if strcmp(sub_id, '016') && strcmp(sess_list{s}, '03')
        runs = {'1', '2'};
    else
        runs = runs_default;
    end

    for r = 1:length(runs)
        session_cnt = session_cnt + 1;

        scan_pattern = ['^ssub-', sub_02, '_ses-', sess_list{s}, '_task-pt_run-0', runs{r}, '_space-MNI152NLin2009cAsym_desc-preproc_bold\.nii$'];
        data_fMRI = cellstr(spm_select('ExtFPList', fullfile(image_dir, ['ses-', sess_list{s}], 'func'), scan_pattern, Inf));

        if isempty(data_fMRI) || isempty(data_fMRI{1})
            error('No scans found for session %s run %s', sess_list{s}, runs{r});
        end

        num_scans = length(data_fMRI);
        session_scan_counts(end+1) = num_scans;
        all_scans = [all_scans; data_fMRI];

        time_filepath = fullfile(beh_dir, files_Time_list{session_cnt});
        rate_filepath = fullfile(beh_dir, files_Rate_list{session_cnt});
        data_tmp = make_behav_csv_run(sub_id, rate_filepath, time_filepath);

        idx_nomiss = ~(data_tmp.RatingValue == 0);
        idx_miss = ~idx_nomiss;

        session_data(session_cnt).image_onsets = data_tmp.image(idx_nomiss) + cumulative_time;
        session_data(session_cnt).question_onsets = data_tmp.question(idx_nomiss) + cumulative_time;
        session_data(session_cnt).question_durations = data_tmp.rating(idx_nomiss) - data_tmp.question(idx_nomiss);
        session_data(session_cnt).response_onsets = data_tmp.rating(idx_nomiss) + cumulative_time;

        % Keep image names as numeric for matching with per_layer CSV
        img_names = data_tmp.ImageName(idx_nomiss);
        if ~isnumeric(img_names)
            img_names = str2double(img_names);
        end
        session_data(session_cnt).image_names = img_names(:);

        if any(idx_miss)
            session_data(session_cnt).miss_onsets = data_tmp.image(idx_miss) + cumulative_time;
            session_data(session_cnt).miss_durations = data_tmp.rating(idx_miss) - data_tmp.image(idx_miss);
        else
            session_data(session_cnt).miss_onsets = [];
            session_data(session_cnt).miss_durations = [];
        end

        confounds = readtable(fullfile(sub_dir, ['ses-', sess_list{s}], 'func', ...
            ['sub-', sub_02, '_ses-', sess_list{s}, '_task-pt_run-0', runs{r}, '_desc-confounds_timeseries.tsv']), ...
            'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
        session_data(session_cnt).motion = [confounds.trans_x, confounds.trans_y, confounds.trans_z, ...
                                            confounds.rot_x, confounds.rot_y, confounds.rot_z];
        session_data(session_cnt).num_scans = num_scans;

        cumulative_time = cumulative_time + num_scans * TR;
    end
end

total_sessions = session_cnt;
total_scans = length(all_scans);
total_trials = sum(cellfun(@length, {session_data.image_onsets}));

fprintf('  Total sessions: %d\n', total_sessions);
fprintf('  Total scans: %d\n', total_scans);
fprintf('  Total trials (non-miss): %d\n\n', total_trials);

% Build SPM design matrix
fprintf('Step 2: Building SPM design matrix...\n');

matlabbatch{1}.spm.stats.fmri_spec.dir = {outdir};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 60;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 30;

matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans = all_scans;

cond_idx = 0;

% Step 3: Create session-specific ImageOnset conditions with layer pmods
fprintf('\nStep 3: Creating session-specific ImageOnset conditions...\n');

for ses = 1:total_sessions
    img_names = session_data(ses).image_names;
    num_trials = length(img_names);

    % Build pmod structure for this session
    pmod_struct = struct('name', {}, 'param', {}, 'poly', {});
    pmod_count = 0;

    for layer_idx = 1:length(selected_layers)
        layer_info = selected_layers{layer_idx};
        layer_name = layer_info.name;
        module_name = layer_info.module;

        % Get data for this layer
        module_rows = strcmp(string(pc_table.module_name), module_name);
        if ~any(module_rows)
            warning('No data for module %s', module_name);
            continue;
        end
        module_data = pc_table(module_rows, :);

        for pc = 1:num_pcs
            pc_col = sprintf('pc%d', pc);

            if ~ismember(pc_col, module_data.Properties.VariableNames)
                continue;
            end

            pc_values = zeros(num_trials, 1);

            for trial = 1:num_trials
                img_id = img_names(trial);  % Numeric image ID
                row_match = (module_data.image_id == img_id);  % Numeric comparison

                if any(row_match)
                    pc_values(trial) = module_data.(pc_col)(row_match);
                end
            end

            % Z-score normalization within session
            pc_z = (pc_values - mean(pc_values, 'omitnan')) ./ std(pc_values, 'omitnan');
            if any(~isfinite(pc_z))
                pc_z(~isfinite(pc_z)) = 0;
            end

            pmod_count = pmod_count + 1;
            pmod_struct(pmod_count).name = sprintf('%s_pc%d', layer_name, pc);
            pmod_struct(pmod_count).param = pc_z(:)';
            pmod_struct(pmod_count).poly = 1;
        end
    end

    % Add ImageOnset condition for this session
    cond_idx = cond_idx + 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).name = sprintf('ImageOnset_S%02d', ses);
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).onset = session_data(ses).image_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).pmod = pmod_struct;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).orth = 0;

    fprintf('  Session %02d: %d trials, %d pmods\n', ses, num_trials, pmod_count);
end

% Step 4: Session-specific Question conditions
fprintf('\nStep 4: Creating session-specific Question conditions...\n');
for ses = 1:total_sessions
    cond_idx = cond_idx + 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).name = sprintf('Question_S%02d', ses);
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).onset = session_data(ses).question_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).duration = session_data(ses).question_durations;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).pmod = struct('name', {}, 'param', {}, 'poly', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).orth = 1;
end

% Step 5: Session-specific Response conditions
fprintf('\nStep 5: Creating session-specific Response conditions...\n');
for ses = 1:total_sessions
    cond_idx = cond_idx + 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).name = sprintf('Response_S%02d', ses);
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).onset = session_data(ses).response_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).pmod = struct('name', {}, 'param', {}, 'poly', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).orth = 1;
end

% Step 6: Session-specific Miss conditions
fprintf('\nStep 6: Creating session-specific Miss conditions...\n');
miss_count = 0;
for ses = 1:total_sessions
    if ~isempty(session_data(ses).miss_onsets)
        cond_idx = cond_idx + 1;
        miss_count = miss_count + 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).name = sprintf('Miss_S%02d', ses);
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).onset = session_data(ses).miss_onsets;
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).duration = session_data(ses).miss_durations;
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).orth = 1;
    end
end
fprintf('  Miss conditions: %d\n', miss_count);
fprintf('  Total conditions: %d\n', cond_idx);

% Step 7: Custom regressors (session constants + motion)
fprintf('\nStep 7: Adding session constants and motion regressors...\n');

all_regressors = {};

% Session constants (S02-S12, S01 is reference)
for ses = 2:total_sessions
    reg_val = zeros(total_scans, 1);
    start_idx = sum(session_scan_counts(1:ses-1)) + 1;
    end_idx = sum(session_scan_counts(1:ses));
    reg_val(start_idx:end_idx) = 1;
    all_regressors{end+1} = struct('name', sprintf('SessionConst_S%02d', ses), 'val', reg_val);
end
fprintf('  Session constants: %d\n', total_sessions - 1);

% Motion regressors per session
motion_names = {'tx', 'ty', 'tz', 'rx', 'ry', 'rz'};
motion_count = 0;
for ses = 1:total_sessions
    for m = 1:6
        reg_val = zeros(total_scans, 1);
        start_idx = sum(session_scan_counts(1:ses-1)) + 1;
        end_idx = sum(session_scan_counts(1:ses));
        reg_val(start_idx:end_idx) = session_data(ses).motion(:, m);
        all_regressors{end+1} = struct('name', sprintf('%s_S%02d', motion_names{m}, ses), 'val', reg_val);
        motion_count = motion_count + 1;
    end
end
fprintf('  Motion regressors: %d\n', motion_count);
fprintf('  Total custom regressors: %d\n', length(all_regressors));

for r = 1:length(all_regressors)
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).regress(r).name = all_regressors{r}.name;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).regress(r).val = all_regressors{r}.val;
end

matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi = {''};
matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi_reg = {''};
matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf = 128;

matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0;
% Use fMRIPrep brain mask (same as Design E)
if ~isempty(fname_mask)
    matlabbatch{1}.spm.stats.fmri_spec.mask = {fname_mask};
else
    matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
end
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

% Model estimation
matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outdir, 'SPM.mat')};
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

% Run
fprintf('\nStep 8: Running SPM batch...\n');

batch_path = fullfile(outdir, 'batch');
if ~exist(batch_path, 'dir'); mkdir(batch_path); end
job_id = cfg_util('initjob', matlabbatch);
cfg_util('savejob', job_id, fullfile(batch_path, 'designI_batch.m'));
spm_jobman('run', matlabbatch);

clearvars matlabbatch;

% Create contrasts
fprintf('\nStep 9: Creating contrasts...\n');
spm_path = fullfile(outdir, 'SPM.mat');
try
    matlabbatch = build_designI_contrasts(spm_path);
    job_id = cfg_util('initjob', matlabbatch);
    cfg_util('savejob', job_id, fullfile(batch_path, 'designI_cons_batch.m'));
    spm_jobman('run', matlabbatch);
catch ME
    warning('Contrast creation failed: %s', ME.message);
end

% Update latest run marker
try
    model_root = fullfile(model_dir, model_name);
    if ~exist(model_root, 'dir'); mkdir(model_root); end
    fid = fopen(fullfile(model_root, 'latest_run.txt'), 'w');
    if fid ~= -1
        fprintf(fid, '%s\n', outdir);
        fclose(fid);
    end
catch
end

fprintf('\n========================================\n');
fprintf('Design I completed for subject %s - %s\n', sub_id, upper(source_label));
fprintf('Output: %s\n', outdir);
fprintf('========================================\n');

end
