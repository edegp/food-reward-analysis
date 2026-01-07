function run_glm_daywise(sub_id, source_label)
%% Day-wise Hierarchical DNN GLM
% Design:
%   - 1 SPM session (all 12 runs concatenated)
%   - ImageOnset_Day1/2/3: Day-level conditions with 42 PMODs each
%   - ImageOnset_S02-S04, S06-S08, S10-S12: Run-specific baselines (no PMODs)
%   - Session-specific Question, Response, Miss conditions
%   - Session-specific motion regressors
%
% This gives:
%   - 12 effective run-specific baselines
%   - 3 × 42 = 126 day-level PMODs (SPM auto HRF convolution)
%
% Usage:
%   run_glm_daywise('001', 'clip')
%   run_glm_daywise('001', 'convnext')

%% Setup
clearvars -except sub_id source_label; close all;

if nargin < 1 || isempty(sub_id)
    error('Subject ID required');
end
if nargin < 2 || isempty(source_label)
    error('Source label required (''clip'' or ''convnext'')');
end

if isnumeric(sub_id)
    sub_id = sprintf('%03d', sub_id);
end

root_dir = '/Users/yuhiaoki/dev/hit/food-brain';
addpath('/Users/yuhiaoki/spm');
addpath(fullfile(root_dir, 'scripts', 'suzuki'));
addpath(fullfile(root_dir, 'scripts', 'common', 'first_level'));

% Memory optimization for parallel execution
spm('defaults', 'fMRI');
% Memory settings (default: no restrictions for serial execution)
% spm_get_defaults('stats.maxmem', 8*2^30);  % Uncomment for parallel execution
% spm_get_defaults('stats.resmem', false);   % Uncomment for parallel execution

sub_02 = sprintf('%02d', str2double(sub_id));

fprintf('\n========================================\n');
fprintf('Day-wise Hierarchical DNN GLM\n');
fprintf('Subject: %s, Source: %s\n', sub_id, source_label);
fprintf('========================================\n');

%% Parameters
TR = 0.8;
sess_list = {'01', '02', '03'};  % 3 days
runs_default = {'1', '2', '3', '4'};  % 4 runs per day

% Day-session mapping
day_sessions = {[1,2,3,4], [5,6,7,8], [9,10,11,12]};

%% Load PC data
pc3_path = fullfile(root_dir, 'data_images', 'dnn_pmods', '3level', ...
    [source_label, '_3level_pcs.csv']);
if ~exist(pc3_path, 'file')
    error('PC file not found: %s', pc3_path);
end
pc3_table = readtable(pc3_path, 'VariableNamingRule', 'preserve');
pc3_table.image_id_num = str2double(string(pc3_table.image_id));
fprintf('Loaded %d images from PC file\n', height(pc3_table));

% Identify PC columns
all_cols = pc3_table.Properties.VariableNames;
global_cols = all_cols(startsWith(all_cols, 'Global_pc'));
shared_cols = all_cols(startsWith(all_cols, 'Shared_'));

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
total_specific = 0;
for g = 1:length(layer_groups)
    total_specific = total_specific + numel(specific_cols.(layer_groups{g}));
end
total_pmods = num_global + num_shared + total_specific;

fprintf('Global PCs: %d\n', num_global);
fprintf('Shared PCs: %d\n', num_shared);
fprintf('Layer-specific PCs: %d\n', total_specific);
fprintf('Total PMODs per day: %d\n', total_pmods);

%% Setup paths
sub_dir = fullfile(root_dir, 'fMRIprep', 'need_info', ['sub-', sub_02]);
image_dir = fullfile(root_dir, 'fMRIprep', 'smoothed', ['sub-', sub_02]);
model_dir = fullfile(root_dir, 'results', 'first_level_analysis', ['sub-', sub_id], 'glm_model');

% Brain mask
deriv_dir = fullfile(root_dir, 'fMRIprep', 'derivatives', ['sub-', sub_02], 'anat');
fname_mask_nii = fullfile(deriv_dir, ['sub-', sub_02, '_space-MNI152NLin2009cAsym_desc-brain_mask.nii']);
fname_mask_gz = [fname_mask_nii, '.gz'];
if exist(fname_mask_nii, 'file')
    fname_mask = fname_mask_nii;
elseif exist(fname_mask_gz, 'file')
    gunzip(fname_mask_gz, deriv_dir);
    fname_mask = fname_mask_nii;
else
    fname_mask = '';
end

model_name = ['glm_dnn_pmods_hierarchical_', source_label, '_daywise'];
nowstr = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
outdir = fullfile(model_dir, model_name, nowstr);
if ~exist(outdir, 'dir'); mkdir(outdir); end
fprintf('Output: %s\n', outdir);

%% Load behavior data
beh_dir = '';
candidate_dirs = {fullfile(root_dir, 'Food_Behavior', ['sub-', sub_id]), ...
                  fullfile(root_dir, 'Food_Behavior', sub_id)};
for cdix = 1:numel(candidate_dirs)
    if exist(candidate_dirs{cdix}, 'dir')
        beh_dir = candidate_dirs{cdix};
        break;
    end
end
if isempty(beh_dir)
    error('Behavior directory not found');
end

files = dir(fullfile(beh_dir, 'rating_data*.csv'));
[~, si] = sort([files.datenum]); files = files(si);
files_Rate_list = {files.name};

files = dir(fullfile(beh_dir, 'GLM_all*.csv'));
[~, si] = sort([files.datenum]); files = files(si);
files_Time_list = {files.name};

%% Collect all data
fprintf('\nCollecting data from all sessions...\n');

all_scans = {};
session_scan_counts = [];
session_data = struct();
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

        % Get scans
        scan_pattern = ['^ssub-', sub_02, '_ses-', sess_list{s}, '_task-pt_run-0', runs{r}, '_space-MNI152NLin2009cAsym_desc-preproc_bold\.nii$'];
        data_fMRI = cellstr(spm_select('ExtFPList', fullfile(image_dir, ['ses-', sess_list{s}], 'func'), scan_pattern, Inf));

        if isempty(data_fMRI) || isempty(data_fMRI{1})
            error('No scans found for session %s run %s', sess_list{s}, runs{r});
        end

        num_scans = length(data_fMRI);
        session_scan_counts(end+1) = num_scans;
        all_scans = [all_scans; data_fMRI];

        % Load behavior
        time_filepath = fullfile(beh_dir, files_Time_list{session_cnt});
        rate_filepath = fullfile(beh_dir, files_Rate_list{session_cnt});
        data_tmp = make_behav_csv_run(sub_id, rate_filepath, time_filepath);

        idx_nomiss = ~(data_tmp.RatingValue == 0);
        idx_miss = ~idx_nomiss;

        session_data(session_cnt).day = s;
        session_data(session_cnt).image_onsets = data_tmp.image(idx_nomiss) + cumulative_time;
        session_data(session_cnt).question_onsets = data_tmp.question(idx_nomiss) + cumulative_time;
        session_data(session_cnt).question_durations = data_tmp.rating(idx_nomiss) - data_tmp.question(idx_nomiss);
        session_data(session_cnt).response_onsets = data_tmp.rating(idx_nomiss) + cumulative_time;

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

        % Motion
        confounds = readtable(fullfile(sub_dir, ['ses-', sess_list{s}], 'func', ...
            ['sub-', sub_02, '_ses-', sess_list{s}, '_task-pt_run-0', runs{r}, '_desc-confounds_timeseries.tsv']), ...
            'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
        session_data(session_cnt).motion = [confounds.trans_x, confounds.trans_y, confounds.trans_z, ...
                                            confounds.rot_x, confounds.rot_y, confounds.rot_z];
        session_data(session_cnt).num_scans = num_scans;

        fprintf('  Session %d (Day%d Run%s): %d scans, %d trials\n', ...
            session_cnt, s, runs{r}, num_scans, length(session_data(session_cnt).image_names));

        cumulative_time = cumulative_time + num_scans * TR;
    end
end

total_sessions = session_cnt;
total_scans = length(all_scans);
fprintf('Total: %d sessions, %d scans\n', total_sessions, total_scans);

%% Initialize SPM
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

%% Build SPM batch
fprintf('\nBuilding SPM design matrix...\n');

matlabbatch{1}.spm.stats.fmri_spec.dir = {outdir};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 60;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 30;

matlabbatch{1}.spm.stats.fmri_spec.sess(1).scans = all_scans;

cond_idx = 0;

%% Create ImageOnset_Day1/2/3 with PMODs
fprintf('\nCreating ImageOnset_Day conditions with PMODs...\n');

for day = 1:3
    % Collect trials for this day
    day_sessions_list = day_sessions{day};
    day_onsets = [];
    day_img_names = [];

    for ses = day_sessions_list
        if ses <= total_sessions
            day_onsets = [day_onsets; session_data(ses).image_onsets(:)];
            day_img_names = [day_img_names; session_data(ses).image_names(:)];
        end
    end

    num_day_trials = length(day_img_names);
    fprintf('  Day %d: %d trials\n', day, num_day_trials);

    % Build PMODs for this day
    pmod_struct = struct('name', {}, 'param', {}, 'poly', {});
    pmod_count = 0;

    % Global PCs
    for pc = 1:num_global
        col_name = global_cols{pc};
        pc_values = zeros(num_day_trials, 1);
        for trial = 1:num_day_trials
            img_id = day_img_names(trial);
            row_match = (pc3_table.image_id_num == img_id);
            if any(row_match)
                pc_values(trial) = pc3_table.(col_name)(row_match);
            end
        end
        pc_z = (pc_values - mean(pc_values, 'omitnan')) ./ std(pc_values, 'omitnan');
        pc_z(~isfinite(pc_z)) = 0;

        pmod_count = pmod_count + 1;
        pmod_struct(pmod_count).name = sprintf('Global_pc%d', pc);
        pmod_struct(pmod_count).param = pc_z(:)';
        pmod_struct(pmod_count).poly = 1;
    end

    % Shared PCs
    for pc = 1:num_shared
        col_name = shared_cols{pc};
        pc_values = zeros(num_day_trials, 1);
        for trial = 1:num_day_trials
            img_id = day_img_names(trial);
            row_match = (pc3_table.image_id_num == img_id);
            if any(row_match)
                pc_values(trial) = pc3_table.(col_name)(row_match);
            end
        end
        pc_z = (pc_values - mean(pc_values, 'omitnan')) ./ std(pc_values, 'omitnan');
        pc_z(~isfinite(pc_z)) = 0;

        pmod_count = pmod_count + 1;
        pmod_name = strrep(col_name, '-', '_');
        pmod_struct(pmod_count).name = pmod_name;
        pmod_struct(pmod_count).param = pc_z(:)';
        pmod_struct(pmod_count).poly = 1;
    end

    % Layer-specific PCs
    for g = 1:length(layer_groups)
        group_name = layer_groups{g};
        cols = specific_cols.(group_name);
        for pc = 1:length(cols)
            col_name = cols{pc};
            pc_values = zeros(num_day_trials, 1);
            for trial = 1:num_day_trials
                img_id = day_img_names(trial);
                row_match = (pc3_table.image_id_num == img_id);
                if any(row_match)
                    pc_values(trial) = pc3_table.(col_name)(row_match);
                end
            end
            pc_z = (pc_values - mean(pc_values, 'omitnan')) ./ std(pc_values, 'omitnan');
            pc_z(~isfinite(pc_z)) = 0;

            pmod_count = pmod_count + 1;
            pmod_struct(pmod_count).name = sprintf('%s_pc%d', group_name, pc);
            pmod_struct(pmod_count).param = pc_z(:)';
            pmod_struct(pmod_count).poly = 1;
        end
    end

    % Add condition
    cond_idx = cond_idx + 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).name = sprintf('ImageOnset_Day%d', day);
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).onset = day_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).pmod = pmod_struct;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).orth = 0;

    fprintf('    PMODs: %d\n', pmod_count);
end

%% Create session-specific ImageOnset (no PMODs) for runs 2,3,4 within each day
fprintf('\nCreating session-specific ImageOnset conditions (no PMODs)...\n');

for day = 1:3
    day_sessions_list = day_sessions{day};
    % Skip first run of each day (captured by ImageOnset_Day)
    for idx = 2:length(day_sessions_list)
        ses = day_sessions_list(idx);
        if ses <= total_sessions && ~isempty(session_data(ses).image_onsets)
            cond_idx = cond_idx + 1;
            matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).name = sprintf('ImageOnset_S%02d', ses);
            matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).onset = session_data(ses).image_onsets;
            matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).duration = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).tmod = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).pmod = struct('name', {}, 'param', {}, 'poly', {});
            matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).orth = 1;
            fprintf('  ImageOnset_S%02d: %d trials\n', ses, length(session_data(ses).image_onsets));
        end
    end
end

%% Session-specific Question conditions
fprintf('\nCreating Question conditions...\n');
for ses = 1:total_sessions
    cond_idx = cond_idx + 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).name = sprintf('Question_S%02d', ses);
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).onset = session_data(ses).question_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).duration = session_data(ses).question_durations;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).pmod = struct('name', {}, 'param', {}, 'poly', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).orth = 1;
end

%% Session-specific Response conditions
fprintf('Creating Response conditions...\n');
for ses = 1:total_sessions
    cond_idx = cond_idx + 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).name = sprintf('Response_S%02d', ses);
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).onset = session_data(ses).response_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).pmod = struct('name', {}, 'param', {}, 'poly', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).orth = 1;
end

%% Session-specific Miss conditions
fprintf('Creating Miss conditions...\n');
for ses = 1:total_sessions
    if ~isempty(session_data(ses).miss_onsets)
        cond_idx = cond_idx + 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).name = sprintf('Miss_S%02d', ses);
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).onset = session_data(ses).miss_onsets;
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).duration = session_data(ses).miss_durations;
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(1).cond(cond_idx).orth = 1;
    end
end

fprintf('Total conditions: %d\n', cond_idx);

%% Motion regressors (session-specific)
fprintf('\nAdding motion regressors...\n');

all_regressors = {};
for ses = 1:total_sessions
    start_idx = sum(session_scan_counts(1:ses-1)) + 1;
    end_idx = sum(session_scan_counts(1:ses));

    for m = 1:6
        reg_val = zeros(total_scans, 1);
        reg_val(start_idx:end_idx) = session_data(ses).motion(:, m);
        all_regressors{end+1} = reg_val;
    end
end

for ridx = 1:length(all_regressors)
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).regress(ridx).name = sprintf('Motion_%03d', ridx);
    matlabbatch{1}.spm.stats.fmri_spec.sess(1).regress(ridx).val = all_regressors{ridx};
end
fprintf('Motion regressors: %d\n', length(all_regressors));

%% Other settings
matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi = {''};
matlabbatch{1}.spm.stats.fmri_spec.sess(1).multi_reg = {''};
matlabbatch{1}.spm.stats.fmri_spec.sess(1).hpf = 128;

matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
if ~isempty(fname_mask)
    matlabbatch{1}.spm.stats.fmri_spec.mask = {fname_mask};
else
    matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
end
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

%% Model estimation
matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(outdir, 'SPM.mat')};
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

%% Run
fprintf('\nRunning SPM batch...\n');
spm_jobman('run', matlabbatch);

%% Build contrasts
fprintf('\nBuilding contrasts...\n');
build_contrasts_daywise(fullfile(outdir, 'SPM.mat'), source_label, ...
    global_cols, shared_cols, layer_groups, specific_cols);

fprintf('\n========================================\n');
fprintf('Day-wise GLM Complete\n');
fprintf('Output: %s\n', outdir);
fprintf('========================================\n');

end
