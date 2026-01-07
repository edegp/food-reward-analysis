function run_lss_glm(varargin)
%% Run LSS (Least Squares Separate) GLM for all subjects
% Each image presentation is modeled as a separate regressor
% This allows estimation of individual beta values for each image
%
% This script:
% 1. Creates LSS design matrices for each run
% 2. Estimates the GLM using SPM
% 3. Extracts beta values for each image regressor
% 4. Saves beta values for encoding model analysis
%
% Usage:
%   run_lss_glm()           - Process all subjects (001-031)
%   run_lss_glm(start_idx)  - Process from start_idx to 31
%   run_lss_glm(start_idx, end_idx) - Process subjects start_idx to end_idx

clearvars -except varargin; close all; clc;

% Set default character encoding to UTF-8
feature('DefaultCharacterSet', 'UTF-8');

% Subject list - all 31 subjects
all_subs = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010', ...
            '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', ...
            '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031'};

% Parse arguments
if nargin == 0
    start_idx = 1;
    end_idx = 31;
elseif nargin == 1
    start_idx = varargin{1};
    end_idx = 31;
elseif nargin == 2
    start_idx = varargin{1};
    end_idx = varargin{2};
else
    error('Too many arguments. Usage: run_lss_glm(start_idx, end_idx)');
end

% Select subjects to process
subs_list = all_subs(start_idx:end_idx);

sess = {'01', '02', '03'};
runs_tmp = {'1', '2', '3', '4'};
model_name = 'lss_glm';

% Check SPM
canonical_dir = spm('Dir');
if isempty(canonical_dir)
    error('SPM not on MATLAB path: unable to locate canonical image.');
end

% Project directories
project_dir = fullfile(fileparts(mfilename('fullpath')), '..');
root_dir = fullfile(project_dir, '..', '..');

% Add parent directory to path for helper functions
addpath(project_dir);

fprintf('==============================================\n');
fprintf('LSS GLM Analysis\n');
fprintf('==============================================\n');
fprintf('Subjects: %d\n', length(subs_list));
fprintf('Model: %s\n\n', model_name);

% Process each subject
for i = 1:length(subs_list)
    fprintf('Processing subject %s...\n', subs_list{i});

    spm('Defaults', 'fMRI');
    spm_jobman('initcfg');

    sub_02 = sprintf('%02d', str2double(subs_list{i}));
    fmriprep_dir = fullfile(root_dir, 'fMRIprep');
    sub_dir = fullfile(root_dir, 'fMRIprep', 'need_info', ['sub-', sub_02]);
    image_dir = fullfile(fmriprep_dir, 'smoothed', ['sub-', sub_02]);
    model_dir = fullfile(root_dir, 'results', 'first_level_analysis', ['sub-', subs_list{i}], 'glm_model');

    % Create output directory
    now = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
    outdir = fullfile(model_dir, model_name, char(now));
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    %%%%%%%%%% Model Specification %%%%%%%%%%
    matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(outdir);
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 0.8;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 60;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 30;

    cnt = 0;
    for s = 1:length(sess)
        % Handle missing data for subject 016, session 03 (run-03, run-04 missing)
        if isequal(subs_list{i}, '016') && isequal(sess{s}, '03')
            runs = {'1', '2'};
        else
            runs = runs_tmp;
        end

        for j = 1:length(runs)
            % Get fMRI data
            data_fMRI = cellstr(spm_select('FPList', ...
                [image_dir, '/ses-', sess{s}, '/func/'], ...
                ['ssub-', sub_02, '_ses-', sess{s}, '_task-pt_run-0', runs{j}, '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii']));

            cnt = cnt + 1;
            matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).scans = data_fMRI;
            matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});

            % Create LSS design matrix for this run
            model_name_tmp = [model_name, '_', num2str(cnt)];
            mat_file = make_lss_glm_run(subs_list{i}, runs{j}, model_dir, model_name, cnt, model_name_tmp);
            matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).multi = cellstr(mat_file);
            matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).regress = struct('name', {}, 'val', {});

            % Add motion correction parameters
            confounds_file = fullfile(sub_dir, ['ses-', sess{s}], 'func', ...
                ['sub-', sub_02, '_ses-', sess{s}, '_task-pt_run-0', runs{j}, '_desc-confounds_timeseries.tsv']);
            confounds = readtable(confounds_file, ...
                'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
            data_MC = [confounds.trans_x, confounds.trans_y, confounds.trans_z, ...
                      confounds.rot_x, confounds.rot_y, confounds.rot_z];
            mc_dir = fullfile(sub_dir, ['ses-', sess{s}], 'func');
            if ~exist(mc_dir, 'dir'); mkdir(mc_dir); end
            mc_file = fullfile(mc_dir, ['rp_sub-', sub_02, '_ses-', sess{s}, ...
                '_task-pt_run-0', runs{j}, '_desc-confounds_timeseries.txt']);
            writematrix(data_MC, mc_file, 'Delimiter', 'tab');
            matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).multi_reg = {mc_file};
            matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).hpf = 128;
        end
    end

    % Model specification parameters
    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0;
    matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

    %%%%%%%%%% Model Estimation %%%%%%%%%%
    matlabbatch{2}.spm.stats.fmri_est.spmmat = cellstr(fullfile(outdir, 'SPM.mat'));
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    % Save batch file
    job_id = cfg_util('initjob', matlabbatch);
    cfg_util('savejob', job_id, fullfile(outdir, 'lss_glm_batch.m'));

    fprintf('  Running SPM GLM estimation...\n');
    spm_jobman('run', matlabbatch);

    matlabbatch = [];

    % Extract beta values for each image
    fprintf('  Extracting beta values...\n');
    extract_lss_betas(subs_list{i}, outdir, cnt);

    fprintf('  Subject %s completed\n\n', subs_list{i});
end

fprintf('==============================================\n');
fprintf('LSS GLM Analysis Complete\n');
fprintf('==============================================\n');

end


function extract_lss_betas(sub_id, spm_dir, n_runs)
%% Extract beta values for each image regressor from LSS GLM
%
% Inputs:
%   sub_id: subject ID (e.g., '001')
%   spm_dir: directory containing SPM.mat
%   n_runs: number of runs

    % Load SPM
    spm_path = fullfile(spm_dir, 'SPM.mat');
    if ~exist(spm_path, 'file')
        error('SPM.mat not found at %s', spm_path);
    end

    load(spm_path);

    % Find all Image_* regressors
    image_regressors = {};
    image_ids = {};
    beta_indices = [];

    for i = 1:length(SPM.xX.name)
        regressor_name = SPM.xX.name{i};
        % Match "Sn(run) Image_XXXX*bf(1)" pattern
        if contains(regressor_name, 'Image_') && contains(regressor_name, '*bf(1)')
            image_regressors{end+1} = regressor_name;
            beta_indices(end+1) = i;

            % Extract image ID from regressor name
            % Format: "Sn(1) Image_0138*bf(1)" -> "0138"
            parts = strsplit(regressor_name, 'Image_');
            if length(parts) >= 2
                id_part = strsplit(parts{2}, '*');
                image_ids{end+1} = id_part{1};
            end
        end
    end

    fprintf('    Found %d image regressors\n', length(image_regressors));

    % Create output directory
    beta_dir = fullfile(spm_dir, 'beta_values');
    if ~exist(beta_dir, 'dir')
        mkdir(beta_dir);
    end

    % Save beta information
    beta_info = struct();
    beta_info.subject_id = sub_id;
    beta_info.n_runs = n_runs;
    beta_info.n_images = length(image_ids);
    beta_info.image_ids = image_ids;
    beta_info.beta_indices = beta_indices;
    beta_info.regressor_names = image_regressors;

    save(fullfile(beta_dir, 'beta_info.mat'), 'beta_info');

    % Save as CSV for easy loading in Python
    beta_table = table(image_ids', beta_indices', 'VariableNames', {'image_id', 'beta_index'});
    writetable(beta_table, fullfile(beta_dir, 'beta_info.csv'));

    fprintf('    Saved beta information to %s\n', beta_dir);
end
