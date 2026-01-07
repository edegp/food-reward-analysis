function run_model_1st_6
%%
clearvars; close all; clc;

subs_list = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031'}; % 被験者リスト
% subs_list = {'017','018', '019', '020'}; % 被験者リスト
% subs_list = {'018'}; % 被験者リスト
% subs_list = {'001'}; % 被験者リスト
sess = {'01','02','03'};
runs_tmp = {'1','2','3','4'};
model_name_list = {'glm_001p_6'};
% Define contrasts to add after estimation (one cell per model)
cons_list = {
    {'ImagexValue'},
    % {'QuestionxValue'},
};
cons_name_list = {
    {'ImagexValue'},
    % {'QuestionxValue'}
};

canonical_dir = spm('Dir');
if isempty(canonical_dir)
    error('SPM not on MATLAB path: unable to locate canonical image.');
end
canonical_img = [fullfile(canonical_dir, 'canonical', 'single_subj_T1.nii'), ',1'];
disp(canonical_img);
% If 1, existing contrasts are deleted and replaced. If 0, contrasts are appended.
delete_existing_contrasts = 1;
% run000_Extract_ImageInfo()
project_dir = fullfile(fileparts(mfilename('fullpath')))
root_dir = fullfile(project_dir, '..', '..', '..'); % project root for absolute paths

addpath(fullfile(root_dir, 'scripts', 'common', 'first_level'));
addpath(fullfile(root_dir, 'scripts', 'common', 'second_level'));
addpath(fullfile(root_dir, 'scripts', 'common', 'visualization'));

for n = 1:length(model_name_list)

    %spm fmri
    model_name = model_name_list{n};

    % Subjects list
    for i = 1:length(subs_list)
        % Avoid changing directories inside parfor (transparency)
        cd(project_dir)

        spm('Defaults', 'fMRI'); % Initialise SPM
        spm_jobman('initcfg'); % Initialise cfg_util

        sub_02 = sprintf('%02d', str2double(subs_list{i}));
        sub_dir = fullfile(root_dir, 'fMRIprep', 'need_info', ['sub-', sub_02]);
        image_dir = fullfile(root_dir, 'fMRIprep', 'smoothed', ['sub-', sub_02]);
        model_dir = fullfile(root_dir, 'results', 'first_level_analysis', ['sub-', subs_list{i}], 'glm_model');

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

        %%%%%%%%%% Model Specification %%%%%%%%%%
        now = datetime('now','Format','yyyyMMdd_HHmmss');
        outdir = fullfile(model_dir, model_name, char(now));
        % ensure output directory exists before saving any batch files
        if ~exist(outdir, 'dir')
            mkdir(outdir);
        end
        matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(outdir);
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 0.8;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 60;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 30;
        cnt = 0;
        for s = 1:length(sess)

            %%%%% 一部の欠損データを処理 %%%%%
            if isequal(subs_list{i},'016') && isequal(sess{s},'03')
                runs = {'1','2'};
            else
                runs = runs_tmp;
            end

            for j = 1:length(runs)
                data_fMRI = cellstr(spm_select('FPList', [image_dir,'/ses-',sess{s},'/func/'],['ssub-',sub_02,'_ses-',sess{s},'_task-pt_run-0',runs{j},'_space-MNI152NLin2009cAsym_desc-preproc_bold.nii']));
                cnt = cnt + 1;
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).scans = data_fMRI;
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});
                model_name_tmp = [model_name,'_',num2str(cnt)];
                % ensure the per-run .mat exists by creating it on the fly

                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).multi = cellstr(make_glm_001p_6_run(subs_list{i}, runs{j}, model_dir, model_name, cnt, model_name_tmp));
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).regress = struct('name', {}, 'val', {});
                % Addition of the motion correction parameters
                confounds = readtable([sub_dir,'/ses-',sess{s},'/func/','sub-',sub_02,'_ses-',sess{s},'_task-pt_run-0',runs{j},'_desc-confounds_timeseries.tsv'], 'FileType','text', 'Delimiter','\t', 'VariableNamingRule','preserve');
                data_MC = [confounds.trans_x, confounds.trans_y, confounds.trans_z, confounds.rot_x, confounds.rot_y, confounds.rot_z];
                mc_dir = fullfile(sub_dir, ['ses-',sess{s}], 'func');
                if ~exist(mc_dir, 'dir'); mkdir(mc_dir); end
                mc_file = fullfile(mc_dir, ['rp_sub-',sub_02,'_ses-',sess{s},'_task-pt_run-0',runs{j},'_desc-confounds_timeseries.txt']);
                writematrix(data_MC, mc_file, 'Delimiter', 'tab');
                file_MC = {mc_file};
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).multi_reg = file_MC;
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).hpf = 128; %%%%%

            end
        end

        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0;
        matlabbatch{1}.spm.stats.fmri_spec.mask = {fname_mask};
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

        %%%%%%%%%% Model Estimation %%%%%%%%%%
        disp('Estimated model:');

        disp(cellstr(fullfile(outdir, 'SPM.mat')))
        matlabbatch{2}.spm.stats.fmri_est.spmmat = cellstr(fullfile(outdir, 'SPM.mat'));
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
        % 既に matlabbatch という構造体があるとする
        job_id = cfg_util('initjob', matlabbatch);  % ワークスペースの matlabbatch を init して job_id を得る
        cfg_util('savejob', job_id, fullfile(outdir,'firstlevel_batch.m'));    % 再現用 matlab スクリプトを生成

        spm_jobman('run', matlabbatch); % exerting the processing!

        % Avoid clear inside parfor (transparency)
        matlabbatch = [];

        try
            spm_path = fullfile(outdir, 'SPM.mat');
            % cnt
            runs = 1:cnt;
            matlabbatch = add_contrasts(spm_path, cons_list{n}, cons_name_list{n},  runs, delete_existing_contrasts);
        catch ME
            warning('add_contrasts failed: %s', ME.message);
        end

        % 既に matlabbatch という構造体があるとする
        job_id = cfg_util('initjob', matlabbatch);  % ワークスペースの matlabbatch を init して job_id を得る
        cfg_util('savejob', job_id, fullfile(outdir, 'firstlevel_addcon_batch.m'));    % 再現用 matlab スクリプトを生成

        spm_jobman('run', matlabbatch); % exerting the processing!

        % Avoid clear inside parfor (transparency)
        matlabbatch = [];

        try
            visualize_firstlevel(spm_path, fullfile(outdir, 'figures'), canonical_img);
        catch ME
            warning('visualize_firstlevel failed: %s', ME.message);
        end

        % Avoid clear inside parfor (transparency)
        matlabbatch = [];

    end

    cd(project_dir)

    for con_title = cons_name_list
        dir_name = run_second_level(subs_list, model_name, con_title{n});
        visualize_secondlevel(model_name, con_title{n}, dir_name);
    end
end
