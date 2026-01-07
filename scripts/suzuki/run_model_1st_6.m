function run_model_1st_6
%%
clear all; close all; clc;
subs_list = {'001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016'}; % 被験者リスト
sess = {'01','02','03'};
runs_tmp = {'1','2','3','4'};
model_name_list = {'glm_001p_6'};

for n = 1:length(model_name_list)
    
    %spm fmri
    model_name = model_name_list{n};
    
    % Subjects list
    for i = 1:length(subs_list)

        spm('Defaults', 'fMRI'); % Initialise SPM
        spm_jobman('initcfg'); % Initialise cfg_util
        
        sub_dir = ['/Volumes/mri_suzuki2/FOOD_bids/derivatives/sub-',subs_list{i}];
        model_dir = ['/Volumes/mri_suzuki2/FOOD_bids/derivatives/sub-',subs_list{i},'/glm_model'];
        
        %%%%%%%%%% Model Specification %%%%%%%%%%
        matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(fullfile(model_dir, model_name));
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
                disp(spm_select('FPList', [sub_dir,'/ses-',sess{s},'/func/'],['s6wr_sub-',subs_list{i},'_ses-',sess{s},'_task-food_run-0',runs{j},'_space-MNI152NLin2009cAsym_desc-preproc_bold.nii']))
                data_fMRI = cellstr(spm_select('FPList', [sub_dir,'/ses-',sess{s},'/func/'],['s6wr_sub-',subs_list{i},'_ses-',sess{s},'_task-food_run-0',runs{j},'_space-MNI152NLin2009cAsym_desc-preproc_bold.nii']));
                cnt = cnt + 1;
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).scans = data_fMRI;
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});
                model_name_tmp = [model_name,'_',num2str(cnt)];
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).multi = cellstr(fullfile(model_dir, model_name, strcat(model_name_tmp, '.mat')));
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).regress = struct('name', {}, 'val', {});
                % Addition of the motion correction parameters
                confounds = tdfread([sub_dir,'/ses-',sess{s},'/func/','sub-',subs_list{i},'_ses-',sess{s},'_task-food_run-0',runs{j},'_desc-confounds_timeseries.tsv']);
                data_MC = [confounds.trans_x, confounds.trans_y, confounds.trans_z, confounds.rot_x, confounds.rot_y, confounds.rot_z];
                writematrix(data_MC, [sub_dir,'/ses-',sess{s},'/func/','rp_sub-',subs_list{i},'_ses-',sess{s},'_task-food_run-0',runs{j},'_desc-confounds_timeseries.txt'], 'Delimiter', 'tab');
                file_MC = {[sub_dir,'/ses-',sess{s},'/func/','rp_sub-',subs_list{i},'_ses-',sess{s},'_task-food_run-0',runs{j},'_desc-confounds_timeseries.txt']};
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).multi_reg = file_MC;
                matlabbatch{1}.spm.stats.fmri_spec.sess(cnt).hpf = 128; %%%%%

            end
        end
        
        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.75; %%%%%%%%%%
        matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
        
        %%%%%%%%%% Model Estimation %%%%%%%%%%
        disp('Estimated model:');
        disp(cellstr(fullfile(model_dir, model_name, 'SPM.mat')))
        matlabbatch{2}.spm.stats.fmri_est.spmmat = cellstr(fullfile(model_dir, model_name, 'SPM.mat'));
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
        
        spm_jobman('run', matlabbatch); % exerting the processing!
        clear matlabbatch
        
    end
    
end
