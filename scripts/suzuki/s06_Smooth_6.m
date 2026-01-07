function s06_Smooth_6
%%
clear all; close all; clc; tic
subs_list = {'014'}; % 複数の被験者を指定可
sessions = {'01','02','03'};
runs_tmp = {'01','02','03','04'};

for s = 1:length(subs_list)

    for k = 1:length(sessions)

        subs = subs_list{s};
        dir_spm = ['/Volumes/mri_suzuki2/FOOD_bids/derivatives/sub-',subs,'/ses-',sessions{k}];
        dir_tmp = [dir_spm,'/func/'];
        disp(dir_spm)

        %%%%% 一部の欠損データを処理 %%%%%
        if isequal(subs,'016') && isequal(sessions{k},'03')
            runs = {'01','02','03'};
        else
            runs = runs_tmp;
        end

        for i = 1:length(runs)

            % gunzip([dir_tmp,'sub-',subs,'_ses-',sessions{k},'_task-food_run-',runs{i},'_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz']) % 解凍

        end

        % データ指定
        data = spm_select('FPList', fullfile(dir_tmp), '^sub.*\_space-MNI152NLin2009cAsym_desc-preproc_bold.nii$');
        data = cellstr(data);
        disp(data)

        % run spm
        spm('Defaults', 'fMRI'); % Initialise SPM
        spm_jobman('initcfg'); % Initialise cfg_util

        matlabbatch{1}.spm.spatial.smooth.data = data;
        matlabbatch{1}.spm.spatial.smooth.fwhm = [6 6 6];
        matlabbatch{1}.spm.spatial.smooth.dtype = 0;
        matlabbatch{1}.spm.spatial.smooth.im = 0;
        matlabbatch{1}.spm.spatial.smooth.prefix = 's6wr_';

        spm_jobman('run', matlabbatch); % exerting the processing!
        clear matlabbatch
        clear data

    end

end
toc

end
