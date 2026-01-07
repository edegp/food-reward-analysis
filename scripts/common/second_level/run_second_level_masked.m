function dir_name = run_second_level_masked(subs_list, model_name, con_title, mask_file)
% Run second-level analysis with explicit brain mask
% mask_file: path to group brain mask (optional, will create if not provided)

    % Get root directory from script location
    script_dir = fileparts(mfilename('fullpath'));
    root_dir = fullfile(script_dir, '..', '..', '..');

    % Create or use group mask
    if nargin < 4 || isempty(mask_file)
        mask_file = fullfile(root_dir, 'rois', 'group_mask', 'group_brain_mask.nii');
        if ~exist(mask_file, 'file')
            fprintf('Creating group brain mask...\n');
            mask_file = create_group_mask();
        end
    end
    fprintf('Using mask: %s\n', mask_file);

    for c = 1:length(con_title)

        spm('Defaults', 'fMRI');
        spm_jobman('initcfg');

        % Make Directory for the RFX
        ts = char(datetime('now','Format','yyyyMMdd_HHmmss'));
        model_name_ch = char(model_name);
        con_title_ch  = char(con_title{c});
        dir_name = fullfile(root_dir, 'results', 'second_level_analysis', model_name_ch, [con_title_ch, '_', ts]);

        if ~( (ischar(dir_name) && isrow(dir_name) && ~isempty(dir_name)) || (isstring(dir_name) && isscalar(dir_name) && strlength(dir_name) > 0) )
            error('run_second_level_masked:InvalidDirName', 'Constructed dir_name is invalid for mkdir: %s', mat2str(dir_name));
        end

        if exist(dir_name ,'dir'), rmdir(dir_name,'s'); end
        if ~exist(dir_name ,'dir'), mkdir(dir_name); end

        % Prepare Each Subjects Data (contrast)
        scans_store = cell(length(subs_list),1);
        for i = 1:length(subs_list)

            % SPM.matの読み込み
            sub_dir = dir(fullfile(root_dir, 'results', 'first_level_analysis', ['sub-',subs_list{i}], 'glm_model', model_name, '**', 'SPM.mat'));
            [~, idx] = max([sub_dir.datenum]);
            disp([idx, length(sub_dir)]);
            sub_dir = sub_dir(idx).folder;
            load(fullfile(sub_dir, 'SPM.mat'))

            % 当該コントラストの指定
            num_cons = length(SPM.xCon);
            con_name = '';
            for cc = 1:num_cons
                name_list = SPM.xCon(cc).name;
                disp([name_list, con_title{c}]);
                if isequal(name_list, con_title{c})
                    con_name = fullfile(sub_dir, SPM.xCon(cc).Vcon.fname);
                    break;
                end
            end
            if isempty(con_name)
                error('run_second_level_masked:ContrastNotFound', ...
                    'Contrast "%s" not found for subject %s', con_title{c}, subs_list{i});
            end
            scans_store{i} = con_name;

        end
        disp(scans_store)

        % Model Specification with explicit mask
        matlabbatch{1}.spm.stats.factorial_design.dir = {dir_name};
        matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = scans_store;
        matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
        matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
        matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
        % Use explicit brain mask
        matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask_file};
        matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

        % Model Estimation
        matlabbatch{2}.spm.stats.fmri_est.spmmat = cellstr(fullfile(dir_name, 'SPM.mat'));
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

        % Contrast Definition
        matlabbatch{3}.spm.stats.con.spmmat = cellstr(fullfile(dir_name, 'SPM.mat'));
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = con_title{c};
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.convec = 1;
        matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
        matlabbatch{3}.spm.stats.con.delete = 1;

        % Save batch
        job_id = cfg_util('initjob', matlabbatch);
        cfg_util('savejob', job_id, fullfile(dir_name, 'secondlevel_batch.m'));

        spm_jobman('run', matlabbatch);
        clear matlabbatch
    end

end
