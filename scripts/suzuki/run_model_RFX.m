function run_model_RFX
%%
clear all; close all; clc
subs_list = {'001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016'}; % 被験者リスト
model_name_list = {'glm_001p_6'};
cons_name_list = {
    {'ImagexValue'}
    };

for m = 1:length(model_name_list)
    
    model_name = model_name_list{m};
    con_title = cons_name_list{m};
    
    for c = 1:length(con_title)
        
        spm('Defaults', 'fMRI'); % Initialise SPM
        spm_jobman('initcfg'); % Initialise cfg_util
        
        % Make Directory for the RFX
        dir_name = [pwd,'/',model_name,'/',con_title{c}];
        if exist(dir_name ,'dir'), rmdir(dir_name,'s'); end
        if ~exist(dir_name ,'dir'), mkdir(dir_name); end
        
        % Prepare Each Subjects Data (contrast)
        scans_store = cell(length(subs_list),1);
        for i = 1:length(subs_list)
            
            % SPM.matの読み込み
            sub_dir = ['/Volumes/mri_suzuki2/FOOD_bids/derivatives/sub-',subs_list{i},'/glm_model/'];
            load(fullfile(sub_dir, model_name, 'SPM.mat'))
            
            % 当該コントラストの指定
            num_cons = length(SPM.xCon);
            for cc = 1:num_cons
                
                name_list = SPM.xCon(cc).name;
                if isequal(name_list,con_title{c})
                    con_name = [sub_dir,model_name,'/',SPM.xCon(cc).Vcon.fname];
                    break;
                end
                    
            end
            scans_store{i} = con_name;
            
        end
        disp(scans_store)
        
        % Model Specification
        matlabbatch{1}.spm.stats.factorial_design.dir = {dir_name};
        matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = scans_store;
        matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
        matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
        matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
        matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
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
        
        spm_jobman('run', matlabbatch); % exerting the processing!
        clear matlabbatch
        
    end
    
end
close all

end

