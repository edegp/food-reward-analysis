function run_model_con
%%
clear all; close all;
subs_list = {'001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016'}; % 被験者リスト
runs = {'1','2','3','4','5','6','7','8','9','10','11','12'}; % Runの数
model_name_list = {'glm_001p_6'};
cons_list = {
    {'ImagexValue'}
    };
cons_name_list = {
    {'ImagexValue'}
    };

for n = 1:length(model_name_list)
    
    % Specify the model and contrast
    model_name = model_name_list{n};
    cons_model = cons_list{n};
    cons_model_name = cons_name_list{n};
    
    for i = 1:length(subs_list)
    %for i = 1:4
        
        spm('Defaults', 'fMRI'); % Initialise SPM
        spm_jobman('initcfg'); % Initialise cfg_util
        
        sub_dir = ['/Volumes/mri_suzuki2/Food_bids/derivatives/sub-',subs_list{i},'/glm_model'];
        disp(['%%%%% ', sub_dir, ' %%%%%'])
        
        %%%%%%%%%% Contrast Definition %%%%%%%%%%
        fullfile(sub_dir, model_name, 'SPM.mat')
        matlabbatch{1}.spm.stats.con.spmmat = cellstr(fullfile(sub_dir, model_name, 'SPM.mat'));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        load(fullfile(sub_dir, model_name, 'SPM.mat'))
        name_list = SPM.xX.name
        
        for c = 1:length(cons_model)
            
            cons = cons_model{c};
            cons_name = cons_model_name{c};
            disp(cons_name)

            idx = [];
            for j = 1:length(runs)
                con_name = ['Sn(',num2str(j),') ',cons,'^1*bf(1)'];
                idx_tmp = strmatch(con_name, name_list, 'exact');
                if isequal(sum(idx_tmp),0)
                    'test';
                    con_name = ['Sn(',num2str(j),') ',cons,'*bf(1)'];
                    idx_tmp = strmatch(con_name, name_list, 'exact');
                end
                idx = [idx, idx_tmp];
            end
            con_vec = zeros(1,length(name_list));
            con_vec(idx) = 1;
            disp(con_vec)
            
            matlabbatch{1}.spm.stats.con.consess{c}.tcon.name = cons_name; % IMPORTANT!!!!!!!!!!
            matlabbatch{1}.spm.stats.con.consess{c}.tcon.convec = con_vec; % CAUTION!!!!!!!!
            matlabbatch{1}.spm.stats.con.consess{c}.tcon.sessrep = 'none';
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        matlabbatch{1}.spm.stats.con.delete = 1; %%%%% NO %%%%%
        
        spm_jobman('run', matlabbatch); % exerting the processing!
        clear matlabbatch
        
    end
    
end
