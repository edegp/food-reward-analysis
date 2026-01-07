function make_glm_001p_6
%%
clear all; close all; clc;

glm_name = 'glm_001p_6'; % モデルの名前
subs_list = {'001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016'}; % 被験者リスト
runs_tmp = {'1','2','3','4','5','6','7','8','9','10','11','12'}; % Runの数

for s = 1:length(subs_list)
    
    subs = subs_list{s};
    disp(['********** ',subs, ' **********'])

    %%%%% ディレクトリの作成 %%%%%
    dir_save = ['/Volumes/mri_suzuki2/Food_bids/derivatives/sub-',subs,'/glm_model/',glm_name]; disp(dir_save)
    if ~exist(dir_save ,'dir'), mkdir(dir_save); end

    %%%%% 一部の欠損データを処理 %%%%%
    if isequal(subs,'016')
        runs = {'1','2','3','4','5','6','7','8','9','10'};
    else
        runs = runs_tmp;
    end
    
    %%%%% 各セッションのデザイン作成 %%%%%
    for i = 1:length(runs)

        disp(['********** run ',runs{i}, ' **********'])
        
        % 行動データの読み込み
        file_tmp = ['/Users/shinsuke/Library/CloudStorage/Dropbox/p24_FoodMRI/data_behav/sub_',subs,'/sub_',subs,'_run_',runs{i},'.csv']; disp(file_tmp)
        data_tmp = readtable(file_tmp)
        
        %%%%% 各条件のデータ整理

        % Index for NO-MISS trials
        idx_nomiss = ~(data_tmp.RatingValue == 0);

        % Image:
        onsets_tmp = data_tmp.image;
        durations_tmp = data_tmp.question - data_tmp.image;
        names{1} = 'Image';
        onsets{1} = onsets_tmp(idx_nomiss);
        durations{1} = durations_tmp(idx_nomiss);

        % Parametric Modulation (Rating value)
        pmod(1).name{1} = 'Value';
        pmod(1).param{1} = data_tmp.RatingValue(idx_nomiss);
        pmod(1).poly{1} = 1;
        orth{1} = 0;
        if isequal(std(data_tmp.RatingValue(idx_nomiss)),0)
            disp(['WARNNING: NO VARIAVILITY!!!'])
        end

        % Parametric Modulation (Luminance)
        pmod(1).name{2} = 'Luminance';
        pmod(1).param{2} = data_tmp.image_L(idx_nomiss);
        pmod(1).poly{2} = 1;


        % Question
        onsets_tmp = data_tmp.question;
        durations_tmp = data_tmp.rating - data_tmp.question;
        names{2} = 'Question';
        onsets{2} = onsets_tmp(idx_nomiss);
        durations{2} = durations_tmp(idx_nomiss);

        % Response
        onsets_tmp = data_tmp.rating;
        durations_tmp = zeros(length(onsets_tmp),1);
        names{3} = 'Response';
        onsets{3} = onsets_tmp(idx_nomiss);
        durations{3} = durations_tmp(idx_nomiss);

        % Feedback
        onsets_tmp = data_tmp.rating;
        durations_tmp = ones(length(onsets_tmp),1) * 0.5;
        names{4} = 'Feedback';
        onsets{4} = onsets_tmp(idx_nomiss);
        durations{4} = durations_tmp(idx_nomiss);

        % Miss trials
        if sum(1 - idx_nomiss) > 0
            onsets_tmp = data_tmp.image;
            durations_tmp = data_tmp.rating - data_tmp.image;
            idx_tmp = ~idx_nomiss;
            names{5} = 'Miss';
            onsets{5} = onsets_tmp(idx_tmp);
            durations{5} = durations_tmp(idx_tmp);
        end

        % Save
        f_name_save = [dir_save,'/',glm_name,'_',runs{i}];
        save(f_name_save,'names','onsets','durations','orth',"pmod")
        
        clear names onsets durations pmod orth;
        
    end
    
end
