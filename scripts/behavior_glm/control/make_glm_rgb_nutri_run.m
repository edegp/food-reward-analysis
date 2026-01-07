function matpath = make_glm_rgb_nutri_run(subs, run_label, model_dir, model_name, cnt, model_name_tmp)
%% Create and save names/onsets/durations/pmod for a single subject-run
%% Model: Value + R + G + B + protein + fat + carbs + kcal
% subs: subject id string like '008' or '016'
% run_label: run number string like '1','2',...
% model_dir: path to results/first_level_analysis/sub-.../glm_model
% model_name: name of the glm folder (e.g. 'glm_rgb_nutri')
% cnt: run count (1,2,...)
% model_name_tmp: optional filename base for saved .mat (e.g. 'glm_rgb_nutri_1')

    % create destination dir if missing
    dir_save = fullfile(model_dir, model_name);
    if ~exist(dir_save,'dir')
        mkdir(dir_save);
    end

    %%%%% 行動データディレクトリの解決 ('sub-001' または '001' を許容) %%%%%
    % Get root directory from script location
    script_dir = fileparts(mfilename('fullpath'));
    root_dir = fullfile(script_dir, '..', '..', '..');
    candidate_dirs = { fullfile(root_dir, 'Food_Behavior', ['sub-', subs]) , ...
                       fullfile(root_dir, 'Food_Behavior', subs) };
    beh_dir = '';
    for cdix = 1:numel(candidate_dirs)
        if exist(candidate_dirs{cdix}, 'dir')
            beh_dir = candidate_dirs{cdix};
            break;
        end
    end
    if isempty(beh_dir)
        error('make_glm_rgb_nutri_run:missingBehaviorDir', ...
            'Behavior directory not found for subject %s. Tried: %s and %s', ...
            subs, candidate_dirs{1}, candidate_dirs{2});
    end

    %%%%% ファイル名リスト %%%%%
    files = dir(fullfile(beh_dir, 'rating_data*.csv'));
    if ~isempty(files)
        [~, si] = sort([files.datenum]);
        files = files(si);
    end
    files_Rate_list = {files.name};

    if isempty(files_Rate_list)
        error('make_glm_rgb_nutri_run:missingRating', 'No rating_data*.csv found for subject %s', subs);
    end

    files = dir(fullfile(beh_dir, 'GLM_all*.csv'));
    if ~isempty(files)
        [~, si] = sort([files.datenum]);
        files = files(si);
    end
    files_Time_list = {files.name};
    if isempty(files_Time_list)
        error('make_glm_rgb_nutri_run:missingTiming', 'No GLM_all*.csv found for subject %s', subs);
    end

    if isequal(length(files_Rate_list), length(files_Time_list))
        disp(sprintf('Found %d behavioral CSV pairs for subject %s', length(files_Rate_list), subs));
    else
        warning('make_glm_rgb_nutri_run:mismatchedCounts', ...
            'Number of rating files (%d) and timing files (%d) differ for subject %s', ...
            length(files_Rate_list), length(files_Time_list), subs);
    end

    if ~isnumeric(cnt)
        error('make_glm_rgb_nutri_run:cnt', 'Expected numeric cnt index as 5th argument');
    end
    if cnt < 1 || cnt > min(length(files_Rate_list), length(files_Time_list))
        error('make_glm_rgb_nutri_run:cntOutOfRange', ...
            'cnt=%d is out of range for subject %s (have %d rating and %d timing files)', ...
            cnt, subs, length(files_Rate_list), length(files_Time_list));
    end

    time_filepath = fullfile(beh_dir, files_Time_list{cnt});
    rate_filepath = fullfile(beh_dir, files_Rate_list{cnt});
    fprintf('make_glm_rgb_nutri_run: subject %s cnt=%d\n  time: %s\n  rate: %s\n', subs, cnt, time_filepath, rate_filepath);

    % get the combined table
    data_tmp = make_behav_csv_run(subs, rate_filepath, time_filepath);

    % Index for NO-MISS trials
    idx_nomiss = ~(data_tmp.RatingValue == 0);

    % Image condition
    onsets_tmp = data_tmp.image;
    durations_tmp = data_tmp.question - data_tmp.image;
    names{1} = 'Image';
    onsets{1} = onsets_tmp(idx_nomiss);
    durations{1} = durations_tmp(idx_nomiss);

    % pmod: Value + R + G + B + protein + fat + carbs + kcal
    pmod(1).name{1} = 'Value';
    pmod(1).param{1} = data_tmp.RatingValue(idx_nomiss);
    pmod(1).poly{1} = 1;

    pmod(1).name{2} = 'R';
    pmod(1).param{2} = data_tmp.image_R(idx_nomiss);
    pmod(1).poly{2} = 1;

    pmod(1).name{3} = 'G';
    pmod(1).param{3} = data_tmp.image_G(idx_nomiss);
    pmod(1).poly{3} = 1;

    pmod(1).name{4} = 'B';
    pmod(1).param{4} = data_tmp.image_B(idx_nomiss);
    pmod(1).poly{4} = 1;

    pmod(1).name{5} = 'Protein';
    pmod(1).param{5} = data_tmp.image_protein(idx_nomiss);
    pmod(1).poly{5} = 1;

    pmod(1).name{6} = 'Fat';
    pmod(1).param{6} = data_tmp.image_fat(idx_nomiss);
    pmod(1).poly{6} = 1;

    pmod(1).name{7} = 'Carbs';
    pmod(1).param{7} = data_tmp.image_carbs(idx_nomiss);
    pmod(1).poly{7} = 1;

    pmod(1).name{8} = 'Kcal';
    pmod(1).param{8} = data_tmp.image_kcal(idx_nomiss);
    pmod(1).poly{8} = 1;

    orth{1} = 0;

    if isequal(std(data_tmp.RatingValue(idx_nomiss)),0)
        disp(['WARNING: NO VARIABILITY in RatingValue!!!'])
    end

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

    % Save as .mat expected by SPM
    if nargin < 6 || isempty(model_name_tmp)
        model_name_tmp = [model_name,'_',num2str(cnt)];
    end
    f_name_save = fullfile(dir_save, model_name_tmp);
    save(f_name_save, 'names', 'onsets', 'durations', 'orth', 'pmod');

    matpath = [f_name_save, '.mat'];
end
