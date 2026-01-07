function matpath = make_lss_glm_run(subs, run_label, model_dir, model_name, cnt, model_name_tmp)
%% Create LSS GLM design for a single subject-run
% Each image presentation is modeled as a separate regressor
% This allows estimation of individual beta values for each image
%
% Inputs:
%   subs: subject id string like '001'
%   run_label: run number string like '1','2',...
%   model_dir: path to results/first_level_analysis/sub-.../glm_model
%   model_name: name of the glm folder (e.g. 'lss_glm')
%   cnt: run count (1,2,...)
%   model_name_tmp: optional filename base for saved .mat

    % Create destination dir if missing
    dir_save = fullfile(model_dir, model_name);
    if ~exist(dir_save,'dir')
        mkdir(dir_save);
    end

    %%%%% Behavior directory resolution %%%%%
    % Get project root directory (go up from scripts/first_level_analysis/lss_model)
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
        error('make_lss_glm_run:missingBehaviorDir', ...
            'Behavior directory not found for subject %s. Tried: %s and %s', ...
            subs, candidate_dirs{1}, candidate_dirs{2});
    end

    %%%%% File lists %%%%%
    files = dir(fullfile(beh_dir, 'rating_data*.csv'));
    if ~isempty(files)
        [~, si] = sort([files.datenum]);
        files = files(si);
    end
    files_Rate_list = {files.name};

    if isempty(files_Rate_list)
        error('make_lss_glm_run:missingRating', ...
            'No rating_data*.csv found for subject %s', subs);
    end

    files = dir(fullfile(beh_dir, 'GLM_all*.csv'));
    if ~isempty(files)
        [~, si] = sort([files.datenum]);
        files = files(si);
    end
    files_Time_list = {files.name};

    if isempty(files_Time_list)
        error('make_lss_glm_run:missingTiming', ...
            'No GLM_all*.csv found for subject %s', subs);
    end

    if cnt < 1 || cnt > min(length(files_Rate_list), length(files_Time_list))
        error('make_lss_glm_run:cntOutOfRange', ...
            'cnt=%d is out of range for subject %s', cnt, subs);
    end

    time_filepath = fullfile(beh_dir, files_Time_list{cnt});
    rate_filepath = fullfile(beh_dir, files_Rate_list{cnt});
    fprintf('make_lss_glm_run: subject %s cnt=%d\n  time: %s\n  rate: %s\n', ...
            subs, cnt, time_filepath, rate_filepath);

    % Get combined table
    data_tmp = make_behav_csv_run(subs, rate_filepath, time_filepath);

    % Index for NO-MISS trials
    idx_nomiss = ~(data_tmp.RatingValue == 0);

    %%%%% LSS: Each image as separate regressor %%%%%
    % We'll create one regressor per image presentation
    % Image IDs are stored in data_tmp.ImageName

    names = {};
    onsets = {};
    durations = {};
    orth = {};

    % Get image onset/duration data
    image_onsets = data_tmp.image(idx_nomiss);
    image_durations = data_tmp.question(idx_nomiss) - data_tmp.image(idx_nomiss);
    image_ids = data_tmp.ImageName(idx_nomiss);

    % Convert to cell array of strings if needed
    if isnumeric(image_ids)
        image_ids = arrayfun(@(x) sprintf('%04d', x), image_ids, 'UniformOutput', false);
    elseif ~iscell(image_ids)
        image_ids = cellstr(string(image_ids));
    end

    % Create one regressor per image
    for i = 1:length(image_ids)
        img_id_str = sprintf('Image_%s', image_ids{i});
        names{i} = img_id_str;
        onsets{i} = image_onsets(i);
        durations{i} = image_durations(i);
        orth{i} = 0;
    end

    % Add other regressors (Question, Response, Feedback)
    n_img = length(image_ids);

    % Question
    onsets_tmp = data_tmp.question(idx_nomiss);
    durations_tmp = data_tmp.rating(idx_nomiss) - data_tmp.question(idx_nomiss);
    names{n_img+1} = 'Question';
    onsets{n_img+1} = onsets_tmp;
    durations{n_img+1} = durations_tmp;
    orth{n_img+1} = 0;

    % Response
    onsets_tmp = data_tmp.rating(idx_nomiss);
    durations_tmp = zeros(length(onsets_tmp),1);
    names{n_img+2} = 'Response';
    onsets{n_img+2} = onsets_tmp;
    durations{n_img+2} = durations_tmp;
    orth{n_img+2} = 0;

    % Feedback
    onsets_tmp = data_tmp.rating(idx_nomiss);
    durations_tmp = ones(length(onsets_tmp),1) * 0.5;
    names{n_img+3} = 'Feedback';
    onsets{n_img+3} = onsets_tmp;
    durations{n_img+3} = durations_tmp;
    orth{n_img+3} = 0;

    % Miss trials (if any)
    if sum(~idx_nomiss) > 0
        onsets_tmp = data_tmp.image(~idx_nomiss);
        durations_tmp = data_tmp.rating(~idx_nomiss) - data_tmp.image(~idx_nomiss);
        names{n_img+4} = 'Miss';
        onsets{n_img+4} = onsets_tmp;
        durations{n_img+4} = durations_tmp;
        orth{n_img+4} = 0;
    end

    % Save as .mat expected by SPM
    % Note: pmod (parametric modulation) not needed for LSS
    if nargin < 6 || isempty(model_name_tmp)
        model_name_tmp = [model_name,'_',num2str(cnt)];
    end
    f_name_save = fullfile(dir_save, model_name_tmp);
    save(f_name_save, 'names', 'onsets', 'durations', 'orth');

    % Return full path
    matpath = [f_name_save, '.mat'];

    fprintf('  Created LSS GLM with %d image regressors\n', length(image_ids));
end
