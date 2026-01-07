function extract_beta_info()
%% Extract beta information from LSS GLM SPM.mat files
% Creates beta_info.csv mapping image IDs to beta indices

project_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(project_dir, '..', '..', '..');
result_dir = fullfile(root_dir, 'results', 'first_level_analysis');

% Subject list
subs_list = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010', ...
             '011', '012', '013', '014', '015', '016', '017', '018', '019', '020'};

fprintf('Extracting beta info for %d subjects...\n', length(subs_list));

for i = 1:length(subs_list)
    sub_id = subs_list{i};
    fprintf('Processing subject %s...\n', sub_id);

    % Find latest LSS GLM directory
    sub_glm_dir = fullfile(result_dir, ['sub-', sub_id], 'glm_model', 'lss_glm');
    if ~exist(sub_glm_dir, 'dir')
        fprintf('  WARNING: LSS GLM directory not found for subject %s\n', sub_id);
        continue;
    end

    % Get latest timestamp directory
    dir_contents = dir(sub_glm_dir);
    timestamp_dirs = {dir_contents([dir_contents.isdir] & ~strcmp({dir_contents.name}, '.') & ~strcmp({dir_contents.name}, '..')).name};

    if isempty(timestamp_dirs)
        fprintf('  WARNING: No LSS GLM results found for subject %s\n', sub_id);
        continue;
    end

    % Sort and get latest
    timestamp_dirs = sort(timestamp_dirs);
    latest_dir = fullfile(sub_glm_dir, timestamp_dirs{end});

    % Load SPM.mat
    spm_file = fullfile(latest_dir, 'SPM.mat');
    if ~exist(spm_file, 'file')
        fprintf('  WARNING: SPM.mat not found for subject %s\n', sub_id);
        continue;
    end

    try
        load(spm_file, 'SPM');
    catch
        fprintf('  WARNING: Could not load SPM.mat for subject %s\n', sub_id);
        continue;
    end

    % Extract image regressor names and indices
    image_ids = {};
    beta_indices = [];

    % Go through all sessions and regressors
    beta_idx = 1;
    for sess = 1:length(SPM.Sess)
        for reg = 1:length(SPM.Sess(sess).U)
            reg_name = SPM.Sess(sess).U(reg).name{1};

            % Check if this is an image regressor (starts with "Image_")
            if startsWith(reg_name, 'Image_')
                % Extract image ID (e.g., "Image_0001" -> "0001")
                img_id = strrep(reg_name, 'Image_', '');
                image_ids{end+1} = img_id;
                beta_indices(end+1) = beta_idx;
            end

            beta_idx = beta_idx + 1;
        end

        % Skip confound regressors (movement, etc.)
        % SPM.Sess(sess).C contains confound regressors
        if isfield(SPM.Sess(sess), 'C') && ~isempty(SPM.Sess(sess).C.C)
            n_confounds = size(SPM.Sess(sess).C.C, 2);
            beta_idx = beta_idx + n_confounds;
        end
    end

    % Create beta_info table
    beta_info = table(image_ids(:), beta_indices(:), ...
                     'VariableNames', {'image_id', 'beta_index'});

    % Save to CSV
    output_dir = fullfile(latest_dir, 'beta_values');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    output_file = fullfile(output_dir, 'beta_info.csv');
    writetable(beta_info, output_file);

    fprintf('  Saved beta info: %d images -> %s\n', length(image_ids), output_file);
end

fprintf('Done!\n');
end
