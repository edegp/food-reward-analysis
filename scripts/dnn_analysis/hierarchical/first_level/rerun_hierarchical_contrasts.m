function rerun_hierarchical_contrasts(sub_id, source)
%% Rerun contrasts for a single subject
% This function clears existing contrasts and rebuilds them with new ones
% including individual Shared F-contrasts and Layer+Shared+Global F-contrasts

    script_dir = fileparts(mfilename('fullpath'));
    root_dir = fullfile(script_dir, '..', '..', '..', '..');

    % Find SPM.mat (Design N = hierarchical)
    model_name = ['glm_dnn_pmods_designN_', source];
    model_dir = fullfile(root_dir, 'results', 'first_level_analysis', ...
                       ['sub-', sub_id], 'glm_model', model_name);

    if ~exist(model_dir, 'dir')
        error('Model directory not found: %s', model_dir);
    end

    % Find latest run directory
    d = dir(model_dir);
    d = d([d.isdir] & ~startsWith({d.name}, '.'));
    if isempty(d)
        error('No run directories found in: %s', model_dir);
    end
    [~, idx] = max([d.datenum]);
    spm_dir = fullfile(model_dir, d(idx).name);
    spm_mat = fullfile(spm_dir, 'SPM.mat');

    if ~exist(spm_mat, 'file')
        error('SPM.mat not found: %s', spm_mat);
    end

    fprintf('Loading SPM.mat from: %s\n', spm_dir);

    % Load 3-level PC info to get column names
    pc3_path = fullfile(root_dir, 'data_images', 'dnn_pmods', '3level', ...
        [source, '_3level_pcs.csv']);
    pc3_table = readtable(pc3_path, 'VariableNamingRule', 'preserve');

    % Identify PC columns
    all_cols = pc3_table.Properties.VariableNames;
    global_cols = all_cols(startsWith(all_cols, 'Global_pc'));
    shared_cols = all_cols(startsWith(all_cols, 'Shared_'));

    % Layer-specific columns
    layer_groups = {'Initial', 'Middle', 'Late', 'Final'};
    specific_cols = struct();
    for c = 1:numel(all_cols)
        col = all_cols{c};
        if contains(col, '_pc') && ~startsWith(col, 'Global') && ~startsWith(col, 'Shared')
            parts = strsplit(col, '_pc');
            layer_name = parts{1};
            if ~isfield(specific_cols, layer_name)
                specific_cols.(layer_name) = {};
            end
            specific_cols.(layer_name){end+1} = col;
        end
    end

    num_global = numel(global_cols);

    % Build contrasts
    build_contrasts(spm_mat, source, num_global, shared_cols, layer_groups, specific_cols);

    fprintf('Contrasts rebuilt for sub-%s (%s)\n', sub_id, source);
end
