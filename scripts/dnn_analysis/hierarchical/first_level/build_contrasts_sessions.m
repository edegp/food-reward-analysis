function build_contrasts_sessions(spm_path, source_label, num_global, shared_cols, layer_groups, specific_cols, total_sessions)
% Build contrasts for Hierarchical DNN GLM (Multi-session design)
%
% Contrasts average across all sessions
% Regressor names follow SPM pattern: Sn(1) ImageOnsetxPC_name^1
%
% Args:
%   spm_path: Path to SPM.mat
%   source_label: 'clip' or 'convnext'
%   num_global: Number of Global PCs
%   shared_cols: Cell array of Shared PC column names
%   layer_groups: Cell array of layer group names
%   specific_cols: Struct with layer-specific column names
%   total_sessions: Number of sessions (typically 12)

if ~exist(spm_path,'file'); error('SPM.mat not found: %s', spm_path); end
load(spm_path, 'SPM');

names = SPM.xX.name;
if isempty(names)
    error('SPM design appears to have no regressors.');
end

num_regressors = size(SPM.xX.X, 2);
fprintf('Design matrix size: %d scans x %d regressors\n', size(SPM.xX.X, 1), num_regressors);

matlabbatch = {};
matlabbatch{1}.spm.stats.con.spmmat = {spm_path};

ci = 0;
num_shared = numel(shared_cols);
weight = 1 / total_sessions;  % Average across sessions

fprintf('Building contrasts for Hierarchical DNN GLM (%s) - %d sessions...\n', upper(source_label), total_sessions);
fprintf('Global PCs: %d\n', num_global);
fprintf('Shared PCs: %d\n', num_shared);
fprintf('Layer Groups: %s\n', strjoin(layer_groups, ', '));

% Helper function to find columns for a pmod across all sessions
    function cols = find_pmod_cols(pmod_name)
        cols = [];
        for ses = 1:total_sessions
            % Pattern: Sn(ses) ImageOnsetxPmod_name^1
            pattern = sprintf('Sn\\(%d\\) ImageOnsetx%s\\^1', ses, pmod_name);
            for col_idx = 1:numel(names)
                if ~isempty(regexp(names{col_idx}, pattern, 'once'))
                    cols = [cols, col_idx];
                end
            end
        end
    end

%% 1. T-contrasts for Global PCs (averaged across sessions)
fprintf('\n=== Global PC T-contrasts (averaged across %d sessions) ===\n', total_sessions);
for pc = 1:num_global
    pmod_name = sprintf('Global_pc%d', pc);
    pc_cols = find_pmod_cols(pmod_name);

    if ~isempty(pc_cols)
        % Positive contrast (average across sessions)
        ci = ci + 1;
        T = zeros(1, num_regressors);
        T(pc_cols) = weight;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = sprintf('Global_pc%d_pos', pc);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';

        % Negative contrast
        ci = ci + 1;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = sprintf('Global_pc%d_neg', pc);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = -T;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';

        if pc <= 3
            fprintf('  Global_pc%d: %d session regressors\n', pc, length(pc_cols));
        end
    end
end
fprintf('  ... (total %d Global PCs)\n', num_global);

%% 2. T-contrasts for Shared PCs
fprintf('\n=== Shared PC T-contrasts ===\n');
for s = 1:num_shared
    shared_col = shared_cols{s};
    pmod_name = strrep(shared_col, '-', '_');
    pc_cols = find_pmod_cols(pmod_name);

    if ~isempty(pc_cols)
        ci = ci + 1;
        T = zeros(1, num_regressors);
        T(pc_cols) = weight;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = sprintf('%s_pos', pmod_name);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';

        ci = ci + 1;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = sprintf('%s_neg', pmod_name);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = -T;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';

        fprintf('  %s: %d session regressors\n', pmod_name, length(pc_cols));
    end
end

%% 3. T-contrasts for Layer-Specific PCs
fprintf('\n=== Layer-Specific PC T-contrasts ===\n');
for group_idx = 1:length(layer_groups)
    group_name = layer_groups{group_idx};
    n_pcs = numel(specific_cols.(group_name));

    for pc = 1:n_pcs
        pmod_name = sprintf('%s_pc%d', group_name, pc);
        pc_cols = find_pmod_cols(pmod_name);

        if ~isempty(pc_cols)
            ci = ci + 1;
            T = zeros(1, num_regressors);
            T(pc_cols) = weight;
            matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = sprintf('%s_pc%d_pos', group_name, pc);
            matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
            matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';

            ci = ci + 1;
            matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = sprintf('%s_pc%d_neg', group_name, pc);
            matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = -T;
            matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';
        end
    end
    fprintf('  %s: %d PCs\n', group_name, n_pcs);
end

%% 4. F-contrast for Global PCs
fprintf('\n=== F-contrasts ===\n');
F_rows = [];
for pc = 1:num_global
    pmod_name = sprintf('Global_pc%d', pc);
    pc_cols = find_pmod_cols(pmod_name);
    if ~isempty(pc_cols)
        F_row = zeros(1, num_regressors);
        F_row(pc_cols) = weight;
        F_rows = [F_rows; F_row];
    end
end
if ~isempty(F_rows)
    ci = ci + 1;
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.name = 'Global_F';
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.weights = F_rows;
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.sessrep = 'none';
    fprintf('  Global_F: %d rows\n', size(F_rows, 1));
end

%% 5. F-contrast for Shared PCs (combined)
F_rows = [];
for s = 1:num_shared
    shared_col = shared_cols{s};
    pmod_name = strrep(shared_col, '-', '_');
    pc_cols = find_pmod_cols(pmod_name);
    if ~isempty(pc_cols)
        F_row = zeros(1, num_regressors);
        F_row(pc_cols) = weight;
        F_rows = [F_rows; F_row];
    end
end
if ~isempty(F_rows)
    ci = ci + 1;
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.name = 'Shared_F';
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.weights = F_rows;
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.sessrep = 'none';
    fprintf('  Shared_F: %d rows\n', size(F_rows, 1));
end

%% 5b. F-contrasts for individual Shared pairs
shared_pairs = {'Initial-Middle', 'Middle-Late', 'Late-Final'};
for pair_idx = 1:length(shared_pairs)
    pair_name = shared_pairs{pair_idx};
    F_rows = [];

    for s = 1:num_shared
        shared_col = shared_cols{s};
        if contains(shared_col, pair_name)
            pmod_name = strrep(shared_col, '-', '_');
            pc_cols = find_pmod_cols(pmod_name);
            if ~isempty(pc_cols)
                F_row = zeros(1, num_regressors);
                F_row(pc_cols) = weight;
                F_rows = [F_rows; F_row];
            end
        end
    end

    if ~isempty(F_rows)
        ci = ci + 1;
        contrast_name = sprintf('Shared_%s_F', strrep(pair_name, '-', '_'));
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.name = contrast_name;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.weights = F_rows;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.sessrep = 'none';
        fprintf('  %s: %d rows\n', contrast_name, size(F_rows, 1));
    end
end

%% 6. F-contrasts for each Layer-Specific group
for group_idx = 1:length(layer_groups)
    group_name = layer_groups{group_idx};
    n_pcs = numel(specific_cols.(group_name));

    F_rows = [];
    for pc = 1:n_pcs
        pmod_name = sprintf('%s_pc%d', group_name, pc);
        pc_cols = find_pmod_cols(pmod_name);
        if ~isempty(pc_cols)
            F_row = zeros(1, num_regressors);
            F_row(pc_cols) = weight;
            F_rows = [F_rows; F_row];
        end
    end

    if ~isempty(F_rows)
        ci = ci + 1;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.name = sprintf('%s_F', group_name);
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.weights = F_rows;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.sessrep = 'none';
        fprintf('  %s_F: %d rows\n', group_name, size(F_rows, 1));
    end
end

%% 7. F-contrasts for each Layer with related Shared PCs
fprintf('\n=== Layer + Related Shared F-contrasts ===\n');

layer_shared_map = struct();
layer_shared_map.Initial = {'Initial-Middle'};
layer_shared_map.Middle = {'Initial-Middle', 'Middle-Late'};
layer_shared_map.Late = {'Middle-Late', 'Late-Final'};
layer_shared_map.Final = {'Late-Final'};

for group_idx = 1:length(layer_groups)
    group_name = layer_groups{group_idx};
    F_rows = [];

    % Add Layer-Specific PCs
    n_pcs = numel(specific_cols.(group_name));
    for pc = 1:n_pcs
        pmod_name = sprintf('%s_pc%d', group_name, pc);
        pc_cols = find_pmod_cols(pmod_name);
        if ~isempty(pc_cols)
            F_row = zeros(1, num_regressors);
            F_row(pc_cols) = weight;
            F_rows = [F_rows; F_row];
        end
    end

    % Add related Shared PCs
    if isfield(layer_shared_map, group_name)
        related_pairs = layer_shared_map.(group_name);
        for p = 1:length(related_pairs)
            pair_name = related_pairs{p};
            for s = 1:num_shared
                shared_col = shared_cols{s};
                if contains(shared_col, pair_name)
                    pmod_name = strrep(shared_col, '-', '_');
                    pc_cols = find_pmod_cols(pmod_name);
                    if ~isempty(pc_cols)
                        F_row = zeros(1, num_regressors);
                        F_row(pc_cols) = weight;
                        F_rows = [F_rows; F_row];
                    end
                end
            end
        end
    end

    if ~isempty(F_rows)
        ci = ci + 1;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.name = sprintf('%s_withShared_F', group_name);
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.weights = F_rows;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.sessrep = 'none';
        fprintf('  %s_withShared_F: %d rows\n', group_name, size(F_rows, 1));
    end
end

%% 7b. F-contrasts for each Layer with related Shared PCs AND Global
fprintf('\n=== Layer + Related Shared + Global F-contrasts ===\n');

for group_idx = 1:length(layer_groups)
    group_name = layer_groups{group_idx};
    F_rows = [];

    % Add Global PCs
    for pc = 1:num_global
        pmod_name = sprintf('Global_pc%d', pc);
        pc_cols = find_pmod_cols(pmod_name);
        if ~isempty(pc_cols)
            F_row = zeros(1, num_regressors);
            F_row(pc_cols) = weight;
            F_rows = [F_rows; F_row];
        end
    end

    % Add Layer-Specific PCs
    n_pcs = numel(specific_cols.(group_name));
    for pc = 1:n_pcs
        pmod_name = sprintf('%s_pc%d', group_name, pc);
        pc_cols = find_pmod_cols(pmod_name);
        if ~isempty(pc_cols)
            F_row = zeros(1, num_regressors);
            F_row(pc_cols) = weight;
            F_rows = [F_rows; F_row];
        end
    end

    % Add related Shared PCs
    if isfield(layer_shared_map, group_name)
        related_pairs = layer_shared_map.(group_name);
        for p = 1:length(related_pairs)
            pair_name = related_pairs{p};
            for s = 1:num_shared
                shared_col = shared_cols{s};
                if contains(shared_col, pair_name)
                    pmod_name = strrep(shared_col, '-', '_');
                    pc_cols = find_pmod_cols(pmod_name);
                    if ~isempty(pc_cols)
                        F_row = zeros(1, num_regressors);
                        F_row(pc_cols) = weight;
                        F_rows = [F_rows; F_row];
                    end
                end
            end
        end
    end

    if ~isempty(F_rows)
        ci = ci + 1;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.name = sprintf('%s_withShared_withGlobal_F', group_name);
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.weights = F_rows;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.sessrep = 'none';
        fprintf('  %s_withShared_withGlobal_F: %d rows\n', group_name, size(F_rows, 1));
    end
end

%% 8. F-contrast for All Layer-Specific PCs combined
F_rows = [];
for group_idx = 1:length(layer_groups)
    group_name = layer_groups{group_idx};
    n_pcs = numel(specific_cols.(group_name));
    for pc = 1:n_pcs
        pmod_name = sprintf('%s_pc%d', group_name, pc);
        pc_cols = find_pmod_cols(pmod_name);
        if ~isempty(pc_cols)
            F_row = zeros(1, num_regressors);
            F_row(pc_cols) = weight;
            F_rows = [F_rows; F_row];
        end
    end
end

if ~isempty(F_rows)
    ci = ci + 1;
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.name = 'AllSpecific_F';
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.weights = F_rows;
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.sessrep = 'none';
    fprintf('  AllSpecific_F: %d rows\n', size(F_rows, 1));
end

%% 9. F-contrast for All PCs (Global + Shared + Specific)
F_rows = [];

% Add Global
for pc = 1:num_global
    pmod_name = sprintf('Global_pc%d', pc);
    pc_cols = find_pmod_cols(pmod_name);
    if ~isempty(pc_cols)
        F_row = zeros(1, num_regressors);
        F_row(pc_cols) = weight;
        F_rows = [F_rows; F_row];
    end
end

% Add Shared
for s = 1:num_shared
    shared_col = shared_cols{s};
    pmod_name = strrep(shared_col, '-', '_');
    pc_cols = find_pmod_cols(pmod_name);
    if ~isempty(pc_cols)
        F_row = zeros(1, num_regressors);
        F_row(pc_cols) = weight;
        F_rows = [F_rows; F_row];
    end
end

% Add Specific
for group_idx = 1:length(layer_groups)
    group_name = layer_groups{group_idx};
    n_pcs = numel(specific_cols.(group_name));
    for pc = 1:n_pcs
        pmod_name = sprintf('%s_pc%d', group_name, pc);
        pc_cols = find_pmod_cols(pmod_name);
        if ~isempty(pc_cols)
            F_row = zeros(1, num_regressors);
            F_row(pc_cols) = weight;
            F_rows = [F_rows; F_row];
        end
    end
end

if ~isempty(F_rows)
    ci = ci + 1;
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.name = 'AllPCs_F';
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.weights = F_rows;
    matlabbatch{1}.spm.stats.con.consess{ci}.fcon.sessrep = 'none';
    fprintf('  AllPCs_F: %d rows\n', size(F_rows, 1));
end

matlabbatch{1}.spm.stats.con.delete = 1;  % Delete existing contrasts

fprintf('\nTotal contrasts: %d\n', ci);
fprintf('Running contrast estimation...\n');

spm_jobman('run', matlabbatch);

fprintf('Contrast estimation complete!\n');

end
