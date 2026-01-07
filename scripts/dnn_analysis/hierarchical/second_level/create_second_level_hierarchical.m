function create_second_level_hierarchical(source_label, varargin)
%% Create 2nd-level analysis for Hierarchical DNN GLM
%
% Uses Layer+Shared F-contrasts approach
% Uses con images from T-contrasts and builds F-contrasts at second level
%
% For each LayerGroup (Initial, Middle, Late, Final):
%   - Collect con images for Layer-Specific PCs (pos direction only)
%   - Collect con images for Related Shared PCs
%   - Build flexible factorial design: Subject x PC (interaction)
%   - Create F-contrast for combined effect
%
% source_label: 'convnext' or 'clip'

% Parse inputs
p = inputParser;
addRequired(p, 'source_label', @(x) ismember(x, {'convnext', 'clip'}));
addParameter(p, 'subjects', arrayfun(@(x) sprintf('%03d', x), 1:31, 'UniformOutput', false));
addParameter(p, 'output_dir', '');
addParameter(p, 'separate_sessions', true);  % Whether first-level used separate SPM sessions
addParameter(p, 'daywise', false);  % Whether first-level used daywise design
parse(p, source_label, varargin{:});

subjects = p.Results.subjects;
output_dir = p.Results.output_dir;
separate_sessions = p.Results.separate_sessions;
daywise = p.Results.daywise;

% Setup paths
project_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(project_dir, '..', '..', '..', '..');

% Initialize SPM
spm('defaults', 'fmri');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);

% Define structure based on source
if strcmp(source_label, 'clip')
    num_global = 28;
else
    num_global = 22;
end

% Define layer groups and their related shared PCs
% Format: {LayerName, NumSpecificPCs, {RelatedSharedPairs}}
layer_groups = {
    'Initial', 2, {'Initial_Middle'}
    'Middle', 3, {'Initial_Middle', 'Middle_Late'}
    'Late', 2, {'Middle_Late', 'Late_Final'}
    'Final', 2, {'Late_Final'}
};

% Shared PC pairs (2 PCs each)
shared_pairs = {'Initial_Middle', 'Middle_Late', 'Late_Final'};

% Output directory
if isempty(output_dir)
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    if daywise
        output_dir = fullfile(root_dir, 'results', 'dnn_analysis', 'second_level', ...
                             ['hierarchical_', source_label, '_v3_daywise'], timestamp);
    elseif separate_sessions
        output_dir = fullfile(root_dir, 'results', 'dnn_analysis', 'second_level', ...
                             ['hierarchical_', source_label, '_v3_sessions'], timestamp);
    else
        output_dir = fullfile(root_dir, 'results', 'dnn_analysis', 'second_level', ...
                             ['hierarchical_', source_label, '_v3'], timestamp);
    end
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('Creating 2nd-level analysis for Hierarchical DNN GLM (%s)\n', upper(source_label));
fprintf('  Approach: Layer + Related Shared PCs combined\n');
if daywise
    fprintf('  First-level: daywise design (3 days, PMODs grouped by day)\n');
elseif separate_sessions
    fprintf('  First-level: separate SPM sessions (12 sessions per subject)\n');
else
    fprintf('  First-level: single SPM session (all runs combined)\n');
end
fprintf('  Output: %s\n', output_dir);

% First, collect all valid subjects and their SPM directories
fprintf('\nVerifying subject data...\n');
valid_subjects = {};

for s = 1:length(subjects)
    subj_id = subjects{s};

    % Find latest GLM directory
    if daywise
        model_name = ['glm_dnn_pmods_hierarchical_', source_label, '_daywise'];
    elseif separate_sessions
        model_name = ['glm_dnn_pmods_hierarchical_', source_label, '_sessions'];
    else
        model_name = ['glm_dnn_pmods_hierarchical_', source_label];
    end
    model_dir = fullfile(root_dir, 'results', 'first_level_analysis', ...
                        ['sub-', subj_id], 'glm_model', model_name);

    if ~exist(model_dir, 'dir')
        continue;
    end

    % Find latest run
    d = dir(model_dir);
    d = d([d.isdir] & ~startsWith({d.name}, '.'));
    if isempty(d)
        continue;
    end
    [~, idx] = max([d.datenum]);
    latest_dir = fullfile(model_dir, d(idx).name);

    % Check SPM.mat exists
    spm_path = fullfile(latest_dir, 'SPM.mat');
    if ~exist(spm_path, 'file')
        continue;
    end

    valid_subjects{end+1} = struct('id', subj_id, 'dir', latest_dir);
end

fprintf('Valid subjects: %d / %d\n', length(valid_subjects), length(subjects));
num_subjects = length(valid_subjects);

if num_subjects < 2
    error('Not enough valid subjects');
end

% Load one SPM.mat to get contrast name to index mapping
load(fullfile(valid_subjects{1}.dir, 'SPM.mat'), 'SPM');
con_name_to_idx = containers.Map();
for i = 1:length(SPM.xCon)
    con_name_to_idx(SPM.xCon(i).name) = i;
end

%% Process each LayerGroup - both with and without Shared PCs
for lg_idx = 1:size(layer_groups, 1)
    layer_name = layer_groups{lg_idx, 1};
    num_specific = layer_groups{lg_idx, 2};
    related_shared = layer_groups{lg_idx, 3};

    % ========== First: Layer-Specific ONLY (no Shared) ==========
    fprintf('\n========================================\n');
    fprintf('Processing %s_only (Layer-Specific only)\n', layer_name);
    fprintf('========================================\n');

    contrast_names_only = {};
    for pc = 1:num_specific
        con_name = sprintf('%s_pc%d_pos', layer_name, pc);
        if con_name_to_idx.isKey(con_name)
            contrast_names_only{end+1} = con_name;
        end
    end

    if ~isempty(contrast_names_only)
        process_layer_group(layer_name, '_only', contrast_names_only, ...
            valid_subjects, con_name_to_idx, output_dir, num_subjects);
    end

    % ========== Second: Layer + Shared ==========
    fprintf('\n========================================\n');
    fprintf('Processing %s_withShared\n', layer_name);
    fprintf('========================================\n');

    % Build list of contrast names for this group
    contrast_names = {};

    % Layer-specific PCs (pos direction)
    for pc = 1:num_specific
        con_name = sprintf('%s_pc%d_pos', layer_name, pc);
        if con_name_to_idx.isKey(con_name)
            contrast_names{end+1} = con_name;
        end
    end

    % Related Shared PCs (pos direction)
    for sh_idx = 1:length(related_shared)
        shared_pair = related_shared{sh_idx};
        for pc = 1:2  % 2 PCs per shared pair
            con_name = sprintf('Shared_%s_pc%d_pos', shared_pair, pc);
            if con_name_to_idx.isKey(con_name)
                contrast_names{end+1} = con_name;
            end
        end
    end

    if ~isempty(contrast_names)
        process_layer_group(layer_name, '_withShared', contrast_names, ...
            valid_subjects, con_name_to_idx, output_dir, num_subjects);
    end

    % ========== Third: Layer + Shared + Global ==========
    fprintf('\n========================================\n');
    fprintf('Processing %s_withShared_withGlobal\n', layer_name);
    fprintf('========================================\n');

    contrast_names = {};

    % Global PCs (pos direction)
    for pc = 1:num_global
        con_name = sprintf('Global_pc%d_pos', pc);
        if con_name_to_idx.isKey(con_name)
            contrast_names{end+1} = con_name;
        end
    end

    % Layer-specific PCs (pos direction)
    for pc = 1:num_specific
        con_name = sprintf('%s_pc%d_pos', layer_name, pc);
        if con_name_to_idx.isKey(con_name)
            contrast_names{end+1} = con_name;
        end
    end

    % Related Shared PCs (pos direction)
    for sh_idx = 1:length(related_shared)
        shared_pair = related_shared{sh_idx};
        for pc = 1:2
            con_name = sprintf('Shared_%s_pc%d_pos', shared_pair, pc);
            if con_name_to_idx.isKey(con_name)
                contrast_names{end+1} = con_name;
            end
        end
    end

    if ~isempty(contrast_names)
        process_layer_group(layer_name, '_withShared_withGlobal', contrast_names, ...
            valid_subjects, con_name_to_idx, output_dir, num_subjects);
    end
end

%% Also process Global and Shared separately
% Global PCs
fprintf('\n========================================\n');
fprintf('Processing Global_F\n');
fprintf('========================================\n');

contrast_names = {};
for pc = 1:num_global
    con_name = sprintf('Global_pc%d_pos', pc);
    if con_name_to_idx.isKey(con_name)
        contrast_names{end+1} = con_name;
    end
end

if ~isempty(contrast_names)
    process_group('Global', contrast_names, valid_subjects, con_name_to_idx, output_dir);
end

% Shared PCs (combined)
fprintf('\n========================================\n');
fprintf('Processing Shared_F\n');
fprintf('========================================\n');

contrast_names = {};
for sh_idx = 1:length(shared_pairs)
    shared_pair = shared_pairs{sh_idx};
    for pc = 1:2
        con_name = sprintf('Shared_%s_pc%d_pos', shared_pair, pc);
        if con_name_to_idx.isKey(con_name)
            contrast_names{end+1} = con_name;
        end
    end
end

if ~isempty(contrast_names)
    process_group('Shared', contrast_names, valid_subjects, con_name_to_idx, output_dir);
end

% Individual Shared pairs
for sh_idx = 1:length(shared_pairs)
    shared_pair = shared_pairs{sh_idx};
    fprintf('\n========================================\n');
    fprintf('Processing Shared_%s_F\n', shared_pair);
    fprintf('========================================\n');

    contrast_names = {};
    for pc = 1:2
        con_name = sprintf('Shared_%s_pc%d_pos', shared_pair, pc);
        if con_name_to_idx.isKey(con_name)
            contrast_names{end+1} = con_name;
        end
    end

    if ~isempty(contrast_names)
        process_group(['Shared_', shared_pair], contrast_names, valid_subjects, con_name_to_idx, output_dir);
    end
end

fprintf('\n========================================\n');
fprintf('2nd-level analysis completed!\n');
fprintf('Output: %s\n', output_dir);
fprintf('========================================\n');

end

function create_group_fcontrast(spm_dir, group_name, num_conditions)
    % Create F-contrast after model estimation
    % The interaction design creates Subject*PC columns

    spm_path = fullfile(spm_dir, 'SPM.mat');
    load(spm_path, 'SPM');

    % Design matrix size
    num_cols = size(SPM.xX.X, 2);
    fprintf('    Design matrix columns: %d\n', num_cols);

    % For Subject x PC interaction, columns correspond to Subject-PC combinations
    % F-contrast: test effect of PC (across subjects)
    % Create identity-like matrix for all columns (main effect of all conditions)
    F = eye(num_cols);

    matlabbatch = {};
    matlabbatch{1}.spm.stats.con.spmmat = {spm_path};
    matlabbatch{1}.spm.stats.con.consess{1}.fcon.name = [group_name, '_withShared_F'];
    matlabbatch{1}.spm.stats.con.consess{1}.fcon.weights = F;
    matlabbatch{1}.spm.stats.con.consess{1}.fcon.sessrep = 'none';
    matlabbatch{1}.spm.stats.con.delete = 1;

    try
        spm_jobman('run', matlabbatch);
        fprintf('  F-contrast created: %s_withShared_F\n', group_name);
    catch ME
        fprintf('  Error creating contrast: %s\n', ME.message);
    end
end

function process_layer_group(layer_name, suffix, contrast_names, valid_subjects, con_name_to_idx, output_dir, num_subjects)
    % Process a layer group (either _only or _withShared)
    num_conditions = length(contrast_names);

    fprintf('  Conditions: %d\n', num_conditions);
    for c = 1:num_conditions
        fprintf('    %d: %s\n', c, contrast_names{c});
    end

    % Collect con files
    all_scans = cell(num_subjects, num_conditions);
    for s = 1:num_subjects
        subj = valid_subjects{s};
        for c = 1:num_conditions
            con_name = contrast_names{c};
            con_idx = con_name_to_idx(con_name);
            con_file = fullfile(subj.dir, sprintf('con_%04d.nii', con_idx));
            if exist(con_file, 'file')
                all_scans{s, c} = con_file;
            end
        end
    end

    missing = sum(cellfun(@isempty, all_scans(:)));
    if missing > 0
        fprintf('  Warning: %d missing files\n', missing);
    end

    % Create output directory
    group_output_dir = fullfile(output_dir, [layer_name, suffix]);
    if ~exist(group_output_dir, 'dir')
        mkdir(group_output_dir);
    end

    % Build flexible factorial design
    matlabbatch = {};
    matlabbatch{1}.spm.stats.factorial_design.dir = {group_output_dir};

    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).name = 'Subject';
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).dept = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).variance = 1;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).gmsca = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).ancova = 0;

    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).name = 'PC';
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).dept = 1;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).variance = 1;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).gmsca = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).ancova = 0;

    scans_list = {};
    imatrix = [];
    for cond_idx = 1:num_conditions
        for subj_idx = 1:num_subjects
            if ~isempty(all_scans{subj_idx, cond_idx})
                scans_list{end+1, 1} = all_scans{subj_idx, cond_idx};
                imatrix = [imatrix; subj_idx, cond_idx];
            end
        end
    end

    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.specall.scans = scans_list;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.specall.imatrix = imatrix;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.maininters{1}.inter.fnums = [1; 2];

    matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(group_output_dir, 'SPM.mat')};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    try
        fprintf('  Running SPM batch...\n');
        spm_jobman('run', matlabbatch);
        fprintf('  Model estimation complete.\n');

        % Create F-contrast
        create_layer_fcontrast(group_output_dir, layer_name, suffix);
    catch ME
        fprintf('  Error: %s\n', ME.message);
    end
end

function create_layer_fcontrast(spm_dir, layer_name, suffix)
    spm_path = fullfile(spm_dir, 'SPM.mat');
    load(spm_path, 'SPM');

    num_cols = size(SPM.xX.X, 2);
    fprintf('    Design matrix columns: %d\n', num_cols);

    F = eye(num_cols);

    % Name based on suffix
    if strcmp(suffix, '_only')
        con_name = [layer_name, '_F'];
    else
        con_name = [layer_name, '_withShared_F'];
    end

    matlabbatch = {};
    matlabbatch{1}.spm.stats.con.spmmat = {spm_path};
    matlabbatch{1}.spm.stats.con.consess{1}.fcon.name = con_name;
    matlabbatch{1}.spm.stats.con.consess{1}.fcon.weights = F;
    matlabbatch{1}.spm.stats.con.consess{1}.fcon.sessrep = 'none';
    matlabbatch{1}.spm.stats.con.delete = 1;

    try
        spm_jobman('run', matlabbatch);
        fprintf('  F-contrast created: %s\n', con_name);
    catch ME
        fprintf('  Error creating contrast: %s\n', ME.message);
    end
end

function process_group(group_name, contrast_names, valid_subjects, con_name_to_idx, output_dir)
    num_subjects = length(valid_subjects);
    num_conditions = length(contrast_names);

    fprintf('  Conditions: %d\n', num_conditions);

    % Collect con files
    all_scans = cell(num_subjects, num_conditions);
    for s = 1:num_subjects
        subj = valid_subjects{s};
        for c = 1:num_conditions
            con_name = contrast_names{c};
            con_idx = con_name_to_idx(con_name);
            con_file = fullfile(subj.dir, sprintf('con_%04d.nii', con_idx));
            if exist(con_file, 'file')
                all_scans{s, c} = con_file;
            end
        end
    end

    % Create output directory
    group_output_dir = fullfile(output_dir, [group_name, '_F']);
    if ~exist(group_output_dir, 'dir')
        mkdir(group_output_dir);
    end

    % Build batch with interaction design
    matlabbatch = {};
    matlabbatch{1}.spm.stats.factorial_design.dir = {group_output_dir};

    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).name = 'Subject';
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).dept = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).variance = 1;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).gmsca = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).ancova = 0;

    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).name = 'PC';
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).dept = 1;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).variance = 1;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).gmsca = 0;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).ancova = 0;

    scans_list = {};
    imatrix = [];
    for cond_idx = 1:num_conditions
        for subj_idx = 1:num_subjects
            if ~isempty(all_scans{subj_idx, cond_idx})
                scans_list{end+1, 1} = all_scans{subj_idx, cond_idx};
                imatrix = [imatrix; subj_idx, cond_idx];
            end
        end
    end

    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.specall.scans = scans_list;
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.specall.imatrix = imatrix;

    % Use Subject × PC interaction
    matlabbatch{1}.spm.stats.factorial_design.des.fblock.maininters{1}.inter.fnums = [1; 2];

    matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(group_output_dir, 'SPM.mat')};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    try
        fprintf('  Running SPM batch (design + estimation)...\n');
        spm_jobman('run', matlabbatch);
        fprintf('  Model estimation complete.\n');

        % Create F-contrast
        create_group_fcontrast(group_output_dir, group_name, num_conditions);
    catch ME
        fprintf('  Error: %s\n', ME.message);
    end
end
