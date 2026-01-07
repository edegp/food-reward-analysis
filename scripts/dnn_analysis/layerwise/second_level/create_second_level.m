function create_second_level_designI(source_label, varargin)
%% Create 2nd-level analysis for Design I (Low-Correlation 3-Layer Analysis)
%
% Design I structure:
%   - 3 layers: Early, Middle, Late
%   - 3 PCs per layer: PC1, PC2, PC3
%   - Total: 9 conditions
%
% source_label: 'convnext' or 'clip'
% Optional name-value pairs:
%   'subjects': cell array of subject IDs (default: all 31 subjects)
%   'output_dir': output directory (default: auto-generated)

% Parse inputs
p = inputParser;
addRequired(p, 'source_label', @(x) ismember(x, {'convnext', 'clip'}));
addParameter(p, 'subjects', arrayfun(@(x) sprintf('%03d', x), 1:31, 'UniformOutput', false));
addParameter(p, 'output_dir', '');
parse(p, source_label, varargin{:});

subjects = p.Results.subjects;
output_dir = p.Results.output_dir;

% Setup paths
project_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(project_dir, '..', '..', '..', '..');

% Initialize SPM
spm('defaults', 'fmri');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);

% Define layers and PCs for Design I
layers = {'Early', 'Middle', 'Late'};
num_pcs = 3;

% Build condition list
conditions = {};
condition_to_layer = {};

for layer_idx = 1:length(layers)
    layer_name = layers{layer_idx};
    for pc = 1:num_pcs
        conditions{end+1} = sprintf('%s PC%d', layer_name, pc);
        condition_to_layer{end+1} = layer_name;
    end
end

num_conditions = length(conditions);
fprintf('Total conditions (Layer × PC): %d\n', num_conditions);

% Output directory
if isempty(output_dir)
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    output_dir = fullfile(root_dir, 'results', 'dnn_analysis', 'second_level', ...
                         ['designI_', source_label], timestamp);
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('Creating 2nd-level analysis (Design I) for %s\n', upper(source_label));
fprintf('  Design: Subject × Layer × PC (%d conditions)\n', num_conditions);
fprintf('  Output: %s\n', output_dir);

% Collect contrast images from each subject
fprintf('Collecting contrast images from %d subjects...\n', length(subjects));

% Initialize contrast_files struct
contrast_files = struct();
for cond_idx = 1:num_conditions
    field_name = matlab.lang.makeValidName(conditions{cond_idx});
    contrast_files.(field_name) = {};
end

% Iterate by subject
for s = 1:length(subjects)
    subj_id = subjects{s};
    fprintf('  Subject %s...', subj_id);

    % Find latest GLM directory for Design I
    model_name = ['glm_dnn_pmods_designI_', source_label];
    model_dir = fullfile(root_dir, 'results', 'first_level_analysis', ...
                       ['sub-', subj_id], 'glm_model', model_name);

    % Read latest_run.txt
    latest_marker = fullfile(model_dir, 'latest_run.txt');
    if ~exist(latest_marker, 'file')
        d = dir(model_dir);
        d = d([d.isdir] & ~startsWith({d.name}, '.'));
        if isempty(d)
            fprintf(' not found\n');
            continue;
        end
        [~, idx] = max([d.datenum]);
        latest_dir = fullfile(model_dir, d(idx).name);
    else
        fid = fopen(latest_marker, 'r');
        latest_dir = strtrim(fgetl(fid));
        fclose(fid);
    end

    % Load SPM.mat
    spm_path = fullfile(latest_dir, 'SPM.mat');
    if ~exist(spm_path, 'file')
        fprintf(' SPM.mat not found\n');
        continue;
    end

    load(spm_path, 'SPM');

    % Build contrast name to index map
    con_map = containers.Map();
    if isfield(SPM, 'xCon') && ~isempty(SPM.xCon)
        for i = 1:length(SPM.xCon)
            con_map(SPM.xCon(i).name) = i;
        end
    end

    % Collect all contrasts for this subject
    num_found = 0;
    for cond_idx = 1:num_conditions
        condition_name = conditions{cond_idx};
        field_name = matlab.lang.makeValidName(condition_name);

        if con_map.isKey(condition_name)
            con_idx_val = con_map(condition_name);
            con_file = fullfile(latest_dir, sprintf('con_%04d.nii', con_idx_val));
            if exist(con_file, 'file')
                contrast_files.(field_name){end+1} = con_file;
                num_found = num_found + 1;
            end
        end
    end
    fprintf(' %d/%d contrasts found\n', num_found, num_conditions);
end

% Report summary
fprintf('\nContrast file summary:\n');
for cond_idx = 1:num_conditions
    condition_name = conditions{cond_idx};
    field_name = matlab.lang.makeValidName(condition_name);
    fprintf('  %s: %d files\n', condition_name, length(contrast_files.(field_name)));
end

% Verify all conditions have files
field_names = fieldnames(contrast_files);
num_scans_per_condition = cellfun(@(fn) length(contrast_files.(fn)), field_names);

if isempty(num_scans_per_condition) || all(num_scans_per_condition == 0)
    error('No contrast files found. Check your first-level results.');
end

if length(unique(num_scans_per_condition(num_scans_per_condition > 0))) > 1
    warning('Unequal number of scans across conditions.');
    min_subjects = min(num_scans_per_condition(num_scans_per_condition > 0));
    fprintf('Using %d subjects (minimum across conditions)\n', min_subjects);

    for fn_idx = 1:length(field_names)
        fn = field_names{fn_idx};
        if length(contrast_files.(fn)) > min_subjects
            contrast_files.(fn) = contrast_files.(fn)(1:min_subjects);
        end
    end
    num_subjects_found = min_subjects;
else
    num_subjects_found = num_scans_per_condition(1);
end

fprintf('Found %d subjects across all conditions\n', num_subjects_found);

if num_subjects_found < 2
    error('Need at least 2 subjects for second-level analysis');
end

% Build Flexible Factorial Design
fprintf('Building Flexible Factorial Design...\n');

matlabbatch = {};
matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};

% Factor 1: Subject
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).name = 'Subject';
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).dept = 0;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).variance = 1;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(1).ancova = 0;

% Factor 2: Condition
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).name = 'Condition';
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).dept = 1;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).variance = 1;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fac(2).ancova = 0;

% Organize scans and create imatrix
all_scans = {};
imatrix = [];

for cond_idx = 1:num_conditions
    field_name = field_names{cond_idx};
    scans = contrast_files.(field_name);

    if isempty(scans)
        continue;
    end

    if isrow(scans)
        scans = scans';
    end

    all_scans = [all_scans; scans];

    for subj_idx = 1:length(scans)
        imatrix = [imatrix; subj_idx, cond_idx];
    end
end

matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.specall.scans = all_scans;
matlabbatch{1}.spm.stats.factorial_design.des.fblock.fsuball.specall.imatrix = imatrix;

matlabbatch{1}.spm.stats.factorial_design.des.fblock.maininters{1}.inter.fnums = [1;2];

matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

% Model estimation
matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(output_dir, 'SPM.mat')};
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

% Save batch
batch_dir = fullfile(output_dir, 'batch');
if ~exist(batch_dir, 'dir')
    mkdir(batch_dir);
end

save(fullfile(batch_dir, 'second_level_design_batch.mat'), 'matlabbatch');

fprintf('Running model specification and estimation...\n');
spm_jobman('run', matlabbatch);

fprintf('2nd-level model created successfully!\n');
fprintf('Output directory: %s\n', output_dir);

% Create contrasts
fprintf('Setting up contrasts...\n');
create_designI_second_level_contrasts(output_dir, layers, num_pcs, conditions, condition_to_layer);

end

function create_designI_second_level_contrasts(spm_dir, layers, num_pcs, conditions, condition_to_layer)
    % Create F-contrasts and T-contrasts for Design I

    spm_path = fullfile(spm_dir, 'SPM.mat');
    load(spm_path, 'SPM');

    num_conditions = length(conditions);

    matlabbatch = {};
    matlabbatch{1}.spm.stats.con.spmmat = {spm_path};

    ci = 0;

    %% F-contrast for each Layer
    for layer_idx = 1:length(layers)
        layer_name = layers{layer_idx};

        layer_indices = [];
        for cond_idx = 1:num_conditions
            if strcmp(condition_to_layer{cond_idx}, layer_name)
                layer_indices(end+1) = cond_idx;
            end
        end

        num_layer_conditions = length(layer_indices);

        ci = ci + 1;
        F = zeros(num_layer_conditions, num_conditions);
        for i = 1:num_layer_conditions
            F(i, layer_indices(i)) = 1;
        end

        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.name = sprintf('F: %s Layer', layer_name);
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.weights = F;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.sessrep = 'none';

        fprintf('  F-contrast %d: %s Layer (%d conditions)\n', ci, layer_name, num_layer_conditions);
    end

    %% T-contrast for each Layer average
    for layer_idx = 1:length(layers)
        layer_name = layers{layer_idx};

        ci = ci + 1;
        T = zeros(1, num_conditions);
        count = 0;
        for cond_idx = 1:num_conditions
            if strcmp(condition_to_layer{cond_idx}, layer_name)
                T(cond_idx) = 1;
                count = count + 1;
            end
        end
        T = T / count;

        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = sprintf('T: %s average', layer_name);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';
        fprintf('  T-contrast %d: %s average\n', ci, layer_name);
    end

    %% T-contrast: All conditions average
    ci = ci + 1;
    T = ones(1, num_conditions) / num_conditions;
    matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = 'T: All average';
    matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
    matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';
    fprintf('  T-contrast %d: All average\n', ci);

    %% Comparison contrasts between layers
    layer_pairs = {
        {'Early', 'Middle'},
        {'Middle', 'Late'},
        {'Early', 'Late'}
    };

    for p = 1:length(layer_pairs)
        layer1 = layer_pairs{p}{1};
        layer2 = layer_pairs{p}{2};

        indices1 = [];
        indices2 = [];
        for cond_idx = 1:num_conditions
            if strcmp(condition_to_layer{cond_idx}, layer1)
                indices1(end+1) = cond_idx;
            elseif strcmp(condition_to_layer{cond_idx}, layer2)
                indices2(end+1) = cond_idx;
            end
        end

        % layer1 > layer2
        ci = ci + 1;
        T = zeros(1, num_conditions);
        T(indices1) = 1 / length(indices1);
        T(indices2) = -1 / length(indices2);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = sprintf('T: %s > %s', layer1, layer2);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';
        fprintf('  T-contrast %d: %s > %s\n', ci, layer1, layer2);

        % layer2 > layer1
        ci = ci + 1;
        T = zeros(1, num_conditions);
        T(indices1) = -1 / length(indices1);
        T(indices2) = 1 / length(indices2);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = sprintf('T: %s > %s', layer2, layer1);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';
        fprintf('  T-contrast %d: %s > %s\n', ci, layer2, layer1);
    end

    matlabbatch{1}.spm.stats.con.delete = 1;

    % Run contrasts
    spm_jobman('run', matlabbatch);

    fprintf('Contrasts created: %d total\n', ci);
end
