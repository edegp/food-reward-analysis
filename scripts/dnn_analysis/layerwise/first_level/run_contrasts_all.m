function run_designI_contrasts_all(source_label)
%% Run Design I contrasts for all subjects
%
% source_label: 'clip' or 'convnext'

if nargin < 1
    error('Source label required (''clip'' or ''convnext'')');
end

% Setup paths
project_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(project_dir, '..', '..', '..', '..');
addpath(project_dir);

% Initialize SPM
spm('defaults', 'fmri');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);

subjects = arrayfun(@(x) sprintf('%03d', x), 1:31, 'UniformOutput', false);

fprintf('======================================\n');
fprintf('Design I Contrasts: %s\n', upper(source_label));
fprintf('Subjects: %d\n', length(subjects));
fprintf('======================================\n');

for s = 1:length(subjects)
    subj_id = subjects{s};
    fprintf('\n=== Subject %d/%d: %s ===\n', s, length(subjects), subj_id);

    model_name = ['glm_dnn_pmods_designI_', source_label];
    model_dir = fullfile(root_dir, 'results', 'first_level_analysis', ...
                        ['sub-', subj_id], 'glm_model', model_name);

    % Read latest_run.txt
    latest_marker = fullfile(model_dir, 'latest_run.txt');
    if ~exist(latest_marker, 'file')
        d = dir(model_dir);
        d = d([d.isdir] & ~startsWith({d.name}, '.'));
        if isempty(d)
            fprintf('  Not found\n');
            continue;
        end
        [~, idx] = max([d.datenum]);
        latest_dir = fullfile(model_dir, d(idx).name);
    else
        fid = fopen(latest_marker, 'r');
        latest_dir = strtrim(fgetl(fid));
        fclose(fid);
    end

    spm_path = fullfile(latest_dir, 'SPM.mat');
    if ~exist(spm_path, 'file')
        fprintf('  SPM.mat not found\n');
        continue;
    end

    % Check if contrasts already exist
    con1_path = fullfile(latest_dir, 'con_0001.nii');
    if exist(con1_path, 'file')
        fprintf('  Contrasts already exist, skipping\n');
        continue;
    end

    % Build and run contrasts
    try
        matlabbatch = build_designI_contrasts(spm_path);
        spm_jobman('run', matlabbatch);
        fprintf('  Done\n');
    catch ME
        fprintf('  Error: %s\n', ME.message);
    end
end

fprintf('\n======================================\n');
fprintf('Completed Design I contrasts for %s\n', upper(source_label));
fprintf('======================================\n');

end
