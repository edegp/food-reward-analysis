function run_designI_all_subjects(source_label, varargin)
%% Run Design I (Low-Correlation 3-Layer Analysis) for all subjects
%
% Design I uses 3 layers with low inter-layer correlation (|r| < 0.5):
%   - CLIP: stage1_1 (Early), stage2_22 (Middle), stage3_2 (Late)
%   - ConvNeXt: features_1_0 (Early), features_5_22 (Middle), features_7_0 (Late)
%
% Structure:
%   - Session-specific ImageOnset (S01-S12)
%   - 9 pmods per session (3 layers × 3 PCs)
%
% Usage:
%   run_designI_all_subjects('clip')
%   run_designI_all_subjects('convnext')
%   run_designI_all_subjects('clip', 'subjects', {'001', '002', '003'})
%   run_designI_all_subjects('clip', 'start_sub', 14, 'end_sub', 31)
%   run_designI_all_subjects('clip', 'start_sub', 14)  % 14 to 31
%
% source_label: 'clip' or 'convnext'
% Optional:
%   'subjects', {cell array} - specify exact subject list
%   'start_sub', N - start subject number (1-31)
%   'end_sub', N - end subject number (1-31, default: 31)
% Default: 31 subjects (001-031)

if nargin < 1
    error('Usage: run_designI_all_subjects(''clip'') or run_designI_all_subjects(''convnext'')');
end

% Parse optional arguments
p = inputParser;
addRequired(p, 'source_label', @ischar);
addParameter(p, 'subjects', [], @iscell);
addParameter(p, 'start_sub', 1, @isnumeric);
addParameter(p, 'end_sub', 31, @isnumeric);
parse(p, source_label, varargin{:});

if ~isempty(p.Results.subjects)
    % Use explicit subject list
    subjects = p.Results.subjects;
else
    % Use start_sub to end_sub range
    start_n = p.Results.start_sub;
    end_n = p.Results.end_sub;
    subjects = arrayfun(@(x) sprintf('%03d', x), start_n:end_n, 'UniformOutput', false);
end

% Define layers for display
if strcmp(source_label, 'clip')
    layer_info = 'stage1_1, stage2_22, stage3_2';
elseif strcmp(source_label, 'convnext')
    layer_info = 'features_1_0, features_5_22, features_7_0';
else
    layer_info = 'unknown';
end

fprintf('======================================\n');
fprintf('Design I: Low-Correlation 3-Layer Analysis\n');
fprintf('======================================\n');
fprintf('Model: %s\n', upper(source_label));
fprintf('Layers: %s\n', layer_info);
fprintf('PCs: 1-3 per layer (9 pmods/session)\n');
fprintf('Subjects: %d\n', length(subjects));
fprintf('Start time: %s\n\n', datetime('now'));

success_count = 0;
fail_count = 0;
failed_subjects = {};

for i = 1:length(subjects)
    sub_id = subjects{i};
    fprintf('\n=== Subject %d/%d: %s ===\n', i, length(subjects), sub_id);

    try
        run_model_dnn_glm_designI(sub_id, source_label);
        fprintf('Subject %s completed successfully\n', sub_id);
        success_count = success_count + 1;
    catch ME
        fprintf('ERROR for subject %s: %s\n', sub_id, ME.message);
        fail_count = fail_count + 1;
        failed_subjects{end+1} = sub_id;
    end
end

fprintf('\n======================================\n');
fprintf('Design I completed!\n');
fprintf('======================================\n');
fprintf('Successful: %d/%d\n', success_count, length(subjects));
fprintf('Failed: %d\n', fail_count);
if ~isempty(failed_subjects)
    fprintf('Failed subjects: %s\n', strjoin(failed_subjects, ', '));
end
fprintf('End time: %s\n', datetime('now'));

end
