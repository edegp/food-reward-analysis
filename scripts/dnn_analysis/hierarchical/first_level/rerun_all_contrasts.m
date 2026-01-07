%% Re-run contrasts for all subjects (both CLIP and ConvNeXt)
% This script re-runs the contrast estimation for the hierarchical DNN GLM
% to add individual Shared F-contrasts

clear; clc;

script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..', '..');
addpath(script_dir);

spm('defaults', 'fmri');
spm_jobman('initcfg');
spm_get_defaults('cmdline', true);

%% Subject list
subs_list = {'002', '003', '005', '006', '007', '008', '009', '010', ...
             '011', '013', '014', '015', '016', '018', '019', '020', ...
             '022', '023', '024', '025'};

sources = {'clip', 'convnext'};

%% Re-run contrasts for all subjects
for src_idx = 1:length(sources)
    source = sources{src_idx};
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('Processing %s\n', upper(source));
    fprintf('%s\n', repmat('=', 1, 60));

    for sub_idx = 1:length(subs_list)
        sub_id = subs_list{sub_idx};
        fprintf('\n--- Subject %s ---\n', sub_id);

        try
            rerun_hierarchical_contrasts(sub_id, source);
        catch ME
            fprintf('Error for sub-%s: %s\n', sub_id, ME.message);
        end
    end
end

fprintf('\n\nAll contrasts re-computed!\n');
fprintf('Next: Run second-level analysis for new contrasts\n');
