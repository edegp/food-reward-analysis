function run_smoothing_subject(subject_nums)
%% Run smoothing for specified subjects
% Usage:
%   run_smoothing_subject(31)        % Single subject
%   run_smoothing_subject([29 31])   % Multiple subjects
%   run_smoothing_subject(1:31)      % Range of subjects
%
% Task command:
%   task smoothing_subject -- 31

if nargin < 1 || isempty(subject_nums)
    error('Subject number(s) required. Usage: run_smoothing_subject(31) or run_smoothing_subject([29 31])');
end

% Add smoothing function to path
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

% Initialize SPM
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% Configuration
session_nums = 1:3;
runs_per_session = 4;
fwhm = [6 6 6];

% External drive paths
derivatives_dir = '/Volumes/Extreme Pro/hit/food-brain/fMRIprep/derivatives';
output_base = '/Volumes/Extreme Pro/hit/food-brain/fMRIprep/smoothed';

fprintf('==============================================\n');
fprintf('Smoothing for subjects: %s\n', mat2str(subject_nums));
fprintf('==============================================\n');
fprintf('Input: %s\n', derivatives_dir);
fprintf('Output: %s\n', output_base);
fprintf('FWHM: [6 6 6] mm\n\n');

for sub_num = subject_nums
    sub_str = sprintf('sub-%02d', sub_num);
    sub_deriv_dir = fullfile(derivatives_dir, sub_str);

    fprintf('\nProcessing %s...\n', sub_str);

    if ~exist(sub_deriv_dir, 'dir')
        fprintf('  ERROR: Directory not found: %s\n', sub_deriv_dir);
        continue;
    end

    for ses = session_nums
        ses_str = sprintf('ses-%02d', ses);
        func_dir = fullfile(sub_deriv_dir, ses_str, 'func');

        if ~exist(func_dir, 'dir')
            fprintf('  Session %s: No func directory\n', ses_str);
            continue;
        end

        for run = 1:runs_per_session
            % Input file
            input_pattern = sprintf('%s_%s_task-pt_run-%02d_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', sub_str, ses_str, run);
            input_file = fullfile(func_dir, input_pattern);

            if ~exist(input_file, 'file')
                fprintf('  %s run-%02d: File not found\n', ses_str, run);
                continue;
            end

            % Output directory
            output_dir = fullfile(output_base, sub_str, ses_str, 'func');
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end

            % Unzip
            [~, base_name, ~] = fileparts(input_file(1:end-3));  % Remove .gz
            output_nii = fullfile(output_dir, [base_name '.nii']);

            % Delete existing smoothed file if re-running
            smoothed_file = fullfile(output_dir, ['s' base_name '.nii']);
            if exist(smoothed_file, 'file')
                fprintf('  Deleting existing smoothed file: %s\n', smoothed_file);
                delete(smoothed_file);
            end

            % Gunzip
            if ~exist(output_nii, 'file')
                fprintf('  Gunzipping: %s_%s_run-%02d\n', sub_str, ses_str, run);
                try
                    gunzip(input_file, output_dir);
                catch ME
                    warning('Failed to gunzip: %s', ME.message);
                    continue;
                end
            end

            % Smooth
            fprintf('  Smoothing: %s_%s_run-%02d\n', sub_str, ses_str, run);
            try
                smoothing({output_nii}, fwhm);
            catch ME
                warning('Failed to smooth: %s', ME.message);
                continue;
            end

            % Delete unsmoothed file
            if exist(output_nii, 'file')
                delete(output_nii);
            end
        end
    end

    % Verify output
    pattern = fullfile(output_base, sub_str, '**', 'ssub-*.nii');
    files = dir(pattern);
    fprintf('  %s: %d smoothed files created\n', sub_str, length(files));
end

fprintf('\n==============================================\n');
fprintf('Smoothing Complete\n');
fprintf('==============================================\n');

end
