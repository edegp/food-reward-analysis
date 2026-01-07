function run_smoothing_all()
%% Rerun smoothing for all subjects
% This script recreates the smoothed directory that was accidentally deleted
clearvars; close all; clc;

% Add helper functions to path
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

% Initialize SPM
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% Paths (2 levels up from this script: scripts/preprocess/)
root_dir = fullfile(script_dir, '..', '..');
derivatives_dir = fullfile(root_dir, 'fMRIprep', 'derivatives');
output_base = fullfile(root_dir, 'fMRIprep', 'smoothed_local');

% Create output directory
if ~exist(output_base, 'dir')
    mkdir(output_base);
end

fprintf('==============================================\n');
fprintf('Smoothing All Subjects\n');
fprintf('==============================================\n');
fprintf('Input: %s\n', derivatives_dir);
fprintf('Output: %s\n', output_base);
fprintf('FWHM: [6 6 6] mm\n\n');

% Find all preprocessed files
fprintf('Finding all preprocessed files...\n');
all_files = {};
for sub_id = 21:28
    sub_str = sprintf('sub-%02d', sub_id);
    sub_dir = fullfile(derivatives_dir, sub_str);

    if ~exist(sub_dir, 'dir')
        fprintf('  Skipping %s (not found)\n', sub_str);
        continue;
    end

    % Find all preproc files for this subject
    pattern = fullfile(sub_dir, '**', '*_desc-preproc_bold.nii.gz');
    files = dir(pattern);

    if isempty(files)
        fprintf('  Skipping %s (no files found)\n', sub_str);
        continue;
    end

    fprintf('  %s: found %d files\n', sub_str, length(files));

    for i = 1:length(files)
        all_files{end+1} = fullfile(files(i).folder, files(i).name);
    end
end

fprintf('\nTotal files to process: %d\n\n', length(all_files));

if isempty(all_files)
    error('No files found to smooth!');
end

% Process in batches of 20 files to avoid memory issues
batch_size = 20;
n_batches = ceil(length(all_files) / batch_size);

for batch_idx = 1:n_batches
    start_idx = (batch_idx - 1) * batch_size + 1;
    end_idx = min(batch_idx * batch_size, length(all_files));
    batch_files = all_files(start_idx:end_idx);

    fprintf('Processing batch %d/%d (files %d-%d)...\n', ...
        batch_idx, n_batches, start_idx, end_idx);

    try
        % Process files one by one to minimize disk usage
        for i = 1:length(batch_files)
            input_file = batch_files{i};

            % Determine output path
            if endsWith(input_file, '.gz')
                nii_file = input_file(1:end-3);
            else
                nii_file = input_file;
            end

            [input_dir, input_name, input_ext] = fileparts(nii_file);

            % Convert: derivatives/sub-01/ses-01/func/file.nii
            % To: smoothed/sub-01/ses-01/func/file.nii
            rel_path = strrep(input_dir, derivatives_dir, '');
            if startsWith(rel_path, filesep)
                rel_path = rel_path(2:end);
            end
            output_dir = fullfile(output_base, rel_path);

            % Create output directory
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end

            output_nii = fullfile(output_dir, [input_name input_ext]);

            % Gunzip to output directory if needed
            if endsWith(input_file, '.gz')
                if ~exist(output_nii, 'file')
                    try
                        gunzip(input_file, output_dir);
                    catch ME
                        warning('Failed to gunzip %s: %s', input_file, ME.message);
                        continue;
                    end
                end
            else
                % Copy .nii file to output directory
                if ~exist(output_nii, 'file')
                    copyfile(input_file, output_nii);
                end
            end

            % Run smoothing on this single file
            try
                smoothing({output_nii}, [6 6 6]);
                fprintf('  Smoothed: %s\n', [input_name input_ext]);
            catch ME
                warning('Failed to smooth %s: %s', output_nii, ME.message);
                continue;
            end

            % Delete the unsmoothed file immediately
            if exist(output_nii, 'file')
                try
                    delete(output_nii);
                catch ME
                    warning('Failed to delete temporary file %s: %s', output_nii, ME.message);
                end
            end
        end

        fprintf('  Batch %d complete\n\n', batch_idx);
    catch ME
        fprintf('ERROR in batch %d: %s\n', batch_idx, ME.message);
        fprintf('%s\n', getReport(ME));
    end
end

fprintf('==============================================\n');
fprintf('Smoothing Complete\n');
fprintf('==============================================\n');
fprintf('Output directory: %s\n', output_base);

% Verify output
fprintf('\nVerifying output...\n');
smoothed_count = 0;
for sub_id = 21:28
    sub_str = sprintf('sub-%02d', sub_id);
    pattern = fullfile(output_base, sub_str, '**', 'ssub-*.nii');
    files = dir(pattern);
    if ~isempty(files)
        fprintf('  %s: %d files\n', sub_str, length(files));
        smoothed_count = smoothed_count + length(files);
    end
end
fprintf('\nTotal smoothed files created: %d\n', smoothed_count);

end
