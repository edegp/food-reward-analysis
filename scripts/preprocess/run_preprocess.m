% Consolidated preprocessing launcher
% Builds a single list of files across subjects/sessions/runs and runs everything in a temporary workspace

% Configuration - adjust as needed
subject_nums = 1:31;   % subjects to process (can be non-contiguous vector)
session_nums = 1:3;   % sessions per subject
runs_per_session = 4; % number of runs per session
fwhm = [6 6 6];       % smoothing kernel (mm)

% Create a temporary workspace for all intermediate and output files
tmp_root = tempdir;
tmp_work_dir = fullfile(tmp_root, ['preproc_' char(java.util.UUID.randomUUID)]);
mkdir(tmp_work_dir);
fprintf('Using temporary work dir: %s\n', tmp_work_dir);

all_files = {};
for i = subject_nums
  for j = session_nums
    for k = 1:runs_per_session
      fp = sprintf('/Volumes/Transcend/hit/food-brain/fMRIprep/derivatives/sub-%02d/ses-%02d/func/sub-%02d_ses-%02d_task-pt_run-%02d_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', i, j, i, j, k);
      all_files{end+1} = fp; %#ok<AGROW>
    end
  end
end

% Quick existence check so the user sees which expected files are missing
missing = {};
for idx = 1:numel(all_files)
  if ~exist(all_files{idx}, 'file')
    missing{end+1} = all_files{idx}; %#ok<AGROW>
  end
end
if ~isempty(missing)
  fprintf('Warning: %d expected files were not found. Listing missing files:\n', numel(missing));
  for m = 1:numel(missing)
    fprintf('  %s\n', missing{m});
  end
else
  fprintf('All %d expected files exist.\n', numel(all_files));
end

% Unzip .nii.gz files into the temporary workspace; unzip_nii returns the (possibly unzipped) paths and tmp dir
[to_process, tmp_unzip_dir] = unzip_nii(all_files, tmp_work_dir);

% Run smoothing once for the full list. smoothing will accept nested or flat cell arrays.
% Since we've unzipped into tmp_work_dir the smoothed outputs (prefixed with 's') will also be written there.
processed = smoothing(to_process, fwhm);

% Copy smoothed outputs (files prefixed with 's') back to the directories
% where the original .nii.gz files live, so results are available alongside inputs.
fprintf('Copying smoothed outputs back to original .nii.gz directories...\n');
for idx = 1:numel(to_process)
  src_path = to_process{idx};
  orig_gz = all_files{idx};
  % skip entries that were missing or not processed
  if ~exist(src_path, 'file')
    continue;
  end
  src_dir = fileparts(src_path);
  [~, base_name, ~] = fileparts(src_path);
  % First, look for smoothed files that start with 's' + base_name in the same dir
  pattern = fullfile(src_dir, ['s' base_name '*']);
  s_list = dir(pattern);
  % Put smoothed files in a parallel folder structure under fMRIprep/smoothed
  % Use strrep for compatibility with char paths across MATLAB versions
  dest_dir = strrep(fileparts(orig_gz), 'derivatives', 'smoothed');
  % If none found locally, search the temporary unzip directory recursively for any
  % smoothed files that contain the base name. This covers cases where smoothing
  % wrote outputs to different subfolders or used a slightly different name.
  if isempty(s_list) && exist('tmp_unzip_dir', 'var') && isfolder(tmp_unzip_dir)
    rec_pattern = fullfile(tmp_unzip_dir, '**', ['*' base_name '*']);
    s_list = dir(rec_pattern);
    % keep only files that look like smoothed outputs (start with 's' or contain '_s')
    s_list = s_list(~[s_list.isdir]);
    keep = false(1, numel(s_list));
    for si = 1:numel(s_list)
      nm = s_list(si).name;
      if startsWith(nm, ['s' base_name]) || contains(nm, ['s' base_name]) || contains(nm, '_s')
        keep(si) = true;
      end
    end
    s_list = s_list(keep);
  end
  if isempty(s_list)
    fprintf('  No smoothed file found for %s (looked in %s and %s)\n', src_path, src_dir, tmp_unzip_dir);
    continue;
  end
  if ~exist(dest_dir, 'dir')
    warning('run_preprocess:destMissing', 'Destination dir %s not found, creating it.', dest_dir);
    mkdir(dest_dir);
  end
  for k = 1:numel(s_list)
    % Use the folder reported by dir() because when we searched recursively
    % the file may not live in the original src_dir
    src_full = fullfile(s_list(k).folder, s_list(k).name);
    dest_full = fullfile(dest_dir, s_list(k).name);
    try
      copyfile(src_full, dest_full);
      fprintf('  Copied %s -> %s\n', src_full, dest_full);
      % remove the copied file from temp workspace
      try
        delete(src_full);
      catch DELME
        warning('run_preprocess:removeTempFail', 'Could not remove temp file %s: %s', src_full, DELME.message);
      end
    catch ME
      warning('run_preprocess:copyFail', 'Failed to copy %s -> %s: %s', src_full, dest_full, ME.message);
    end
  end
end
