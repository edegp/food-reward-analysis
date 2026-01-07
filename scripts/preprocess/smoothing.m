%-----------------------------------------------------------------------
% Consolidated smoothing helper
% Accepts:
% - files: cell array of file paths (flat) or nested cell arrays (subjects -> sessions -> runs)
% - fwhm: optional [x y z] vector specifying Gaussian kernel in mm (default [6 6 6])
% - output_dir: optional output directory for unzipped files (default: tempdir/smoothing_unzip)
% Returns:
% - processed: cell array of files passed to SPM
%-----------------------------------------------------------------------
function processed = smoothing(files, fwhm, output_dir)
  if nargin < 2 || isempty(fwhm)
    fwhm = [6 6 6];
  end
  if nargin < 3 || isempty(output_dir)
    output_dir = '';
  end

  % Expect files to be a flat cell array of paths (or a single string)
  if ischar(files) || isstring(files)
    files = {char(files)};
  elseif iscell(files)
    % assume a flat cell array; ensure column
    files = reshape(files, [], 1);
  else
    error('smoothing:badInput', '`files` must be a cell array of paths or a string.');
  end

  % Validate entries: ensure strings and check existence
  valid_flags = false(size(files));
  missing = {};
  for i = 1:numel(files)
    entry = files{i};
    if ~(ischar(entry) || isstring(entry))
      warning('smoothing:unexpectedType', 'Ignoring unsupported entry at index %d of type %s', i, class(entry));
      continue
    end
    entry = char(entry);
    files{i} = entry;
    if exist(entry, 'file') == 2
      valid_flags(i) = true;
    else
      missing{end+1} = entry; %#ok<AGROW>
    end
  end

  valid = files(valid_flags);

  if ~isempty(missing)
    fprintf('smoothing: %d files missing, they will be skipped.\n', numel(missing));
    nshow = min(10, numel(missing));
    for ii = 1:nshow
      fprintf('  missing: %s\n', missing{ii});
    end
    if numel(missing) > nshow
      fprintf('  ... (and %d more)\n', numel(missing) - nshow);
    end
  end

  if isempty(valid)
    error('smoothing:NoValidFiles', 'No valid files found to smooth.');
  end

  % Prepare matlabbatch for a single SPM call
  % If any files are compressed (.nii.gz), gunzip them into a temp folder
  if ~isempty(output_dir)
    tmp_unzip_dir = fullfile(output_dir, 'smoothing_temp');
  else
    tmp_unzip_dir = fullfile(tempdir, 'smoothing_unzip');
  end
  if ~exist(tmp_unzip_dir, 'dir')
    mkdir(tmp_unzip_dir);
  end

  unzipped = cell(size(valid));
  for i = 1:numel(valid)
    f = valid{i};
    if endsWith(f, '.gz')
      % prefer an already-uncompressed sibling if present
      maybe = f(1:end-3);
      if exist(maybe, 'file') == 2
        unzipped{i} = maybe;
      else
        try
          outs = gunzip(f, tmp_unzip_dir);
          unzipped{i} = outs{1};
        catch ME
          warning('smoothing:gunzipFail', 'Failed to gunzip %s: %s', f, ME.message);
          unzipped{i} = f; % fall back to original
        end
      end
    else
      unzipped{i} = f;
    end
  end

  matlabbatch = {};
  % Ensure SPM gets a proper cellstr column vector
  matlabbatch{1}.spm.spatial.smooth.data = cellstr(reshape(unzipped, [], 1));
  matlabbatch{1}.spm.spatial.smooth.fwhm = fwhm;
  matlabbatch{1}.spm.spatial.smooth.dtype = 0;
  matlabbatch{1}.spm.spatial.smooth.im = 0;
  matlabbatch{1}.spm.spatial.smooth.prefix = 's';

  % Run SPM job
  fprintf('Running SPM smoothing on %d files with FWHM [%s].\n', numel(valid), num2str(fwhm));

  % Sanity: print first few files and existence checks to help SPM debugging
  nshow = min(10, numel(unzipped));
  for ii = 1:nshow
    fprintf('  [%d] %s  exist=%d\n', ii, matlabbatch{1}.spm.spatial.smooth.data{ii}, exist(matlabbatch{1}.spm.spatial.smooth.data{ii}, 'file'));
  end
  % Additional SPM-level verification: try spm_vol on a few entries
  if exist('spm_vol', 'file') == 2
    for ii = 1:nshow
      fname = matlabbatch{1}.spm.spatial.smooth.data{ii};
      try
        V = spm_vol(fname);
        fprintf('    spm_vol ok: %s  dims=%s\n', fname, mat2str(size(V(1))));
      catch ME
        fprintf('    spm_vol FAIL for %s: %s\n', fname, ME.message);
      end
    end
  else
    fprintf('    spm functions not found on path; cannot run spm_vol checks.\n');
  end

  % Print a compact dump of matlabbatch data type info
  fprintf('matlabbatch data type: %s, numel=%d\n', class(matlabbatch{1}.spm.spatial.smooth.data), numel(matlabbatch{1}.spm.spatial.smooth.data));

  spm_jobman('run', matlabbatch);

  % cleanup temporary unzipped files created in tmp_unzip_dir
  try
    for i = 1:numel(unzipped)
      uf = unzipped{i};
      if startsWith(uf, tmp_unzip_dir) && exist(uf, 'file') == 2
        delete(uf);
      end
    end
    % remove temp dir if empty
    d = dir(tmp_unzip_dir);
    if numel(d) <= 2
      rmdir(tmp_unzip_dir);
      fprintf('smoothing: Removed temporary unzipped files in %s\n', tmp_unzip_dir);
    end
  catch ME
    warning('smoothing:cleanupFail', 'Failed to clean up temp files: %s', ME.message);
  end

  processed = valid;
end
