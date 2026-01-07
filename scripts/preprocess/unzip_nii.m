% unzip_nii(files, [target_dir])
% Unzip any .nii.gz entries in files into a target directory (or a temp dir if not provided).
% Returns:
% - out_files: cell array of paths pointing to uncompressed .nii files (or original paths for non-gz)
% - tmp_dir: path to directory containing uncompressed files (empty if none created)
function [out_files, tmp_dir] = unzip_nii(files, target_dir)
if nargin < 2
  target_dir = '';
end
if ischar(files) || isstring(files)
  files = {char(files)};
end
files = reshape(files, [], 1);
tmp_dir = '';
out_files = files;
for i = 1:numel(files)
  f = files{i};
  if endsWith(f, '.gz')
    if isempty(target_dir)
      if isempty(tmp_dir)
        tmp_dir = fullfile(tempdir, ['unzip_nii_' char(java.util.UUID.randomUUID)]);
        mkdir(tmp_dir);
      end
      use_dir = tmp_dir;
    else
      use_dir = target_dir;
      if isempty(tmp_dir)
        tmp_dir = target_dir; % report back the dir we used
      end
    end
    % prefer existing sibling .nii if present
    plain = f(1:end-3);
    if exist(plain, 'file') == 2
      out_files{i} = plain;
    else
      try
        outs = gunzip(f, use_dir);
        out_files{i} = outs{1};
      catch ME
        warning('unzip_nii:fail', 'Failed to gunzip %s: %s', f, ME.message);
        out_files{i} = f; % fallback
      end
    end
  else
    out_files{i} = f;
  end
end
end
