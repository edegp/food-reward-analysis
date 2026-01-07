%-----------------------------------------------------------------------
% Job saved on 03-Oct-2025 11:50:30 by cfg_util (rev $Rev: 8183 $)
% spm SPM - SPM25 (25.01.02)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
function visualize_firstlevel(spm_path, outdir, canonical_img)
% visualize_firstlevel(spm_path)
% visualize_firstlevel(spm_path, outdir)
% Inputs
%   spm_path : full path to SPM.mat for a first-level run
%   outdir   : (optional) directory to save figures (default: [SPM_folder]/figures)
%   canonical_img : path to anatomical image for overlay (with volume index)
% The function will look for con_*.nii and spmT_*.nii in the same folder as SPM.mat
% and create orthogonal slice figures using SPM's orthviews, saving PNG files.

  if nargin < 2 || isempty(outdir)
    outdir = fullfile(fileparts(spm_path), 'figures');
  end

  if isstring(outdir)
    outdir = char(outdir);
  elseif iscell(outdir)
    outdir = outdir{1};
  end

  try
    outdir_abs = char(java.io.File(outdir).getCanonicalPath());
    outdir = outdir_abs;
  catch
    % best effort; keep original outdir
  end

  if ~exist(outdir, 'dir')
    mkdir(outdir);
  end

  if isstring(canonical_img) || (iscell(canonical_img) && isstring(canonical_img{1}))
    canonical_img = char(canonical_img);
  elseif iscell(canonical_img)
    canonical_img = canonical_img{1};
  end

  matlabbatch{1}.spm.stats.results.dir = cellstr(outdir);
  matlabbatch{1}.spm.stats.results.spmmat = cellstr(spm_path);
  matlabbatch{1}.spm.stats.results.conspec.titlestr = '';
  matlabbatch{1}.spm.stats.results.conspec.contrasts = 1;
  matlabbatch{1}.spm.stats.results.conspec.threshdesc = 'FWE';
  matlabbatch{1}.spm.stats.results.conspec.thresh = 0.05;
  matlabbatch{1}.spm.stats.results.conspec.extent = 0;
  matlabbatch{1}.spm.stats.results.conspec.conjunction = 1;
  matlabbatch{1}.spm.stats.results.conspec.mask.none = 1;
  matlabbatch{1}.spm.stats.results.units = 1;
  matlabbatch{1}.spm.stats.results.export{1}.tspm.basename = 'spmT_';
  matlabbatch{1}.spm.stats.results.export{2}.jpg = true;

  spm_jobman('run', matlabbatch);

  try
    xSPM = evalin('base', 'xSPM');
    hReg = evalin('base', 'hReg');
    if isempty(canonical_img)
      warning('canonical_img is empty; skipping sections overlay.');
    else
      spm_sections(xSPM, hReg, canonical_img);
      sections_png = fullfile(outdir, 'sections_overlay.png');
      if ~exist(fileparts(sections_png), 'dir')
        mkdir(fileparts(sections_png));
      end
      print(gcf, sections_png, '-dpng', '-r300');
      close(gcf);
      clc;
    end
  catch ME
    warning('Failed to generate sections overlay: %s', ME.message);
  end
end
