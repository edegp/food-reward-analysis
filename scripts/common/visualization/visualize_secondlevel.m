function visualize_secondlevel(model_name, con_title, dir_name)
  % Change to the directory that holds the SPM.mat you want to inspect.
  target_dir = dir_name;

  spm('Defaults','fMRI');
  spm_jobman('initcfg');
  matlabbatch{1}.spm.stats.results.spmmat = {fullfile(target_dir,'SPM.mat')};
  matlabbatch{1}.spm.stats.results.conspec.titlestr = '';
  matlabbatch{1}.spm.stats.results.conspec.contrasts = 1; % use the only contrast in that file
  matlabbatch{1}.spm.stats.results.conspec.threshdesc = 'FWE'; % or 'none' for uncorrected
  matlabbatch{1}.spm.stats.results.conspec.thresh = 0.05;
  matlabbatch{1}.spm.stats.results.conspec.extent = 20;
  matlabbatch{1}.spm.stats.results.conspec.conjunction = 1;
  matlabbatch{1}.spm.stats.results.units = 1; % slices in mm
  matlabbatch{1}.spm.stats.results.print = false; %
  matlabbatch{1}.spm.stats.results.export{2}.jpg = true;

  % set true to auto-save PS/PDF
  spm_jobman('run', matlabbatch);
  clear matlabbatch
end
