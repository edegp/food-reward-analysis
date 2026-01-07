function group_mask_file = create_group_mask()
% Create group brain mask from all subjects' fMRIprep brain masks

spm('Defaults', 'fMRI');

script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');

% Create group mask directory
mask_dir = fullfile(root_dir, 'rois', 'group_mask');
if ~exist(mask_dir, 'dir'), mkdir(mask_dir); end

% Collect all subject masks (sub-01 format in derivatives)
subs_list = 1:31;
mask_files = {};
for i = 1:length(subs_list)
    sub = sprintf('%02d', subs_list(i));
    deriv_dir = fullfile(root_dir, 'fMRIprep', 'derivatives', ['sub-', sub], 'anat');
    mask_nii = fullfile(deriv_dir, ['sub-', sub, '_space-MNI152NLin2009cAsym_desc-brain_mask.nii']);
    mask_gz = fullfile(deriv_dir, ['sub-', sub, '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz']);
    if exist(mask_nii, 'file')
        mask_files{end+1} = mask_nii;
    elseif exist(mask_gz, 'file')
        fprintf('Gunzipping: %s\n', mask_gz);
        gunzip(mask_gz, deriv_dir);
        mask_files{end+1} = mask_nii;
    end
end
fprintf('Found %d brain masks\n', length(mask_files));

% Create group mask (intersection)
if ~isempty(mask_files)
    V = spm_vol(mask_files{1});
    Y = spm_read_vols(V);
    group_mask = Y > 0;
    for i = 2:length(mask_files)
        Vi = spm_vol(mask_files{i});
        Yi = spm_read_vols(Vi);
        group_mask = group_mask & (Yi > 0);
    end

    % Save group mask
    group_mask_file = fullfile(mask_dir, 'group_brain_mask.nii');
    Vout = V;
    Vout.fname = group_mask_file;
    Vout.dt = [spm_type('uint8') 0];
    spm_write_vol(Vout, uint8(group_mask));
    fprintf('Group mask saved: %s\n', group_mask_file);
    fprintf('Mask voxels: %d\n', sum(group_mask(:)));
else
    error('No brain masks found');
end

end
