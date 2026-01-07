%% ROI Analysis for glm_rgb_nutri - ImagexValue contrast
% Extract mean T values from predefined ROIs and generate barplot
% (Requires SPM on MATLAB path)

%% Define paths (3 levels up from this script: scripts/behavior_glm/roi/)
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');
% Note: Update the timestamp directory as needed
spm_file = fullfile(root_dir, 'results', 'second_level_analysis', 'glm_rgb_nutri', 'ImagexValue_20251221_145043', 'SPM.mat');
roi_dir = fullfile(root_dir, 'rois', 'AAL2');
output_dir = fullfile(root_dir, 'results', 'roi_analysis');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Define ROIs to analyze
roi_names = {
    'vmPFC_mask.nii'
    'OFC_medial_bi_mask.nii'
    'Striatum_L_mask.nii'
    'Striatum_R_mask.nii'
    'Insula_bi_mask.nii'
    'Amygdala_bi_mask.nii'
};

% Japanese labels
roi_labels_jp = {'腹内側前頭前皮質', '内側眼窩前頭皮質', '線条体（左）', '線条体（右）', '島皮質', '扁桃体'};

%% Load spmT image
spmT_file = fullfile(fileparts(spm_file), 'spmT_0001.nii');
V_t = spm_vol(spmT_file);
Y_t = spm_read_vols(V_t);

fprintf('\n=== ROI Analysis: glm_rgb_nutri - ImagexValue ===\n\n');

mean_t_values = zeros(length(roi_names), 1);
n_voxels_list = zeros(length(roi_names), 1);

for i = 1:length(roi_names)
    roi_file = fullfile(roi_dir, roi_names{i});

    if ~exist(roi_file, 'file')
        fprintf('%-25s %s\n', roi_labels_jp{i}, 'FILE NOT FOUND');
        continue;
    end

    % Load ROI mask
    V_roi = spm_vol(roi_file);
    Y_roi = spm_read_vols(V_roi);

    % Resample ROI to spmT space if needed
    if ~isequal(V_t.dim, V_roi.dim)
        Y_roi_resampled = zeros(V_t.dim);
        for z = 1:V_t.dim(3)
            for y = 1:V_t.dim(2)
                for x = 1:V_t.dim(1)
                    mni = V_t.mat * [x; y; z; 1];
                    roi_vox = V_roi.mat \ mni;
                    rx = round(roi_vox(1));
                    ry = round(roi_vox(2));
                    rz = round(roi_vox(3));
                    if rx >= 1 && rx <= V_roi.dim(1) && ...
                       ry >= 1 && ry <= V_roi.dim(2) && ...
                       rz >= 1 && rz <= V_roi.dim(3)
                        Y_roi_resampled(x,y,z) = Y_roi(rx, ry, rz);
                    end
                end
            end
        end
        Y_roi = Y_roi_resampled;
    end

    % Get T values within ROI
    roi_mask = Y_roi > 0;
    t_values = Y_t(roi_mask & ~isnan(Y_t));

    if isempty(t_values)
        continue;
    end

    mean_t = mean(t_values);
    n_voxels = length(t_values);

    mean_t_values(i) = mean_t;
    n_voxels_list(i) = n_voxels;

    fprintf('%-20s T=%.3f (n=%d voxels)\n', roi_labels_jp{i}, mean_t, n_voxels);
end

%% Calculate p-values from T values (one-sample t-test, df = n-1)
n_subjects = 31;
df = n_subjects - 1;

% Custom tcdf using incomplete beta function (no Statistics Toolbox needed)
% tcdf(t, v) = 1 - 0.5*betainc(v/(v+t^2), v/2, 0.5) for t > 0
tcdf_custom = @(t, v) 0.5 * (1 + sign(t) .* (1 - betainc(v./(v + t.^2), v/2, 0.5)));

% One-tailed p-value for positive T, two-tailed for negative T
p_values = zeros(length(mean_t_values), 1);
for i = 1:length(mean_t_values)
    if mean_t_values(i) > 0
        p_values(i) = 1 - tcdf_custom(mean_t_values(i), df);  % one-tailed
    else
        p_values(i) = 2 * tcdf_custom(mean_t_values(i), df);  % two-tailed for negative
    end
end

% Print p-values
fprintf('\n--- P-values ---\n');
for i = 1:length(roi_labels_jp)
    fprintf('%-20s T=%.3f, p=%.4f\n', roi_labels_jp{i}, mean_t_values(i), p_values(i));
end

%% Create bar plot - White background, Japanese labels, no title
fig = figure('Position', [100, 100, 1000, 600], 'Color', 'white');
set(fig, 'PaperPositionMode', 'auto');
set(fig, 'InvertHardcopy', 'off');  % Preserve colors when printing
ax = axes('Parent', fig);
set(ax, 'Color', 'white');  % White axes background

% Create bars with colors based on significance
colors = zeros(length(mean_t_values), 3);
for i = 1:length(mean_t_values)
    if mean_t_values(i) > 0 && p_values(i) < 0.05
        colors(i,:) = [0.8, 0.2, 0.2];  % Red for significant positive
    elseif mean_t_values(i) > 0
        colors(i,:) = [0.95, 0.5, 0.3];    % Orange for non-significant positive
    else
        colors(i,:) = [0.3, 0.5, 0.8];  % Blue for negative
    end
end

% Calculate effect size (d = T / sqrt(n))
effect_size = mean_t_values / sqrt(n_subjects);

b = bar(effect_size, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.5);
b.CData = colors;

% Add reference lines
hold on;
yline(0, 'k-', 'LineWidth', 1.5);

% Japanese labels - larger font
set(gca, 'XTickLabel', roi_labels_jp, 'XTickLabelRotation', 45, 'FontSize', 14, 'FontName', 'Hiragino Sans');
ylabel('効果量', 'FontSize', 18, 'FontName', 'Hiragino Sans');
xlabel('関心領域 (ROI)', 'FontSize', 18, 'FontName', 'Hiragino Sans');

% Add significance asterisks on bars
for i = 1:length(effect_size)
    if effect_size(i) >= 0
        y_pos = effect_size(i) + 0.03;
    else
        y_pos = effect_size(i) - 0.05;
    end

    % Determine significance level
    if p_values(i) < 0.001
        sig_str = '***';
    elseif p_values(i) < 0.01
        sig_str = '**';
    elseif p_values(i) < 0.05
        sig_str = '*';
    else
        sig_str = '';
    end

    if ~isempty(sig_str)
        text(i, y_pos, sig_str, 'HorizontalAlignment', 'center', 'FontSize', 18, 'FontWeight', 'bold');
    end
end

% Adjust y-axis
y_max = max(effect_size) + 0.15;
y_min = min(effect_size) - 0.15;
ylim([min(-0.25, y_min), max(0.6, y_max)]);

% Clean style
set(gca, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.2);
set(gca, 'XColor', 'k', 'YColor', 'k');

% Add legend for significance
text(5.3, 0.55, '有意水準:', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Hiragino Sans');
text(5.3, 0.50, '* p < 0.05', 'FontSize', 11, 'FontName', 'Hiragino Sans');
text(5.3, 0.45, '** p < 0.01', 'FontSize', 11, 'FontName', 'Hiragino Sans');
text(5.3, 0.40, '*** p < 0.001', 'FontSize', 11, 'FontName', 'Hiragino Sans');

% Save figure
print(fig, fullfile(output_dir, 'roi_effectsize_rgb_nutri_ImagexValue.png'), '-dpng', '-r300');
fprintf('\nFigure saved: %s\n', fullfile(output_dir, 'roi_effectsize_rgb_nutri_ImagexValue.png'));

saveas(fig, fullfile(output_dir, 'roi_effectsize_rgb_nutri_ImagexValue.pdf'));

% Save results
results.roi_labels_jp = roi_labels_jp;
results.mean_t = mean_t_values;
results.p_values = p_values;
results.n_voxels = n_voxels_list;
results.n_subjects = n_subjects;
results.df = df;
save(fullfile(output_dir, 'roi_analysis_rgb_nutri.mat'), 'results');

close(fig);
fprintf('Done!\n');
