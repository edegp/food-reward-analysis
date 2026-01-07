%% Plot ROI Analysis Results - Clean Design (Significant ROIs only)
% Effect Size (Cohen's d) on Y-axis
% (Requires SPM on MATLAB path)

%% Define paths (3 levels up from this script: scripts/behavior_glm/roi/)
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');

%% ROI data - only significant ROIs (T > 1.70, p < 0.05)
roi_names = {'vmPFC', 'mOFC', 'L-Striatum', 'R-Striatum', 'Insula'};
mean_t = [2.262, 1.963, 3.074, 2.501, 1.702];

% Convert T to Cohen's d
n_subjects = 30;
effect_size = mean_t / sqrt(n_subjects);

%% Create clean bar plot
fig = figure('Position', [100, 100, 600, 450], 'Color', 'white');
ax = axes('Parent', fig);

% Orange color for all bars
colors = repmat([0.95, 0.55, 0.25], length(effect_size), 1);

b = bar(ax, effect_size, 'FaceColor', 'flat', 'EdgeColor', 'none');
b.CData = colors;

% Zero line
hold on;
yline(0, 'k-', 'LineWidth', 0.5);

% Labels with dark black color
set(ax, 'XTickLabel', roi_names, 'XTickLabelRotation', 30, ...
    'FontSize', 12, 'FontWeight', 'normal', ...
    'XColor', 'k', 'YColor', 'k');
ylabel('効果量', 'FontSize', 14, 'Color', 'k', 'FontWeight', 'bold');
title('Food Value × Image (N=30)', 'FontSize', 15, 'FontWeight', 'bold', 'Color', 'k');

% Add value labels on bars - black text
for i = 1:length(effect_size)
    text(i, effect_size(i) + 0.025, sprintf('%.2f', effect_size(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'k', 'FontWeight', 'bold');
end

% Clean white background
set(ax, 'Color', 'white');
set(fig, 'Color', 'white');
ylim([0, 0.7]);
set(ax, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1);

% Save figure
output_dir = fullfile(root_dir, 'results', 'roi_analysis');

print(fig, fullfile(output_dir, 'roi_effectsize_clean.png'), '-dpng', '-r300');
fprintf('Figure saved to: %s\n', fullfile(output_dir, 'roi_effectsize_clean.png'));

print(fig, fullfile(output_dir, 'roi_effectsize_clean.pdf'), '-dpdf');
fprintf('Figure saved to: %s\n', fullfile(output_dir, 'roi_effectsize_clean.pdf'));

close(fig);
