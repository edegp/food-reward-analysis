%% Plot ROI Analysis Results as Bar Chart
% No Statistics Toolbox required
% (Requires SPM on MATLAB path)

%% Define paths (3 levels up from this script: scripts/behavior_glm/roi/)
script_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(script_dir, '..', '..', '..');

%% ROI data from analysis
roi_names = {'vmPFC', 'OFC medial', 'OFC lateral', 'Striatum L', 'Striatum R', 'Insula', 'Amygdala'};
mean_t = [2.262, 1.963, -0.605, 3.074, 2.501, 1.702, 1.237];

%% Create bar plot
figure('Position', [100, 100, 800, 500], 'Color', 'white');

% Create bars with colors based on value
colors = zeros(length(mean_t), 3);
for i = 1:length(mean_t)
    if mean_t(i) > 2
        colors(i,:) = [0.8, 0.2, 0.2];  % Red for high positive
    elseif mean_t(i) > 1
        colors(i,:) = [1, 0.6, 0.4];    % Orange for moderate positive
    elseif mean_t(i) > 0
        colors(i,:) = [1, 0.8, 0.6];    % Light orange for low positive
    else
        colors(i,:) = [0.4, 0.6, 0.8];  % Blue for negative
    end
end

b = bar(mean_t, 'FaceColor', 'flat');
b.CData = colors;

% Add reference lines
hold on;
yline(0, 'k-', 'LineWidth', 1);
yline(1.699, 'r--', 'LineWidth', 1, 'Label', 'p<0.05 (T=1.70, df=29)');  % T threshold for p<0.05, df=29

% Labels
set(gca, 'XTickLabel', roi_names, 'XTickLabelRotation', 45, 'FontSize', 12);
ylabel('Mean T-value', 'FontSize', 14);
xlabel('ROI', 'FontSize', 14);
title('ROI Analysis: ImagexValue Contrast (N=30)', 'FontSize', 16);

% Add value labels on bars
for i = 1:length(mean_t)
    if mean_t(i) >= 0
        text(i, mean_t(i) + 0.15, sprintf('%.2f', mean_t(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    else
        text(i, mean_t(i) - 0.25, sprintf('%.2f', mean_t(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    end
end

% Adjust y-axis
ylim([-1.5, 4]);
grid on;
box off;

% Save figure
output_dir = fullfile(root_dir, 'results', 'roi_analysis');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save as PNG
saveas(gcf, fullfile(output_dir, 'roi_barplot_ImagexValue.png'));
fprintf('Figure saved to: %s\n', fullfile(output_dir, 'roi_barplot_ImagexValue.png'));

% Save as PDF
saveas(gcf, fullfile(output_dir, 'roi_barplot_ImagexValue.pdf'));
fprintf('Figure saved to: %s\n', fullfile(output_dir, 'roi_barplot_ImagexValue.pdf'));

%% Also create a table summary
fprintf('\n=== Summary Table ===\n');
fprintf('ROI\t\t\tMean T\tSignificance\n');
fprintf('-------------------------------------------\n');
t_crit = 1.699;  % T critical for p<0.05, two-tailed, df=29
for i = 1:length(roi_names)
    if abs(mean_t(i)) > t_crit
        sig = '*';
    else
        sig = '';
    end
    fprintf('%-15s\t%.3f\t%s\n', roi_names{i}, mean_t(i), sig);
end
fprintf('\n* p < 0.05 (uncorrected)\n');
fprintf('Note: T critical = %.3f for df=29, two-tailed\n', t_crit);

close(gcf);
