%% Plot ROI Analysis Results with Effect Size (Cohen's d) on Y-axis
% Cohen's d = t / sqrt(n) for one-sample t-test

addpath('/Users/yuhiaoki/spm');

%% ROI data from analysis
roi_names = {'vmPFC', 'OFC medial', 'OFC lateral', 'Striatum L', 'Striatum R', 'Insula', 'Amygdala'};
mean_t = [2.262, 1.963, -0.605, 3.074, 2.501, 1.702, 1.237];

% Convert T to Cohen's d: d = t / sqrt(n)
n_subjects = 30;
effect_size = mean_t / sqrt(n_subjects);

%% Create bar plot
figure('Position', [100, 100, 800, 500], 'Color', 'white');

% Create bars with colors based on effect size
colors = zeros(length(effect_size), 3);
for i = 1:length(effect_size)
    if effect_size(i) > 0.5
        colors(i,:) = [0.8, 0.2, 0.2];  % Red for large positive
    elseif effect_size(i) > 0.2
        colors(i,:) = [1, 0.6, 0.4];    % Orange for medium positive
    elseif effect_size(i) > 0
        colors(i,:) = [1, 0.8, 0.6];    % Light orange for small positive
    else
        colors(i,:) = [0.4, 0.6, 0.8];  % Blue for negative
    end
end

b = bar(effect_size, 'FaceColor', 'flat');
b.CData = colors;

% Add reference lines for effect size interpretation
hold on;
yline(0, 'k-', 'LineWidth', 1);
yline(0.2, 'b--', 'LineWidth', 1, 'Label', 'Small (0.2)');
yline(0.5, 'g--', 'LineWidth', 1, 'Label', 'Medium (0.5)');
yline(0.8, 'r--', 'LineWidth', 1, 'Label', 'Large (0.8)');

% Labels
set(gca, 'XTickLabel', roi_names, 'XTickLabelRotation', 45, 'FontSize', 12);
ylabel('Effect Size (Cohen''s d)', 'FontSize', 14);
xlabel('ROI', 'FontSize', 14);
title('ROI Analysis: ImagexValue Contrast (N=30)', 'FontSize', 16);

% Add value labels on bars
for i = 1:length(effect_size)
    if effect_size(i) >= 0
        text(i, effect_size(i) + 0.03, sprintf('%.2f', effect_size(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    else
        text(i, effect_size(i) - 0.05, sprintf('%.2f', effect_size(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    end
end

% Adjust y-axis
ylim([-0.3, 0.9]);
grid on;
box off;

% Add legend for effect size interpretation
text(6.5, 0.85, 'Effect Size Guidelines:', 'FontSize', 10, 'FontWeight', 'bold');
text(6.5, 0.78, '0.2 = Small', 'FontSize', 9, 'Color', 'b');
text(6.5, 0.71, '0.5 = Medium', 'FontSize', 9, 'Color', [0 0.5 0]);
text(6.5, 0.64, '0.8 = Large', 'FontSize', 9, 'Color', 'r');

% Save figure
output_dir = '/Users/yuhiaoki/dev/hit/food-brain/results/roi_analysis';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save as PNG
saveas(gcf, fullfile(output_dir, 'roi_effectsize_ImagexValue.png'));
fprintf('Figure saved to: %s\n', fullfile(output_dir, 'roi_effectsize_ImagexValue.png'));

% Save as PDF
saveas(gcf, fullfile(output_dir, 'roi_effectsize_ImagexValue.pdf'));
fprintf('Figure saved to: %s\n', fullfile(output_dir, 'roi_effectsize_ImagexValue.pdf'));

%% Summary table
fprintf('\n=== Effect Size Summary ===\n');
fprintf('ROI\t\t\tT-value\tCohen''s d\tInterpretation\n');
fprintf('--------------------------------------------------------\n');
for i = 1:length(roi_names)
    if abs(effect_size(i)) >= 0.8
        interp = 'Large';
    elseif abs(effect_size(i)) >= 0.5
        interp = 'Medium';
    elseif abs(effect_size(i)) >= 0.2
        interp = 'Small';
    else
        interp = 'Negligible';
    end
    fprintf('%-15s\t%.3f\t%.3f\t\t%s\n', roi_names{i}, mean_t(i), effect_size(i), interp);
end

fprintf('\nNote: Cohen''s d = t / sqrt(n), n = %d\n', n_subjects);

close(gcf);
