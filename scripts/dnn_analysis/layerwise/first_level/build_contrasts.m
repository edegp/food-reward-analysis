function matlabbatch = build_designI_contrasts(spm_path)
% Build contrasts for Design I (Low-Correlation 3-Layer Analysis)
%
% Structure:
%   ImageOnset_S01-S12: Session-specific with 9 pmods (3 layers × 3 PCs)
%   Layers: Early, Middle, Late
%   PCs: pc1, pc2, pc3
%
% Contrasts created:
%   - T-contrasts for each Layer × PC (summed across sessions)
%   - T-contrasts for each Layer (all PCs)
%   - F-contrasts for each Layer

if ~exist(spm_path,'file'); error('SPM.mat not found: %s', spm_path); end
load(spm_path, 'SPM');

names = SPM.xX.name;
if isempty(names)
    error('SPM design appears to have no regressors.');
end

num_regressors = numel(names);

matlabbatch = {};
matlabbatch{1}.spm.stats.con.spmmat = {spm_path};

ci = 0;

% Layer names
layers = {'Early', 'Middle', 'Late'};
num_pcs = 3;

fprintf('Building contrasts for Design I...\n');

%% T-contrasts for each Layer × PC combination (summed across sessions)
for layer_idx = 1:length(layers)
    layer_name = layers{layer_idx};

    for pc = 1:num_pcs
        pmod_name = sprintf('%s_pc%d', layer_name, pc);
        pattern = sprintf('ImageOnset_S\\d+x%s\\^1', pmod_name);

        pc_cols = [];
        for col_idx = 1:numel(names)
            if ~isempty(regexp(names{col_idx}, pattern, 'once'))
                pc_cols = [pc_cols, col_idx];
            end
        end

        if ~isempty(pc_cols)
            ci = ci + 1;
            T = zeros(1, num_regressors);
            T(pc_cols) = 1;

            contrast_name = sprintf('%s PC%d', layer_name, pc);
            matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = contrast_name;
            matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
            matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';

            fprintf('  T-contrast: %s (%d regressors)\n', contrast_name, length(pc_cols));
        end
    end
end

%% T-contrasts for each Layer (all PCs combined)
for layer_idx = 1:length(layers)
    layer_name = layers{layer_idx};

    layer_cols = [];
    for pc = 1:num_pcs
        pmod_name = sprintf('%s_pc%d', layer_name, pc);
        pattern = sprintf('ImageOnset_S\\d+x%s\\^1', pmod_name);

        for col_idx = 1:numel(names)
            if ~isempty(regexp(names{col_idx}, pattern, 'once'))
                layer_cols = [layer_cols, col_idx];
            end
        end
    end

    if ~isempty(layer_cols)
        ci = ci + 1;
        T = zeros(1, num_regressors);
        T(layer_cols) = 1;

        contrast_name = sprintf('%s (all PCs)', layer_name);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = contrast_name;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';

        fprintf('  T-contrast: %s (%d regressors)\n', contrast_name, length(layer_cols));
    end
end

%% F-contrasts for each Layer (testing if any PC has effect)
for layer_idx = 1:length(layers)
    layer_name = layers{layer_idx};

    F_rows = [];
    for pc = 1:num_pcs
        pmod_name = sprintf('%s_pc%d', layer_name, pc);
        pattern = sprintf('ImageOnset_S\\d+x%s\\^1', pmod_name);

        pc_cols = [];
        for col_idx = 1:numel(names)
            if ~isempty(regexp(names{col_idx}, pattern, 'once'))
                pc_cols = [pc_cols, col_idx];
            end
        end

        if ~isempty(pc_cols)
            F_row = zeros(1, num_regressors);
            F_row(pc_cols) = 1;
            F_rows = [F_rows; F_row];
        end
    end

    if ~isempty(F_rows)
        ci = ci + 1;
        contrast_name = sprintf('F: %s', layer_name);
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.name = contrast_name;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.weights = F_rows;
        matlabbatch{1}.spm.stats.con.consess{ci}.fcon.sessrep = 'none';

        fprintf('  F-contrast: %s (%d PCs)\n', contrast_name, size(F_rows, 1));
    end
end

%% Comparison contrasts between layers
layer_pairs = {
    {'Early', 'Middle'},
    {'Middle', 'Late'},
    {'Early', 'Late'}
};

for p = 1:length(layer_pairs)
    layer1 = layer_pairs{p}{1};
    layer2 = layer_pairs{p}{2};

    % Get columns for layer1
    cols1 = [];
    for pc = 1:num_pcs
        pmod_name = sprintf('%s_pc%d', layer1, pc);
        pattern = sprintf('ImageOnset_S\\d+x%s\\^1', pmod_name);
        for col_idx = 1:numel(names)
            if ~isempty(regexp(names{col_idx}, pattern, 'once'))
                cols1 = [cols1, col_idx];
            end
        end
    end

    % Get columns for layer2
    cols2 = [];
    for pc = 1:num_pcs
        pmod_name = sprintf('%s_pc%d', layer2, pc);
        pattern = sprintf('ImageOnset_S\\d+x%s\\^1', pmod_name);
        for col_idx = 1:numel(names)
            if ~isempty(regexp(names{col_idx}, pattern, 'once'))
                cols2 = [cols2, col_idx];
            end
        end
    end

    if ~isempty(cols1) && ~isempty(cols2)
        % layer1 > layer2
        ci = ci + 1;
        T = zeros(1, num_regressors);
        T(cols1) = 1;
        T(cols2) = -1;

        contrast_name = sprintf('%s > %s', layer1, layer2);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = contrast_name;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';

        fprintf('  T-contrast: %s\n', contrast_name);

        % layer2 > layer1
        ci = ci + 1;
        T = zeros(1, num_regressors);
        T(cols1) = -1;
        T(cols2) = 1;

        contrast_name = sprintf('%s > %s', layer2, layer1);
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.name = contrast_name;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.weights = T;
        matlabbatch{1}.spm.stats.con.consess{ci}.tcon.sessrep = 'none';

        fprintf('  T-contrast: %s\n', contrast_name);
    end
end

if ci == 0
    error('No valid contrasts could be constructed.');
end

matlabbatch{1}.spm.stats.con.delete = 1;

fprintf('Created %d contrasts for Design I\n', ci);

end
