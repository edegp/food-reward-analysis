function build_contrasts_daywise(spm_path, source_label, global_cols, shared_cols, layer_groups, specific_cols)
%% Build contrasts for Day-wise Hierarchical DNN GLM
% Creates contrasts that average PMODs across days
%
% Each PC has 3 day-level estimates (Day1, Day2, Day3)
% Contrast: average across days with weight 1/3

fprintf('\n=== Building Contrasts (Day-wise Design) ===\n');

load(spm_path, 'SPM');
reg_names = SPM.xX.name;
num_regs = length(reg_names);

fprintf('Total regressors: %d\n', num_regs);

% Helper function
    function cols = find_reg_cols(pattern)
        cols = [];
        for i = 1:num_regs
            if ~isempty(regexp(reg_names{i}, pattern, 'once'))
                cols(end+1) = i;
            end
        end
    end

matlabbatch = {};
matlabbatch{1}.spm.stats.con.spmmat = {spm_path};
con_idx = 0;

weight = 1/3;  % Average across 3 days

%% Global PC contrasts
fprintf('\nCreating Global PC contrasts...\n');
num_global = length(global_cols);
for pc = 1:num_global
    con_vec = zeros(1, num_regs);

    for day = 1:3
        pattern = sprintf('Sn\\(1\\) ImageOnset_Day%dxGlobal_pc%d\\^1', day, pc);
        cols = find_reg_cols(pattern);
        if ~isempty(cols)
            con_vec(cols) = weight;
        end
    end

    if any(con_vec ~= 0)
        con_idx = con_idx + 1;
        matlabbatch{1}.spm.stats.con.consess{con_idx}.tcon.name = sprintf('Global_pc%d_pos', pc);
        matlabbatch{1}.spm.stats.con.consess{con_idx}.tcon.weights = con_vec;
        matlabbatch{1}.spm.stats.con.consess{con_idx}.tcon.sessrep = 'none';
    end
end
fprintf('  Global contrasts: %d\n', con_idx);

%% Shared PC contrasts
fprintf('Creating Shared PC contrasts...\n');
shared_start = con_idx;
for pc = 1:length(shared_cols)
    col_name = shared_cols{pc};
    % Extract type (e.g., Shared_Initial_Middle_pc1 -> Initial_Middle, 1)
    parts = regexp(col_name, 'Shared_(.+)_pc(\d+)', 'tokens');
    if ~isempty(parts)
        shared_type = parts{1}{1};
        pc_num = str2double(parts{1}{2});

        con_vec = zeros(1, num_regs);
        pmod_name = strrep(col_name, '-', '_');

        for day = 1:3
            pattern = sprintf('Sn\\(1\\) ImageOnset_Day%dx%s\\^1', day, regexprep(pmod_name, '([()])', '\\$1'));
            cols = find_reg_cols(pattern);
            if ~isempty(cols)
                con_vec(cols) = weight;
            end
        end

        if any(con_vec ~= 0)
            con_idx = con_idx + 1;
            matlabbatch{1}.spm.stats.con.consess{con_idx}.tcon.name = sprintf('Shared_%s_pc%d_pos', strrep(shared_type, '-', '_'), pc_num);
            matlabbatch{1}.spm.stats.con.consess{con_idx}.tcon.weights = con_vec;
            matlabbatch{1}.spm.stats.con.consess{con_idx}.tcon.sessrep = 'none';
        end
    end
end
fprintf('  Shared contrasts: %d\n', con_idx - shared_start);

%% Layer-specific PC contrasts
fprintf('Creating Layer-specific PC contrasts...\n');
layer_start = con_idx;
for g = 1:length(layer_groups)
    group_name = layer_groups{g};
    cols = specific_cols.(group_name);

    for pc = 1:length(cols)
        con_vec = zeros(1, num_regs);

        for day = 1:3
            pattern = sprintf('Sn\\(1\\) ImageOnset_Day%dx%s_pc%d\\^1', day, group_name, pc);
            cols_found = find_reg_cols(pattern);
            if ~isempty(cols_found)
                con_vec(cols_found) = weight;
            end
        end

        if any(con_vec ~= 0)
            con_idx = con_idx + 1;
            % Capitalize first letter
            group_cap = [upper(group_name(1)), group_name(2:end)];
            matlabbatch{1}.spm.stats.con.consess{con_idx}.tcon.name = sprintf('%s_pc%d_pos', group_cap, pc);
            matlabbatch{1}.spm.stats.con.consess{con_idx}.tcon.weights = con_vec;
            matlabbatch{1}.spm.stats.con.consess{con_idx}.tcon.sessrep = 'none';
        end
    end
end
fprintf('  Layer-specific contrasts: %d\n', con_idx - layer_start);

fprintf('\nTotal contrasts: %d\n', con_idx);

%% Run contrast estimation
if con_idx > 0
    matlabbatch{1}.spm.stats.con.delete = 1;
    spm_jobman('run', matlabbatch);
    fprintf('Contrasts estimated successfully\n');
else
    warning('No contrasts created!');
end

%% Verify
load(spm_path, 'SPM');
fprintf('\nContrast verification (first 10):\n');
for i = 1:min(10, length(SPM.xCon))
    fprintf('  %d: %s\n', i, SPM.xCon(i).name);
end
if length(SPM.xCon) > 10
    fprintf('  ... and %d more\n', length(SPM.xCon) - 10);
end

fprintf('\n=== Contrasts Complete ===\n');

end
