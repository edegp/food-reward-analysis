function layer_groups = load_layer_groups()
    % Load layer groups configuration from JSON file
    % Returns a struct with layer group definitions for each source

    % Get the directory where this function is located
    func_dir = fileparts(mfilename('fullpath'));
    config_file = fullfile(func_dir, 'layer_groups.json');

    % Check if file exists
    if ~exist(config_file, 'file')
        error('Configuration file not found: %s', config_file);
    end

    % Read JSON file
    fid = fopen(config_file, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);

    % Parse JSON
    layer_groups = jsondecode(str);
end
