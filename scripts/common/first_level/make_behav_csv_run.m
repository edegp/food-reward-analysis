function data_T_save = make_behav_csv_run(subs, rate_filepath, time_filepath)
%% Create a per-run behavior CSV from raw rate and timing CSVs
% subs: subject id string like '001'
% rate_filepath: full path to rating CSV (e.g. '../../Food_Behavior/001/rating_data_x.csv')
% time_filepath: full path to timing CSV (e.g. '../../Food_Behavior/001/GLM_all_x.csv')

    script_dir = fileparts(mfilename('fullpath'));
    root_dir = fullfile(script_dir, '..', '..', '..');

    phase_list = {'image','question','rating'};
    data_T_images = readtable(fullfile(root_dir, 'data_images', 'data_image_info.csv'), 'VariableNamingRule', 'preserve');
    %%%%% 時間の情報
    data_T_tmp = readtable(time_filepath, 'VariableNamingRule', 'preserve');
    data_T_save = table();
    for k = 1:length(phase_list)
        idx = find(strcmp(data_T_tmp{:, 'type'}, phase_list{k}));
        data_T_save.(phase_list{k}) = data_T_tmp{idx, 'onset'};
    end

    %%%%% 評定の情報
    data_T_tmp = readtable(rate_filepath, 'VariableNamingRule', 'preserve');
    idx_omit = (data_T_tmp{:, 'Rating'} == 8); % 課題に関係ないボタン押し
    data_T_tmp(idx_omit, :) = [];
    data_T_save.('ImageName') = data_T_tmp{:, 'Image Name'};
    data_T_save.('RatingValue') = data_T_tmp{:, 'Rating'};

    %%%%% 画像の情報 (Luminance)
    image_L = [];
    for k = 1:height(data_T_tmp)
        idx_image = data_T_tmp{k, 'Image Name'};
        image_L = [image_L; data_T_images{idx_image, 'L'}];
    end
    data_T_save.('image_L') = image_L;

    %%%%% 画像の情報 (RGB & 栄養価) from food_value.csv
    data_T_food = readtable(fullfile(root_dir, 'data_images', 'food_value.csv'), 'VariableNamingRule', 'preserve');

    % RGB values
    image_R = [];
    image_G = [];
    image_B = [];
    % Nutrition values
    image_protein = [];
    image_fat = [];
    image_carbs = [];
    image_kcal = [];

    for k = 1:height(data_T_tmp)
        idx_image = data_T_tmp{k, 'Image Name'};
        image_R = [image_R; data_T_food{idx_image, 'red'}];
        image_G = [image_G; data_T_food{idx_image, 'green'}];
        image_B = [image_B; data_T_food{idx_image, 'blue'}];
        image_protein = [image_protein; data_T_food{idx_image, 'protein_100g'}];
        image_fat = [image_fat; data_T_food{idx_image, 'fat_100g'}];
        image_carbs = [image_carbs; data_T_food{idx_image, 'carbs_100g'}];
        image_kcal = [image_kcal; data_T_food{idx_image, 'kcal_100g'}];
    end
    data_T_save.('image_R') = image_R;
    data_T_save.('image_G') = image_G;
    data_T_save.('image_B') = image_B;
    data_T_save.('image_protein') = image_protein;
    data_T_save.('image_fat') = image_fat;
    data_T_save.('image_carbs') = image_carbs;
    data_T_save.('image_kcal') = image_kcal;

end
