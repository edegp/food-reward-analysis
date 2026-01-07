function run000_Extract_ImageInfo
%%
clear all; close all; clc;
num_imgs = 896; % 画像枚数

data_image_info = [];
for n = 1:num_imgs

    % 画像ファイルの名前取得
    if n < 10
        img_ID = ['000',num2str(n)];
    elseif n < 100
        img_ID = ['00',num2str(n)];
    else
        img_ID = ['0',num2str(n)];
    end

    % 画像ファイルの読み込み
    I = imread(['../../Database/',img_ID,'.jpg']);

    % RGB値
    R = mean(mean(double(I(:,:,1))));
    G = mean(mean(double(I(:,:,2))));
    B = mean(mean(double(I(:,:,3))));

    % 輝度の計算（知覚的な明度 L*（CIELAB））
    % Use a local rgb2lab implementation to avoid requiring the
    % Image Processing Toolbox / license at runtime.
    Lab = rgb2lab_custom(I); % returns L*, a*, b* image (L* scale 0-100)
    L = mean(mean(Lab(:,:,1))); % 0–100（double）

    % データの保存
    data_image_info = [data_image_info; [n, R, G, B, L]];

end

% CSVファイル保存
data_T_save = table(data_image_info); % テーブル形式に変換
data_T_save = splitvars(data_T_save, 'data_image_info', 'NewVariableNames', {'imgID','R','G','B','L'}); % 変数名を追加
fname_save = '../../data_images/data_image_info.csv'; disp(fname_save)
% Ensure output directory exists
outdir = fileparts(fname_save);
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
writetable(data_T_save, fname_save);

end

% -------------------------------------------------------------------------
function Lab = rgb2lab_custom(I)
% RGB2LAB_CUSTOM Convert an uint8 RGB image (sRGB) to CIELAB without
% requiring Image Processing Toolbox. Input I can be uint8 [0..255] or
% double in [0..1]. Output Lab is double with L* in [0,100].

    % Convert to double in [0,1]
    if ~isfloat(I)
        I = double(I) / 255;
    else
        % if values are >1, assume 0-255
        if max(I(:)) > 1
            I = I / 255;
        end
    end

    % Linearize sRGB (inverse gamma)
    thresh = 0.04045;
    I_lin = zeros(size(I));
    mask = I <= thresh;
    I_lin(mask) = I(mask) / 12.92;
    I_lin(~mask) = ((I(~mask) + 0.055) / 1.055) .^ 2.4;

    % sRGB to XYZ (D65)
    M = [0.4124564, 0.3575761, 0.1804375;
         0.2126729, 0.7151522, 0.0721750;
         0.0193339, 0.1191920, 0.9503041];
    % reshape for matrix multiply
    [h, w, ~] = size(I_lin);
    rgb = reshape(I_lin, [], 3)'; % 3 x N
    xyz = M * rgb; % 3 x N
    X = reshape(xyz(1, :), h, w);
    Y = reshape(xyz(2, :), h, w);
    Z = reshape(xyz(3, :), h, w);

    % Reference white (D65) scaled to Y=1
    Xn = 0.95047; Yn = 1.00000; Zn = 1.08883;

    % f(t) function
    delta = 6/29;
    eps = delta^3;
    fx = f_xyz(X / Xn, eps, delta);
    fy = f_xyz(Y / Yn, eps, delta);
    fz = f_xyz(Z / Zn, eps, delta);

    L = 116 * fy - 16;
    a = 500 * (fx - fy);
    b = 200 * (fy - fz);

    Lab = cat(3, L, a, b);
end

function ft = f_xyz(t, eps, delta)
    ft = t .^ (1/3);
    small = t <= eps;
    if any(small(:))
        ft(small) = (t(small) * (841/108)) + (4/29);
    end
end
