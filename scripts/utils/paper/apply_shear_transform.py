#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
#   "pillow",
# ]
# ///
"""
シアー変換を適用して3D効果を作成するスクリプト
"""

import numpy as np
from PIL import Image
import sys

def apply_shear_transform(image_path, output_path, shear_x=0.3, shear_y=0.1):
    """
    画像にシアー変換を適用して3D効果を作成

    Parameters:
    -----------
    image_path : str
        入力画像のパス
    output_path : str
        出力画像のパス
    shear_x : float
        X方向のシアー係数（デフォルト: 0.3）
    shear_y : float
        Y方向のシアー係数（デフォルト: 0.1）
    """
    # 画像を読み込む
    img = Image.open(image_path)

    # 透明度を保持するためにRGBAモードに変換
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    width, height = img.size

    # シアー変換行列
    # [1, shear_x, 0]
    # [shear_y, 1, 0]
    shear_matrix = (1, shear_x, -shear_x * width / 2,
                    shear_y, 1, -shear_y * height / 2)

    # 変換後のサイズを計算
    new_width = int(width + abs(shear_x) * height)
    new_height = int(height + abs(shear_y) * width)

    # シアー変換を適用
    img_sheared = img.transform(
        (new_width, new_height),
        Image.AFFINE,
        shear_matrix,
        resample=Image.BICUBIC
    )

    # 背景を透明にする
    img_sheared.save(output_path)
    print(f"シアー変換適用完了: {output_path}")
    print(f"元のサイズ: {width}x{height}")
    print(f"変換後のサイズ: {new_width}x{new_height}")
    print(f"シアー係数: X={shear_x}, Y={shear_y}")


def create_3d_effect(image_path, output_path, angle=15):
    """
    遠近法を使った3D効果を作成

    Parameters:
    -----------
    image_path : str
        入力画像のパス
    output_path : str
        出力画像のパス
    angle : float
        傾き角度（度）
    """
    # 画像を読み込む
    img = Image.open(image_path)

    # 透明度を保持
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    width, height = img.size

    # 遠近法変換のための係数を計算
    # Y軸周りの回転をシミュレート
    theta = np.radians(angle)

    # 遠近法変換行列（簡易版）
    # 右側が奥に見えるように変換
    scale_factor = np.cos(theta)
    shear_factor = np.sin(theta) * 0.5

    # 変換行列
    transform_matrix = (
        scale_factor, shear_factor, 0,
        0, 1, 0
    )

    # 新しいサイズを計算
    new_width = int(width * 1.5)
    new_height = int(height * 1.2)

    # 変換を適用
    img_3d = img.transform(
        (new_width, new_height),
        Image.AFFINE,
        transform_matrix,
        resample=Image.BICUBIC
    )

    img_3d.save(output_path)
    print(f"3D効果適用完了: {output_path}")


def create_isometric_view(image_path, output_path):
    """
    アイソメトリック（等角投影）ビューを作成

    Parameters:
    -----------
    image_path : str
        入力画像のパス
    output_path : str
        出力画像のパス
    """
    img = Image.open(image_path)

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    width, height = img.size

    # アイソメトリック変換
    # 30度の角度で変換
    angle = 30
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))

    # 変換行列（アイソメトリック投影）
    transform_matrix = (
        cos_angle, -sin_angle * 0.5, 0,
        0, 1, -height * 0.2
    )

    new_width = int(width * 1.5)
    new_height = int(height * 1.5)

    img_iso = img.transform(
        (new_width, new_height),
        Image.AFFINE,
        transform_matrix,
        resample=Image.BICUBIC
    )

    img_iso.save(output_path)
    print(f"アイソメトリックビュー作成完了: {output_path}")


if __name__ == "__main__":
    input_image = "/Users/yuhiaoki/dev/hit/food-brain/Database/0002.jpg"

    # 1. 基本的なシアー変換（逆方向）
    apply_shear_transform(
        input_image,
        "/Users/yuhiaoki/dev/hit/food-brain/Database/0002_shear.png",
        shear_x=-0.3,
        shear_y=-0.1
    )

    # 2. 3D効果（遠近法）
    create_3d_effect(
        input_image,
        "/Users/yuhiaoki/dev/hit/food-brain/Database/0002_3d.png",
        angle=20
    )

    # 3. アイソメトリックビュー
    create_isometric_view(
        input_image,
        "/Users/yuhiaoki/dev/hit/food-brain/Database/0002_isometric.png"
    )

    print("\n全ての変換が完了しました！")
    print("生成されたファイル:")
    print("  - 0002_shear.png (基本シアー変換)")
    print("  - 0002_3d.png (3D遠近法効果)")
    print("  - 0002_isometric.png (アイソメトリックビュー)")
