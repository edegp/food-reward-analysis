#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
#   "pillow",
# ]
# ///
"""
CNNアーキテクチャ図のような遠近法3D効果を作成
"""

import numpy as np
from PIL import Image

def create_cnn_style_perspective(image_path, output_path, depth=0.6):
    """
    CNNアーキテクチャ図のような遠近法3D効果を作成

    Parameters:
    -----------
    image_path : str
        入力画像のパス
    output_path : str
        出力画像のパス
    depth : float
        奥行きの比率（0.3-0.8、小さいほど強い3D効果）
    """
    # 画像を読み込む
    img = Image.open(image_path)

    # RGBAモードに変換（透明度を保持）
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    width, height = img.size

    # 十分に大きなキャンバスを作成（画像が確実に収まるサイズ）
    # 傾きが大きいので、上部に十分な余裕を持たせる
    canvas_width = int(width * 2)
    canvas_height = int(height * 2.5)  # 高さを増やす

    # 変換後の4点を定義（キャンバスの中央に配置）
    margin_left = int(width * 0.3)
    margin_top = int(height * 0.8)  # 上部のマージンをさらに増やす（はみ出し防止）

    # 元の画像の4隅
    original_points = np.array([
        [0, 0],           # 左上
        [width, 0],       # 右上
        [width, height],  # 右下
        [0, height]       # 左下
    ], dtype=np.float32)

    # 変換後の4点
    # CNNアーキテクチャ図のように：
    # 1. 左側は固定
    # 2. 上辺と下辺を同じ角度で上に傾ける
    # 3. 右側を縮小して遠近法効果（奥行き）を出す

    right_width = int(width * depth)

    # 上方向への傾き角度（右側がどれだけ上に上がるか）
    upward_tilt = int(height * 0.45)  # 右側を上に45%持ち上げる

    transformed_points = np.array([
        [margin_left, margin_top],                              # 左上（固定）
        [margin_left + right_width, margin_top - upward_tilt],  # 右上（上に持ち上げ + 奥）
        [margin_left + right_width, margin_top + height - upward_tilt], # 右下（上に持ち上げ + 奥）
        [margin_left, margin_top + height]                      # 左下（固定）
    ], dtype=np.float32)

    # 変換行列を計算
    def find_coeffs(source_coords, target_coords):
        """遠近法変換の係数を計算"""
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(source_coords).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    coeffs = find_coeffs(original_points, transformed_points)

    # 遠近法変換を適用
    img_perspective = img.transform(
        (canvas_width, canvas_height),
        Image.PERSPECTIVE,
        coeffs,
        Image.BICUBIC
    )

    # 枠線を追加
    from PIL import ImageDraw

    final_img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    final_img.paste(img_perspective, (0, 0), img_perspective)

    # 黒い枠線を描画
    draw = ImageDraw.Draw(final_img)
    border_width = 3  # 枠線の太さ

    # 変換後の4点に沿って枠線を描く
    points = [
        tuple(transformed_points[0]),  # 左上
        tuple(transformed_points[1]),  # 右上
        tuple(transformed_points[2]),  # 右下
        tuple(transformed_points[3]),  # 左下
        tuple(transformed_points[0])   # 左上に戻る（閉じる）
    ]

    draw.line(points, fill=(0, 0, 0, 255), width=border_width)

    # 不要な透明部分をトリミング
    bbox = final_img.getbbox()
    if bbox:
        final_img = final_img.crop(bbox)

    final_img.save(output_path)
    print(f"遠近法3D効果を作成しました: {output_path}")
    print(f"元のサイズ: {width}x{height}")
    print(f"変換後のサイズ: {final_img.size[0]}x{final_img.size[1]}")
    print(f"奥行き比率: {depth}")

    # 検証：画像を再度読み込んで確認
    verify_img = Image.open(output_path)
    print(f"保存後のサイズ確認: {verify_img.size[0]}x{verify_img.size[1]}")


def create_simple_perspective(image_path, output_path, depth=0.6):
    """
    シンプルな遠近法効果（影なし）

    Parameters:
    -----------
    image_path : str
        入力画像のパス
    output_path : str
        出力画像のパス
    depth : float
        奥行きの深さ（0.3-0.8）
    """
    img = Image.open(image_path)

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    width, height = img.size

    canvas_width = int(width * 2)
    canvas_height = int(height * 2.5)  # 高さを増やす

    margin_left = int(width * 0.3)
    margin_top = int(height * 0.8)  # 上部のマージンをさらに増やす（はみ出し防止）

    original_points = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    right_width = int(width * depth)

    # 上方向への傾き角度（右側がどれだけ上に上がるか）
    upward_tilt = int(height * 0.45)

    transformed_points = np.array([
        [margin_left, margin_top],
        [margin_left + right_width, margin_top - upward_tilt],
        [margin_left + right_width, margin_top + height - upward_tilt],
        [margin_left, margin_top + height]
    ], dtype=np.float32)

    def find_coeffs(source_coords, target_coords):
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(source_coords).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    coeffs = find_coeffs(original_points, transformed_points)

    img_perspective = img.transform(
        (canvas_width, canvas_height),
        Image.PERSPECTIVE,
        coeffs,
        Image.BICUBIC
    )

    # 不要な透明部分をトリミング
    bbox = img_perspective.getbbox()
    if bbox:
        img_perspective = img_perspective.crop(bbox)

    img_perspective.save(output_path)
    print(f"シンプルな遠近法3D効果を作成しました: {output_path}")
    print(f"保存後のサイズ: {img_perspective.size[0]}x{img_perspective.size[1]}")


if __name__ == "__main__":
    input_image = "/Users/yuhiaoki/dev/hit/food-brain/Database/0002.jpg"

    # CNN図のような遠近法効果（枠線付き）
    create_cnn_style_perspective(
        input_image,
        "/Users/yuhiaoki/dev/hit/food-brain/Database/0002_cnn_style.png",
        depth=1.0  # 奥行きなし（右側も元のサイズのまま）
    )

    # シンプルな遠近法効果（影なし）
    create_simple_perspective(
        input_image,
        "/Users/yuhiaoki/dev/hit/food-brain/Database/0002_perspective.png",
        depth=1.0  # 奥行きなし
    )

    print("\n全ての変換が完了しました！")
    print("生成されたファイル:")
    print("  - 0002_cnn_style.png (CNN図スタイル - 影付き)")
    print("  - 0002_perspective.png (シンプルな遠近法)")
