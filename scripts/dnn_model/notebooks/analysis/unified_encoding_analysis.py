#!/usr/bin/env python3
"""
統合エンコーディング分析スクリプト
CNNモデルとCLIPモデルの中間層特徴量を使用して、
主観的価値、栄養価、色を予測します。
"""

import os

# ============================================================================
# 一時ファイルをProjectCにリダイレクト（ルートパーティション圧迫防止）
# ============================================================================
TEMP_DIR = "/mnt/ProjectC/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
os.environ["TORCH_HOME"] = "/mnt/ProjectC/.cache/torch"
os.environ["HF_HOME"] = "/mnt/ProjectC/.cache/huggingface"

import gc
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from collections import OrderedDict

# OpenCLIP imports
from open_clip import create_model_and_transforms

# Local imports
from src.const import DATA_PATH, ROOT_PATH

plt.rcParams["font.serif"] = ["noto"]
sns.set_theme()

# ============================================================================
# Configuration
# ============================================================================
# CPUを強制使用（GPUはトレーニング用に確保）
device = "cpu"
print(f"使用デバイス: {device}")

# クロスバリデーション設定
N_SPLITS = 8
RANDOM_STATE = 42

# CLIP設定 - ConvNeXt Baseを使用（224x224入力でCNNと統一）
CLIP_MODEL_NAME = "convnext_base"  # 同じConvNeXtアーキテクチャで事前学習データの比較
CLIP_PRETRAINED = "laion400m_s13b_b51k"

# Ridge回帰のハイパーパラメータ
RIDGE_ALPHA = 1.0

# 保存先ディレクトリ
OUTPUT_DIR = os.path.join(DATA_PATH, "output", "results", "unified_encoding_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Dataset Class
# ============================================================================
class ImageDataset(Dataset):
    """画像データセット"""

    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f"{i:04d}.jpg" for i in range(1, 897)]
        self.image_files = [f for f in self.image_files if os.path.exists(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, idx


# ============================================================================
# Feature Extraction Functions
# ============================================================================
def extract_features_by_activation(model, dataset, device, save_dir, model_type='convnext'):
    """
    活性化関数（GELU/ReLU）にフックを登録して、層ごとに特徴量を抽出してファイルに保存
    メモリ効率を改善するため、層ごとに段階的に処理してファイルに保存後メモリを解放

    Args:
        model: PyTorchモデル
        dataset: 画像データセット
        device: デバイス
        save_dir: 保存先ディレクトリ
        model_type: モデル名（ファイル名に使用）

    Returns:
        layer_files: 各層の特徴量ファイルパスのリスト
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # まず登録するフック数を数える
    hook_count = 0
    def count_hooks(module: nn.Module):
        nonlocal hook_count
        for name, child in module.named_children():
            if name == "transformer":
                continue
            if isinstance(child, (nn.GELU, nn.ReLU)) and hasattr(child, "register_forward_hook"):
                hook_count += 1
            count_hooks(child)

    count_hooks(model)
    print(f"{model_type}: 検出した活性化層数: {hook_count}")

    layer_files = []

    # 層ごとに処理（メモリ効率のため）
    for target_layer_idx in range(hook_count):
        current_layer_idx = [0]  # クロージャ用
        layer_features = []

        def hook_fn(module, inp, out):
            # 現在の対象層のみ保存
            if current_layer_idx[0] == target_layer_idx:
                feat = out.detach().cpu()
                if len(feat.shape) > 2:
                    feat = feat.reshape(feat.size(0), -1)
                layer_features.append(feat)
            current_layer_idx[0] += 1

        # フックを登録
        handles = []
        def register_hooks(module: nn.Module):
            for name, child in module.named_children():
                if name == "transformer":
                    continue
                if isinstance(child, (nn.GELU, nn.ReLU)) and hasattr(child, "register_forward_hook"):
                    handle = child.register_forward_hook(hook_fn)
                    handles.append(handle)
                register_hooks(child)

        register_hooks(model)

        # 全画像を処理
        with torch.no_grad():
            for image, idx in tqdm(dataset, desc=f"{model_type} Layer {target_layer_idx+1}/{hook_count}", leave=False):
                current_layer_idx[0] = 0  # リセット
                image = image.unsqueeze(0).to(device)
                _ = model(image)

        # フックを解除
        for handle in handles:
            handle.remove()

        # 特徴量を結合してファイルに保存
        if layer_features:
            batch_feats = torch.cat(layer_features, dim=0)
            save_path = os.path.join(save_dir, f"{model_type}_layer_{target_layer_idx}.pt")
            torch.save(batch_feats, save_path)
            layer_files.append(save_path)
            print(f"Layer {target_layer_idx}: shape {batch_feats.shape} -> {os.path.basename(save_path)}")

            # メモリ解放（ルートパーティション圧迫防止）
            del batch_feats
            layer_features.clear()
            torch.cuda.empty_cache()
            gc.collect()

    return layer_files


def evaluate_features_from_files(layer_files, target_data, valid_indices):
    """
    保存された特徴量ファイルを読み込んでCV評価を実行

    Args:
        layer_files: 各層の特徴量ファイルパスのリスト
        target_data: ターゲット変数の辞書
        valid_indices: 有効なインデックス（欠損値除外後）

    Returns:
        all_results: 各層の評価結果
    """
    all_results = OrderedDict()

    for layer_idx, file_path in enumerate(layer_files):
        print(f"\n評価中: {os.path.basename(file_path)}")

        # 特徴量を読み込み
        features = torch.load(file_path)
        print(f"  特徴量形状: {features.shape}")

        # フィルタリングして評価
        features_filtered = features[valid_indices].numpy()
        layer_results = evaluate_layer_with_cv(features_filtered, target_data, f"layer_{layer_idx}")
        all_results[f"layer_{layer_idx}"] = layer_results
        print(f"  主観的価値: {layer_results['res_L']:.3f}")

        # メモリ解放
        del features, features_filtered

    return all_results


def extract_clip_features_by_activation(model, dataset, device, save_dir):
    """
    CLIP用：活性化関数にフックを登録して、層ごとに特徴量を抽出してファイルに保存
    メモリ効率を改善するため、層ごとに段階的に処理

    Args:
        model: CLIPモデル
        dataset: 画像データセット
        device: デバイス
        save_dir: 保存先ディレクトリ

    Returns:
        layer_files: 各層の特徴量ファイルパスのリスト
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # まず登録するフック数を数える（named_modulesを使用、クラス名で判定）
    gelu_modules = []
    for name, module in model.visual.named_modules():
        if "transformer" in name:
            continue
        # timm.layers.activations.GELU または nn.GELU のどちらも検出
        if type(module).__name__ == 'GELU':
            gelu_modules.append((name, module))

    hook_count = len(gelu_modules)
    print(f"CLIP: 検出した活性化層数: {hook_count}")

    layer_files = []

    # 層ごとに処理（メモリ効率のため）
    for target_layer_idx in range(hook_count):
        current_layer_idx = [0]  # クロージャ用
        layer_features = []

        def hook_fn(module, inp, out):
            # 現在の対象層のみ保存
            if current_layer_idx[0] == target_layer_idx:
                feat = out.detach().cpu()
                if len(feat.shape) > 2:
                    feat = feat.reshape(feat.size(0), -1)
                layer_features.append(feat)
            current_layer_idx[0] += 1

        # フックを登録（gelu_modulesを使用）
        handles = []
        for name, module in gelu_modules:
            handle = module.register_forward_hook(hook_fn)
            handles.append(handle)

        # 全画像を処理
        with torch.no_grad():
            for image, idx in tqdm(dataset, desc=f"CLIP Layer {target_layer_idx+1}/{hook_count}", leave=False):
                current_layer_idx[0] = 0  # リセット
                image = image.unsqueeze(0).to(device)
                _ = model.encode_image(image)

        # フックを解除
        for handle in handles:
            handle.remove()

        # 特徴量を結合してファイルに保存
        if layer_features:
            batch_feats = torch.cat(layer_features, dim=0)
            save_path = os.path.join(save_dir, f"CLIP_layer_{target_layer_idx}.pt")
            torch.save(batch_feats, save_path)
            layer_files.append(save_path)
            print(f"CLIP Layer {target_layer_idx}: shape {batch_feats.shape} -> {os.path.basename(save_path)}")

            # メモリ解放（ルートパーティション圧迫防止）
            del batch_feats
            layer_features.clear()
            torch.cuda.empty_cache()
            gc.collect()

    return layer_files


def extract_clip_intermediate_features(model, dataset, device):
    """CLIP中間層から特徴量を抽出"""
    model.eval()

    # CLIPのvisual encoderのブロックを取得
    if hasattr(model.visual, 'trunk'):
        # ConvNext型の場合
        layer_names = [f'visual.trunk.stages.{i}' for i in range(4)]
    else:
        layer_names = []

    layer_features = {name: [] for name in layer_names}

    def get_hook(name):
        def hook(module, input, output):
            feat = output.detach()
            if len(feat.shape) > 2:
                feat = feat.reshape(feat.size(0), -1)
            layer_features[name].append(feat.cpu())
        return hook

    handles = []
    for name in layer_names:
        module = model
        for attr in name.split('.'):
            module = getattr(module, attr)
        handle = module.register_forward_hook(get_hook(name))
        handles.append(handle)

    with torch.no_grad():
        for image, idx in tqdm(dataset, desc="CLIP中間層特徴抽出"):
            image = image.unsqueeze(0).to(device)
            _ = model.encode_image(image)

    for handle in handles:
        handle.remove()

    result = {}
    for name in layer_names:
        if layer_features[name]:
            result[name] = torch.cat(layer_features[name], dim=0)

    return result


# ============================================================================
# Evaluation Functions
# ============================================================================
def pearson_scorer(y_true, y_pred):
    """ピアソン相関係数を計算"""
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def evaluate_layer_with_cv(X, y_dict, layer_name, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """各層の特徴量でクロスバリデーション評価"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}

    for target_name, y in y_dict.items():
        scores = []
        labels = y if isinstance(y, np.ndarray) else y.values

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, labels), 1):
            X_train, X_test = X[train_idx], X[val_idx]
            y_train, y_test = labels[train_idx], labels[val_idx]

            # Ridge回帰
            ridge = Ridge(alpha=RIDGE_ALPHA)
            ridge.fit(X_train, y_train)

            # 予測
            y_pred = ridge.predict(X_test)

            # スコア計算
            score = pearson_scorer(y_test, y_pred)
            scores.append(score)

        # 平均スコア
        results[target_name] = np.mean(scores)

    return results


# ============================================================================
# Visualization Functions
# ============================================================================
def plot_encoding_results(all_results, save_path):
    """エンコーディング分析結果をプロット（RSA_and_regression.ipynb形式）"""
    # 各モデルごとに個別のグラフを作成
    for model_name, layer_results in all_results.items():
        # データの整形
        plot_data = []
        for layer_idx, (layer_name, target_results) in enumerate(layer_results.items()):
            for target_name, score in target_results.items():
                plot_data.append({
                    'Layer': layer_idx,
                    'attribute': target_name,
                    'Score': score
                })

        melt_df = pd.DataFrame(plot_data)

        # カテゴリ定義（RSA_and_regression.ipynbと同じ）
        group_dict = {
            "Subjective value": ["res_L"],
            "Healthiness": ["res_H"],
            "Color (RGB)": ["R", "G", "B"],
            "Nutritional value": ["kcal_100g", "protein_100g", "fat_100g", "carbs_100g"],
        }

        # 各カテゴリの3層移動平均を計算
        df_list = []
        for key, attrs in group_dict.items():
            attr_df = pd.DataFrame(
                melt_df[melt_df["attribute"].isin(attrs)]
                .groupby("Layer")["Score"]
                .mean().sort_index()
                .rolling(window=3, min_periods=1, step=3)
                .mean()
                .dropna()
                .reset_index(drop=True)
            )
            attr_df["attr"] = key
            attr_df.index += 1
            df_list.append(attr_df)

        data = pd.concat(df_list)

        # プロット作成（2x2グリッド）
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for i, (attr, d) in enumerate(data.groupby("attr")):
            ax = axes.flatten()[i]
            sns.lineplot(
                data=d.reset_index(),
                x="index",
                y="Score",
                color=colors[i],
                marker="o",
                markersize=15,
                linewidth=4,
                ax=ax
            )
            ax.set_ylim(0.15, 1)

            if i > 1:
                ax.set_xlabel("Layer (3-layer average)", fontsize=36, fontweight="bold")
                ax.set_xticklabels(
                    range(0, int(d.index.max()) + 2, 2),
                    fontsize=30,
                    fontweight="bold",
                )
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if i % 2 == 0:
                ax.set_ylabel("")
                labels = ax.get_yticklabels()
                new_labels = [label if j % 2 == 0 else "" for j, label in enumerate(labels)]
                ax.set_yticklabels(new_labels, fontsize=32, fontweight="bold")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            ax.set_title(attr, fontsize=40, fontweight="bold", pad=-2)

        plt.tight_layout()

        # y軸ラベル（回転）
        plt.text(
            s="Explanatory power",
            x=-14.2,
            y=1,
            fontsize=42,
            fontweight="bold",
            va="center",
            rotation=90,
        )

        # タイトル
        plt.text(
            x=-4.2,
            y=2.27,
            s=f"Representation in {model_name}",
            fontsize=42,
            fontweight="bold",
            va="center",
        )

        # 保存
        model_save_path = save_path.replace('.png', f'_{model_name}.png')
        fig.savefig(model_save_path, bbox_inches="tight")
        print(f"プロットを保存: {model_save_path}")
        plt.close()


def plot_encoding_combined(all_results, save_path):
    """PretrainedとFinetunedを同じグラフに表示（Finetuned=点線）"""
    # ConvNeXt PretrainedとFinetunedをペアで処理
    model_pairs = [
        ("ConvNeXt", "ConvNeXt_Finetuned"),
        ("CLIP_ConvNeXt", None),  # CLIPはFinetunedなし
    ]

    for pretrained_name, finetuned_name in model_pairs:
        if pretrained_name not in all_results:
            continue

        # カテゴリ定義
        group_dict = {
            "Subjective value": ["res_L"],
            "Healthiness": ["res_H"],
            "Color (RGB)": ["R", "G", "B"],
            "Nutritional value": ["kcal_100g", "protein_100g", "fat_100g", "carbs_100g"],
        }

        # プロット作成（2x2グリッド）
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        # Pretrainedのデータ処理
        pretrained_results = all_results[pretrained_name]
        plot_data_pre = []
        for layer_idx, (layer_name, target_results) in enumerate(pretrained_results.items()):
            for target_name, score in target_results.items():
                plot_data_pre.append({
                    'Layer': layer_idx,
                    'attribute': target_name,
                    'Score': score
                })
        melt_df_pre = pd.DataFrame(plot_data_pre)

        # Finetunedのデータ処理（存在する場合）
        melt_df_ft = None
        if finetuned_name and finetuned_name in all_results:
            finetuned_results = all_results[finetuned_name]
            plot_data_ft = []
            for layer_idx, (layer_name, target_results) in enumerate(finetuned_results.items()):
                for target_name, score in target_results.items():
                    plot_data_ft.append({
                        'Layer': layer_idx,
                        'attribute': target_name,
                        'Score': score
                    })
            melt_df_ft = pd.DataFrame(plot_data_ft)

        for i, (attr, attrs) in enumerate(group_dict.items()):
            ax = axes.flatten()[i]

            # Pretrained（実線）
            attr_df_pre = pd.DataFrame(
                melt_df_pre[melt_df_pre["attribute"].isin(attrs)]
                .groupby("Layer")["Score"]
                .mean().sort_index()
                .rolling(window=3, min_periods=1, step=3)
                .mean()
                .dropna()
                .reset_index(drop=True)
            )
            attr_df_pre.index += 1

            ax.plot(
                attr_df_pre.index,
                attr_df_pre["Score"],
                color=colors[i],
                marker="o",
                markersize=12,
                linewidth=3,
                linestyle="-",
                label="Pretrained"
            )

            # Finetuned（点線）
            if melt_df_ft is not None:
                attr_df_ft = pd.DataFrame(
                    melt_df_ft[melt_df_ft["attribute"].isin(attrs)]
                    .groupby("Layer")["Score"]
                    .mean().sort_index()
                    .rolling(window=3, min_periods=1, step=3)
                    .mean()
                    .dropna()
                    .reset_index(drop=True)
                )
                attr_df_ft.index += 1

                ax.plot(
                    attr_df_ft.index,
                    attr_df_ft["Score"],
                    color=colors[i],
                    marker="s",
                    markersize=10,
                    linewidth=3,
                    linestyle="--",
                    label="Finetuned"
                )

            ax.set_ylim(0.15, 1)

            if i > 1:
                ax.set_xlabel("Layer (3-layer average)", fontsize=36, fontweight="bold")
                ax.tick_params(axis='x', labelsize=30)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if i % 2 == 0:
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelsize=32)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            ax.set_title(attr, fontsize=40, fontweight="bold", pad=-2)

            # 凡例（最初のサブプロットのみ）
            if i == 0 and melt_df_ft is not None:
                ax.legend(fontsize=20, loc="lower right")

        plt.tight_layout()

        # y軸ラベル（回転）
        plt.text(
            s="Explanatory power",
            x=-14.2,
            y=1,
            fontsize=42,
            fontweight="bold",
            va="center",
            rotation=90,
        )

        # タイトル（モデル名のみ）
        base_name = pretrained_name.replace("_Pretrained", "").replace("CLIP_", "CLIP ")
        plt.text(
            x=-4.2,
            y=2.27,
            s=base_name,
            fontsize=42,
            fontweight="bold",
            va="center",
        )

        # 保存
        model_save_path = save_path.replace('.png', f'_{pretrained_name}_combined.png')
        fig.savefig(model_save_path, bbox_inches="tight")
        print(f"プロットを保存: {model_save_path}")
        plt.close()


def plot_encoding_barplot(all_results, save_path):
    """エンコーディング分析のグレーバーチャート（日本語）"""
    # 各モデル×各層×各ターゲットの平均スコアを計算
    plot_data = []
    target_map = {
        "res_L": "好み",
        "res_H": "健康度",
        "res_T": "美味しさ",
        "R": "赤",
        "G": "緑",
        "B": "青",
        "kcal_100g": "カロリー",
        "protein_100g": "タンパク質",
        "fat_100g": "脂質",
        "carbs_100g": "炭水化物"
    }

    category_map = {
        "res_L": "主観的価値",
        "res_H": "健康度",
        "res_T": "美味しさ",
        "R": "色",
        "G": "色",
        "B": "色",
        "kcal_100g": "栄養価",
        "protein_100g": "栄養価",
        "fat_100g": "栄養価",
        "carbs_100g": "栄養価"
    }

    for model_name, layer_results in all_results.items():
        for layer_name, target_results in layer_results.items():
            for target_name, score in target_results.items():
                if target_name in target_map:
                    plot_data.append({
                        "モデル": model_name,
                        "属性": target_map[target_name],
                        "カテゴリ": category_map[target_name],
                        "スコア": score
                    })

    df = pd.DataFrame(plot_data)

    # 各モデル×カテゴリの平均を計算
    avg_df = df.groupby(["モデル", "カテゴリ"])["スコア"].mean().reset_index()

    # カテゴリごとにプロット
    for category in avg_df["カテゴリ"].unique():
        fig = plt.figure(figsize=(16, 9), dpi=300)
        category_df = avg_df[avg_df["カテゴリ"] == category]

        # グレーのパレット
        n_models = len(category_df)
        gray_palette = [
            "#5E5F5F",  # 濃いグレー
            "#7D7D7D",  # 中程度のグレー
            "#959595",  # やや薄いグレー
            "#ADADAD"   # 薄いグレー
        ][:n_models]

        sns.barplot(
            data=category_df,
            x="モデル",
            y="スコア",
            palette=gray_palette,
            width=0.6,
            errorbar=None
        )

        plt.xticks(fontsize=42, fontweight="bold")
        plt.xlabel("", fontsize=36)
        plt.yticks(fontsize=46, fontweight="bold")
        plt.ylabel("相関係数", fontsize=52, labelpad=20, fontweight="bold")
        plt.ylim(0.0, 1.0)
        plt.title(f"{category}の予測精度（層平均）", fontsize=48, fontweight="bold", pad=20)

        plt.tight_layout()

        # 保存
        category_save_path = save_path.replace('.png', f'_{category}.png')
        fig.savefig(category_save_path, bbox_inches="tight")
        print(f"バーチャートを保存: {category_save_path}")
        plt.close()


# ============================================================================
# Main Function
# ============================================================================
def main():
    print("="*80)
    print("統合エンコーディング分析")
    print("="*80)

    # データの読み込み
    print("\nデータを読み込み中...")
    resp = pd.read_csv(os.path.join(DATA_PATH, "data_responses_NCNP_2types.csv"))
    food_value = pd.read_csv(os.path.join(DATA_PATH, "food_value.csv"))

    # 肥満フラグの追加
    resp["is_obesity"] = resp["BMI"] >= 25

    # 外れ値の除外
    outlier = resp["sub_ID"].unique()[
        (resp.groupby("sub_ID")["res_L"].value_counts().unstack() > 896 * 0.75).any(axis=1)
        | (
            (resp.groupby("sub_ID")["res_L"].unique().apply(lambda x: len(x)) <= 4)
            & (resp.groupby("sub_ID")["res_L"].value_counts().unstack() > 896 * 0.65).any(axis=1)
        )
    ]
    print(f"被験者 {outlier} ({len(outlier)}人) を除外")

    resp_filtered = resp[~resp["sub_ID"].isin(outlier)]

    # 主観的価値の平均値計算（全被験者）
    res_L_by_group = resp_filtered.groupby(["img", "is_obesity"])["res_L"].mean()
    res_H_by_group = resp_filtered.groupby(["img", "is_obesity"])["res_H"].mean()
    res_T_by_group = resp_filtered.groupby(["img", "is_obesity"])["res_T"].mean()

    res_L_mean = res_L_by_group.groupby("img").mean()
    res_H_mean = res_H_by_group.groupby("img").mean()
    res_T_mean = res_T_by_group.groupby("img").mean()

    # 栄養価データ
    kcal_100g = food_value["kcal_100g"]
    protein_100g = food_value["protein_100g"]
    fat_100g = food_value["fat_100g"]
    carbs_100g = food_value["carbs_100g"]

    # 欠損値のフィルタリング（元のノートブックと同様）
    gram_value_is_not_nan = ~np.isnan(food_value["grams_total"].values)
    print(f"栄養価データ: 欠損値を除外 ({np.sum(~gram_value_is_not_nan)}個)")

    # 画像RGB値の計算
    image_dir = os.path.join(ROOT_PATH, "Database")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    image_rgb = pd.DataFrame(
        [np.array(Image.open(os.path.join(image_dir, img)).convert("RGB")).mean(axis=(0, 1))
         for img in image_files],
        columns=["R", "G", "B"]
    )

    # ターゲットデータ（欠損値を除外）
    target_data = {
        "res_L": res_L_mean.values[gram_value_is_not_nan],
        "res_H": res_H_mean.values[gram_value_is_not_nan],
        "res_T": res_T_mean.values[gram_value_is_not_nan],
        "kcal_100g": kcal_100g.values[gram_value_is_not_nan],
        "protein_100g": protein_100g.values[gram_value_is_not_nan],
        "fat_100g": fat_100g.values[gram_value_is_not_nan],
        "carbs_100g": carbs_100g.values[gram_value_is_not_nan],
        "R": image_rgb["R"].values[gram_value_is_not_nan],
        "G": image_rgb["G"].values[gram_value_is_not_nan],
        "B": image_rgb["B"].values[gram_value_is_not_nan]
    }

    # 有効なインデックスを保存（特徴量フィルタリング用）
    valid_indices = np.where(gram_value_is_not_nan)[0]

    print(f"データ数: {len(res_L_mean)} -> {len(target_data['res_L'])} (欠損値除外後)")

    # CNN用の前処理
    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cnn_dataset = ImageDataset(image_dir, transform=cnn_transform)

    # ============================================================================
    # エンコーディング分析（ConvNeXtとCLIPのみ、以前のv16分析と同様に全GELU層から抽出）
    # 抽出と評価を分離し、メモリ効率を改善
    # ============================================================================
    all_results = {}

    # 特徴量保存先ディレクトリ（rcloneキャッシュを避けるため直接ローカルパスを使用）
    features_dir = "/mnt/ProjectC/dev/foodReward/tmp/features"
    os.makedirs(features_dir, exist_ok=True)

    # ConvNext Base（全GELU層から抽出 = 36層）
    print("\n" + "="*80)
    print("ConvNext Baseのエンコーディング分析（全GELU層）")
    print("="*80)

    # 既存の特徴量ファイルをチェック
    existing_convnext = sorted(glob.glob(os.path.join(features_dir, "ConvNeXt_layer_*.pt")))
    if len(existing_convnext) >= 36:
        print(f"既存のConvNeXt特徴量を使用 ({len(existing_convnext)}ファイル)")
        convnext_layer_files = existing_convnext
    else:
        convnext_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1).to(device)
        # Step 1: 特徴量抽出（層ごとにファイル保存）
        print("\n[Step 1] 特徴量抽出...")
        convnext_layer_files = extract_features_by_activation(
            convnext_model, cnn_dataset, device,
            save_dir=features_dir, model_type='ConvNeXt'
        )
        # モデルをメモリから解放
        del convnext_model
        torch.cuda.empty_cache()

    # Step 2: 評価（既存結果があればスキップ）
    existing_json = os.path.join(OUTPUT_DIR, "encoding_results_20251130_153055.json")
    if os.path.exists(existing_json):
        import json
        with open(existing_json, 'r') as f:
            existing_results = json.load(f)
        if "ConvNeXt" in existing_results and len(existing_results["ConvNeXt"]) >= 36:
            print(f"既存のConvNeXt評価結果を使用 ({len(existing_results['ConvNeXt'])}層)")
            all_results["ConvNeXt"] = existing_results["ConvNeXt"]
        else:
            print("\n[Step 2] CV評価...")
            convnext_results = evaluate_features_from_files(convnext_layer_files, target_data, valid_indices)
            all_results["ConvNeXt"] = convnext_results
    else:
        print("\n[Step 2] CV評価...")
        convnext_results = evaluate_features_from_files(convnext_layer_files, target_data, valid_indices)
        all_results["ConvNeXt"] = convnext_results

    # ConvNext Base Finetuned（全GELU層から抽出 = 36層）
    print("\n" + "="*80)
    print("ConvNext Base Finetunedのエンコーディング分析（全GELU層）")
    print("="*80)

    # Finetunedモデルのパス（v9）
    finetuned_model_path = "/mnt/h/foodReward/model/v9/res_L/convnext_base_regression.pth"
    if os.path.exists(finetuned_model_path):
        print(f"Finetunedモデルを読み込み: {finetuned_model_path}")

        # 既存の特徴量ファイルをチェック
        existing_finetuned = sorted(glob.glob(os.path.join(features_dir, "ConvNeXt_Finetuned_layer_*.pt")))
        if len(existing_finetuned) >= 36:
            print(f"既存のConvNeXt Finetuned特徴量を使用 ({len(existing_finetuned)}ファイル)")
            finetuned_layer_files = existing_finetuned
        else:
            # Finetunedモデルを読み込み
            finetuned_model = models.convnext_base(weights=None)
            finetuned_model.classifier[2] = nn.Linear(finetuned_model.classifier[2].in_features, 1)
            finetuned_model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
            finetuned_model = finetuned_model.to(device)

            # Step 1: 特徴量抽出（層ごとにファイル保存）
            print("\n[Step 1] 特徴量抽出...")
            finetuned_layer_files = extract_features_by_activation(
                finetuned_model, cnn_dataset, device,
                save_dir=features_dir, model_type='ConvNeXt_Finetuned'
            )
            # モデルをメモリから解放
            del finetuned_model
            gc.collect()

        # Step 2: 評価
        print("\n[Step 2] CV評価...")
        finetuned_results = evaluate_features_from_files(finetuned_layer_files, target_data, valid_indices)
        all_results["ConvNeXt_Finetuned"] = finetuned_results
    else:
        print("Finetunedモデルが見つかりません。スキップします。")

    # CLIP ConvNeXt Base（全GELU層から抽出）
    print("\n" + "="*80)
    print("CLIP ConvNeXt Baseのエンコーディング分析（全GELU層）")
    print("="*80)
    clip_model, _, clip_preprocess = create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED
    )
    clip_model = clip_model.to(device)
    clip_dataset = ImageDataset(image_dir, transform=clip_preprocess)

    # Step 1: 特徴量抽出（層ごとにファイル保存）
    print("\n[Step 1] 特徴量抽出...")
    clip_layer_files = extract_clip_features_by_activation(
        clip_model, clip_dataset, device,
        save_dir=features_dir
    )

    # モデルをメモリから解放
    del clip_model
    torch.cuda.empty_cache()

    # Step 2: 評価
    print("\n[Step 2] CV評価...")
    clip_results = evaluate_features_from_files(clip_layer_files, target_data, valid_indices)
    all_results["CLIP_ConvNeXt"] = clip_results

    # ============================================================================
    # 結果の保存と可視化
    # ============================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 可視化
    plot_path = os.path.join(OUTPUT_DIR, f"encoding_results_{timestamp}.png")
    plot_encoding_combined(all_results, plot_path)  # PretrainedとFinetunedを同じグラフに

    # 詳細結果をJSON保存
    results_json = {}
    for model_name, layer_results in all_results.items():
        results_json[model_name] = {}
        for layer_name, target_results in layer_results.items():
            results_json[model_name][layer_name] = {
                target: float(score) for target, score in target_results.items()
            }

    json_path = os.path.join(OUTPUT_DIR, f"encoding_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n詳細結果を保存: {json_path}")

    print("\n" + "="*80)
    print("エンコーディング分析完了!")
    print("="*80)


if __name__ == "__main__":
    main()
