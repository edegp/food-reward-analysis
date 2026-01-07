#!/usr/bin/env python3
"""
統合クロスバリデーション分析スクリプト
CNNモデル（VGG16, ConvNext, ResNet152）とCLIP（ConvNeXt Base）を
同じクロスバリデーション分割で評価します。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json
from datetime import datetime

# OpenCLIP imports
from open_clip import create_model_and_transforms, get_tokenizer

# Torchvision models
from torchvision import models

# Local imports
from src.const import DATA_PATH, ROOT_PATH

plt.rcParams["font.serif"] = ["noto"]

# ============================================================================
# Configuration
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")

# クロスバリデーション設定（両方のノートブックで統一）
N_SPLITS = 8
RANDOM_STATE = 42

# CLIP設定 - ConvNeXt Baseを使用（224x224入力でCNNと統一）
CLIP_MODEL_NAME = "convnext_base"  # 同じConvNeXtアーキテクチャで事前学習データの比較
CLIP_PRETRAINED = "laion400m_s13b_b51k"

# Ridge回帰のハイパーパラメータ
RIDGE_ALPHA = 1.0

# PCA次元削減設定
PCA_DIM = 512  # 全モデルをこの次元数に統一

# 保存先ディレクトリ
OUTPUT_DIR = os.path.join(ROOT_PATH, "results", "unified_cv_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Dataset Classes
# ============================================================================
class ImageDataset(Dataset):
    """CNN用の画像データセット"""

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


class CLIPDataset(Dataset):
    """CLIP用の画像データセット"""

    def __init__(self, image_dir: str, df: pd.DataFrame, transform=None):
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['img']:04d}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, idx


# ============================================================================
# Feature Extraction Functions
# ============================================================================
def extract_cnn_features(model, dataset, device, model_type='vgg'):
    """CNNモデルから特徴量を抽出（プリトレインモデルから）"""
    model.eval()
    feature_vectors = []

    with torch.no_grad():
        for image, idx in tqdm(dataset, desc="特徴抽出中"):
            image = image.unsqueeze(0).to(device)

            if model_type == 'vgg':
                # VGG: avgpool の後の特徴量を取得
                x = model.features(image)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                feature_vectors.append(x.cpu())
            elif model_type == 'resnet':
                # ResNet: avgpool の後の特徴量を取得
                x = model.conv1(image)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                feature_vectors.append(x.cpu())
            elif model_type == 'convnext':
                # ConvNext: avgpool の後の特徴量を取得
                x = model.features(image)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                feature_vectors.append(x.cpu())

    return torch.cat(feature_vectors, dim=0)


def extract_clip_features(model, dataset, device):
    """CLIPモデルから画像特徴量を抽出"""
    model.eval()
    feature_list = []

    with torch.no_grad():
        for image, idx in tqdm(dataset, desc="CLIP特徴抽出中"):
            image = image.unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            feature_list.append(image_features.cpu())

    return torch.cat(feature_list, dim=0)


# ============================================================================
# Evaluation Functions
# ============================================================================
def pearson_scorer(y_true, y_pred):
    """ピアソン相関係数を計算"""
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def apply_pca(X, n_components=PCA_DIM):
    """PCAで次元削減"""
    print(f"PCA適用前の次元数: {X.shape[1]}")

    if X.shape[1] <= n_components:
        print(f"次元数が{n_components}以下のため、PCAをスキップします")
        return X

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_reduced = pca.fit_transform(X)

    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA適用後の次元数: {X_reduced.shape[1]}")
    print(f"保持された分散: {explained_variance:.2f}%")

    return X_reduced


def evaluate_with_cv(X, y_dict, model_name, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """クロスバリデーションによる評価"""
    print(f"\n{'='*60}")
    print(f"モデル: {model_name}")
    print(f"{'='*60}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}

    for target_name, y in y_dict.items():
        print(f"\nターゲット: {target_name}")

        results[target_name] = {
            "y_tests": [],
            "y_preds": [],
            "scores": [],
            "mse_scores": []
        }

        labels = y.values

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, labels), 1):
            X_train, X_test = X[train_idx], X[val_idx]
            y_train, y_test = labels[train_idx], labels[val_idx]

            # Ridge回帰
            ridge = Ridge(alpha=RIDGE_ALPHA)
            ridge.fit(X_train, y_train)

            # 予測
            y_pred = ridge.predict(X_test)

            # 結果保存
            results[target_name]["y_tests"].append(y_test)
            results[target_name]["y_preds"].append(y_pred)

            # スコア計算
            score = pearson_scorer(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            results[target_name]["scores"].append(score)
            results[target_name]["mse_scores"].append(mse)

            print(f"  Fold {fold_idx}: r={score:.4f}, MSE={mse:.4f}")

        # 全体のスコア
        all_y_test = np.concatenate(results[target_name]["y_tests"])
        all_y_pred = np.concatenate(results[target_name]["y_preds"])
        overall_score = pearson_scorer(all_y_test, all_y_pred)
        overall_mse = mean_squared_error(all_y_test, all_y_pred)

        results[target_name]["overall_score"] = overall_score
        results[target_name]["overall_mse"] = overall_mse
        results[target_name]["avg_score"] = np.mean(results[target_name]["scores"])
        results[target_name]["avg_mse"] = np.mean(results[target_name]["mse_scores"])

        print(f"\n  平均: r={results[target_name]['avg_score']:.4f}, MSE={results[target_name]['avg_mse']:.4f}")
        print(f"  全体: r={overall_score:.4f}, MSE={overall_mse:.4f}")

    return results


# ============================================================================
# Visualization Functions
# ============================================================================
def plot_results(all_results, save_path):
    """結果をプロット"""
    model_names = list(all_results.keys())
    target_names = ["res_L"]
    target_titles = ["Preference"]

    n_models = len(model_names)
    n_targets = len(target_names)

    fig, axes = plt.subplots(n_models, n_targets, figsize=(20, 5*n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("統合クロスバリデーション結果", fontsize=16, y=0.995)

    for i, model_name in enumerate(model_names):
        results = all_results[model_name]

        for j, (target_name, title) in enumerate(zip(target_names, target_titles)):
            ax = axes[i, j]

            if target_name in results:
                y_test = np.concatenate(results[target_name]["y_tests"])
                y_pred = np.concatenate(results[target_name]["y_preds"])

                ax.scatter(y_test, y_pred, alpha=0.6, s=20)

                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

                corr = results[target_name]["overall_score"]
                mse = results[target_name]["overall_mse"]

                ax.set_title(f"{model_name} - {title}\nr={corr:.4f}, MSE={mse:.4f}")
                ax.set_xlabel("True")
                ax.set_ylabel("Predicted")
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"{model_name} - {title}\nNo data")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nプロットを保存: {save_path}")
    plt.close()


def create_summary_table(all_results, save_path):
    """サマリーテーブルを作成"""
    target_names = ["res_L"]

    summary_data = []
    for model_name, results in all_results.items():
        row = {"Model": model_name}
        for target_name in target_names:
            if target_name in results:
                score = results[target_name]["overall_score"]
                row[target_name] = f"{score:.4f}"
            else:
                row[target_name] = "N/A"
        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    print("\n" + "="*80)
    print("モデル比較サマリー（相関係数）")
    print("="*80)
    print(df.to_string(index=False))

    # 最高性能モデル
    print("\n" + "="*80)
    print("各ターゲットでの最高性能モデル")
    print("="*80)

    for target_name in target_names:
        best_score = -1
        best_model = "N/A"

        for model_name, results in all_results.items():
            if target_name in results:
                score = results[target_name]["overall_score"]
                if score > best_score:
                    best_score = score
                    best_model = model_name

        print(f"{target_name}: {best_model} ({best_score:.4f})")

    # CSVに保存
    df.to_csv(save_path, index=False)
    print(f"\nサマリーテーブルを保存: {save_path}")


def plot_correlation_barplot(all_results, save_path):
    """相関係数のグレーバーチャート（日本語）"""
    # データの準備
    plot_data = []
    target_map = {
        "res_L": "好み"
    }

    for target_name, target_jp in target_map.items():
        for model_name, results in all_results.items():
            if target_name in results:
                plot_data.append({
                    "モデル": model_name,
                    "属性": target_jp,
                    "相関係数": results[target_name]["overall_score"]
                })

    df = pd.DataFrame(plot_data)

    # 各属性ごとにプロット
    for target_jp in target_map.values():
        fig = plt.figure(figsize=(16, 9), dpi=300)
        target_df = df[df["属性"] == target_jp]

        # グレーのパレット
        n_models = len(target_df)
        gray_palette = [
            "#5E5F5F",  # 濃いグレー
            "#7D7D7D",  # 中程度のグレー
            "#959595",  # やや薄いグレー
            "#ADADAD"   # 薄いグレー
        ][:n_models]

        sns.barplot(
            data=target_df,
            x="モデル",
            y="相関係数",
            palette=gray_palette,
            width=0.6,
            errorbar=None
        )

        plt.xticks(fontsize=42, fontweight="bold")
        plt.xlabel("", fontsize=36)
        plt.yticks(fontsize=46, fontweight="bold")
        plt.ylabel("相関係数", fontsize=52, labelpad=20, fontweight="bold")
        plt.ylim(0.0, 1.0)
        plt.title(f"{target_jp}の予測精度", fontsize=48, fontweight="bold", pad=20)

        plt.tight_layout()

        # 保存
        target_save_path = save_path.replace('.png', f'_{target_jp}.png')
        fig.savefig(target_save_path, bbox_inches="tight")
        print(f"バーチャートを保存: {target_save_path}")
        plt.close()


# ============================================================================
# Main Function
# ============================================================================
def main():
    print("="*80)
    print("統合クロスバリデーション分析")
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

    # 主観的価値の平均値計算（全被験者：健常者+肥満者の平均）
    # まず各グループ（健常者/肥満者）ごとの平均を計算
    res_L_by_group = resp_filtered.groupby(["img", "is_obesity"])["res_L"].mean()

    # 各画像について健常者と肥満者の平均を取る
    res_L_mean = res_L_by_group.groupby("img").mean()

    # ターゲットデータ
    target_data = {
        "res_L": res_L_mean
    }

    print(f"データ数: {len(res_L_mean)}")

    # 画像ディレクトリ
    image_dir = os.path.join(ROOT_PATH, "Database")

    # CNN用の前処理
    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # CNNデータセット
    cnn_dataset = ImageDataset(image_dir, transform=cnn_transform)

    # ============================================================================
    # CNNモデルの評価（プリトレインモデル、分類器付け替えなし）
    # ============================================================================
    all_results = {}

    # VGG16
    print("\n" + "="*80)
    print("VGG16モデルの特徴抽出（プリトレイン）")
    print("="*80)
    vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    vgg_features = extract_cnn_features(vgg_model, cnn_dataset, device, model_type='vgg')
    print(f"VGG16特徴量の形状: {vgg_features.shape}")

    # PCAで次元削減
    vgg_features_pca = apply_pca(vgg_features.cpu().numpy())

    vgg_results = evaluate_with_cv(vgg_features_pca, target_data, "VGG16_Pretrained")
    all_results["VGG16_Pretrained"] = vgg_results

    # ConvNext Base
    print("\n" + "="*80)
    print("ConvNext Baseモデルの特徴抽出（プリトレイン）")
    print("="*80)
    convnext_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1).to(device)
    convnext_features = extract_cnn_features(convnext_model, cnn_dataset, device, model_type='convnext')
    print(f"ConvNext特徴量の形状: {convnext_features.shape}")

    # PCAで次元削減
    convnext_features_pca = apply_pca(convnext_features.cpu().numpy())

    convnext_results = evaluate_with_cv(convnext_features_pca, target_data, "ConvNeXt_Pretrained")
    all_results["ConvNeXt_Pretrained"] = convnext_results

    # ResNet152
    print("\n" + "="*80)
    print("ResNet152モデルの特徴抽出（プリトレイン）")
    print("="*80)
    resnet_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).to(device)
    resnet_features = extract_cnn_features(resnet_model, cnn_dataset, device, model_type='resnet')
    print(f"ResNet152特徴量の形状: {resnet_features.shape}")

    # PCAで次元削減
    resnet_features_pca = apply_pca(resnet_features.cpu().numpy())

    resnet_results = evaluate_with_cv(resnet_features_pca, target_data, "ResNet152_Pretrained")
    all_results["ResNet152_Pretrained"] = resnet_results

    # ============================================================================
    # CLIPモデルの評価
    # ============================================================================
    print("\n" + "="*80)
    print("CLIP (ConvNeXt Base)モデルの特徴抽出")
    print("="*80)

    # CLIPモデルとトランスフォームの作成
    clip_model, _, clip_preprocess = create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED
    )
    clip_model = clip_model.to(device)

    # CLIP用のデータフレーム作成
    clip_df = pd.DataFrame({
        "img": range(1, 897)
    })

    # CLIPデータセット
    clip_dataset = CLIPDataset(image_dir, clip_df, transform=clip_preprocess)

    # CLIP特徴量抽出
    clip_features = extract_clip_features(clip_model, clip_dataset, device)
    print(f"CLIP特徴量の形状: {clip_features.shape}")

    # PCAで次元削減
    clip_features_pca = apply_pca(clip_features.cpu().numpy())

    clip_results = evaluate_with_cv(clip_features_pca, target_data, "CLIP_ConvNeXt_Base")
    all_results["CLIP_ConvNeXt_Base"] = clip_results

    # ============================================================================
    # 結果の保存と可視化
    # ============================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # サマリーテーブル（コンソール出力とCSV保存）
    summary_path = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.csv")
    create_summary_table(all_results, summary_path)

    # 可視化はノートブックで実行するため、ここではスキップ
    # plot_path = os.path.join(OUTPUT_DIR, f"results_{timestamp}.png")
    # plot_results(all_results, plot_path)
    # barplot_path = os.path.join(OUTPUT_DIR, f"barplot_{timestamp}.png")
    # plot_correlation_barplot(all_results, barplot_path)

    # 詳細結果をJSON保存
    results_json = {}
    for model_name, results in all_results.items():
        results_json[model_name] = {}
        for target_name, target_results in results.items():
            results_json[model_name][target_name] = {
                "overall_score": float(target_results["overall_score"]),
                "overall_mse": float(target_results["overall_mse"]),
                "avg_score": float(target_results["avg_score"]),
                "avg_mse": float(target_results["avg_mse"]),
                "fold_scores": [float(s) for s in target_results["scores"]],
                "fold_mse": [float(s) for s in target_results["mse_scores"]]
            }

    json_path = os.path.join(OUTPUT_DIR, f"detailed_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n詳細結果を保存: {json_path}")

    print("\n" + "="*80)
    print("分析完了!")
    print("="*80)


if __name__ == "__main__":
    main()
