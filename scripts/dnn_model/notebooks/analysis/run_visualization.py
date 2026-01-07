#!/usr/bin/env python3
"""
可視化を実行（Notebookのコードをスクリプトとして実行）
"""

import os
import json
import sys
from glob import glob

# matplotlibとseabornのインポート
try:
    import matplotlib
    matplotlib.use('Agg')  # GUIなしでグラフを保存
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import dill
except ImportError as e:
    print(f"エラー: 必要なライブラリがインストールされていません")
    print(f"詳細: {e}")
    print("\n以下のコマンドでインストールしてください:")
    print("  pip install matplotlib seaborn pandas dill")
    sys.exit(1)

# 環境設定
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_PATH)

# 日本語フォント設定（RSA_and_regression_v16.ipynbと同じ設定）
sns.set_theme(font="BIZ UDPGothic")
sns.set_style("whitegrid")
plt.rcParams["font.family"] = ["BIZ UDPGothic"]

# モデルの表示順序（CLIP, ConvNeXt, ResNet, VGG16）
# CV結果用のキー名
MODEL_ORDER_CV = ["CLIP_ConvNeXt_Base", "ConvNeXt_Pretrained", "ResNet152_Pretrained", "VGG16_Pretrained"]
# Encoding結果用のキー名
MODEL_ORDER_ENC = ["CLIP_ConvNeXt", "ConvNeXt", "ResNet152", "VGG16"]

MODEL_DISPLAY_NAMES = {
    # CV用
    "CLIP_ConvNeXt_Base": "CLIP",
    "ConvNeXt_Pretrained": "ConvNeXt",
    "ResNet152_Pretrained": "ResNet",
    "VGG16_Pretrained": "VGG16",
    # Encoding用
    "CLIP_ConvNeXt": "CLIP",
    "ConvNeXt": "ConvNeXt",
    "ResNet152": "ResNet",
    "VGG16": "VGG16",
}

# 結果ディレクトリ
cv_results_dir = os.path.join(ROOT_PATH, "data", "output", "results", "unified_cv_analysis")
encoding_results_dir = os.path.join(ROOT_PATH, "data", "output", "results", "unified_encoding_analysis")
output_dir = os.path.join(ROOT_PATH, "data", "output", "results", "visualizations")
os.makedirs(output_dir, exist_ok=True)

# ファインチューニング結果のパス
finetuned_v9_path = os.path.join(ROOT_PATH, "model", "v9", "res_L", "avg_metric.pkl")
finetuned_v10_path = os.path.join(ROOT_PATH, "model", "v10", "res_L", "avg_metric.pkl")

print("="*80)
print("可視化実行")
print("="*80)

# 最新のJSONファイルを取得（テストデータを除外）
cv_json_files = sorted([f for f in glob(os.path.join(cv_results_dir, "detailed_results_*.json")) if "_test" not in f])
encoding_json_files = sorted([f for f in glob(os.path.join(encoding_results_dir, "encoding_results_*.json")) if "_test" not in f])

if not cv_json_files or not encoding_json_files:
    print("❌ テストデータが見つかりません。")
    sys.exit(1)

latest_cv_json = cv_json_files[-1]
latest_encoding_json = encoding_json_files[-1]

print(f"読み込み: {os.path.basename(latest_cv_json)}")
print(f"読み込み: {os.path.basename(latest_encoding_json)}")

# JSONファイルを読み込み
with open(latest_cv_json, 'r') as f:
    cv_results = json.load(f)

with open(latest_encoding_json, 'r') as f:
    encoding_results = json.load(f)

print("\n✓ データ読み込み完了")

# ============================================================================
# 1. CVバーチャート（グレー）
# ============================================================================
print("\n1. CVバーチャート（グレー）を生成中...")

plot_data = []
target_map = {
    "res_L": "主観的価値（好み）"
}

# モデル順序に従ってデータを追加
for target_name, target_jp in target_map.items():
    for model_name in MODEL_ORDER_CV:
        if model_name in cv_results and target_name in cv_results[model_name]:
            plot_data.append({
                "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                "Attribute": target_jp,
                "Correlation": cv_results[model_name][target_name]["overall_score"]
            })

df = pd.DataFrame(plot_data)

# 各属性ごとにプロット
for idx, target_jp in enumerate(target_map.values()):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    target_df = df[df["Attribute"] == target_jp]

    # グレーのパレット
    n_models = len(target_df)
    gray_palette = [
        "#5E5F5F",  # 濃いグレー
        "#7D7D7D",  # 中程度のグレー
        "#959595",  # やや薄いグレー
        "#ADADAD"   # 薄いグレー
    ][:n_models]

    # モデル順序を維持するためにorderを指定
    model_order_display = [MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER_CV if MODEL_DISPLAY_NAMES.get(m, m) in target_df["Model"].values]

    sns.barplot(
        data=target_df,
        x="Model",
        y="Correlation",
        palette=gray_palette,
        width=0.6,
        order=model_order_display
    )

    plt.xticks(fontsize=42, fontweight="bold", rotation=15)
    plt.xlabel("", fontsize=36)
    plt.yticks(fontsize=46, fontweight="bold")
    plt.ylabel("相関係数", fontsize=46, labelpad=20, fontweight="bold")
    plt.ylim(0.0, 1.0)
    plt.title(f"{target_jp}の予測精度", fontsize=32, fontweight="bold", pad=20)

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"cv_barplot_{idx+1}_preference.png")
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  ✓ 保存: {os.path.basename(save_path)}")
    plt.close()

# ============================================================================
# 2. エンコーディング折れ線グラフ（RSA_and_regression copy.ipynb形式）
# ============================================================================
print("\n2. エンコーディング折れ線グラフを生成中...")

# カテゴリ定義（RSA_and_regression copy.ipynbと同じ）
group_dict = {
    "主観的価値": ["res_L"],
    "健康度": ["res_H"],
    "栄養価": ["kcal_100g", "protein_100g", "fat_100g", "carbs_100g"],
    "色（RGB）": ["R", "G", "B"],
}

# カテゴリ順序（groupbyはアルファベット順になるので明示的に指定）
category_order = ["主観的価値", "健康度", "栄養価", "色（RGB）"]
# Seaborn mutedパレット
colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]

# モデル名とタイトルのマッピング
model_titles = {
    "CLIP_ConvNeXt": "CLIPでの内部表現",
    "ConvNeXt": "DCNNでの内部表現",
}

for model_key in ["CLIP_ConvNeXt", "ConvNeXt"]:
    if model_key not in encoding_results:
        print(f"  ⚠ {model_key} のデータがありません")
        continue

    layer_results = encoding_results[model_key]

    # データの整形
    plot_data = []
    for layer_idx, (layer_name, target_results) in enumerate(layer_results.items()):
        for target_name, score in target_results.items():
            plot_data.append({
                'Layer': layer_idx,
                'target': target_name,
                'Score': score
            })

    melt_df = pd.DataFrame(plot_data)

    # 各カテゴリの3層移動平均を計算
    df_list = []
    for key, attrs in group_dict.items():
        filtered = melt_df[melt_df["target"].isin(attrs)]
        if len(filtered) == 0:
            continue
        attr_df = pd.DataFrame(
            filtered
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

    # プロット作成（2x2グリッド）- RSA_and_regression copy.ipynbと同じ形式
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)

    for i, attr in enumerate(category_order):
        d = data[data["attr"] == attr]
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
        ax.set_ylim(0.1, 1)

        if i > 1:
            ax.set_xlabel("層（3層平均）", fontsize=36, fontweight="bold")
            ax.set_xticklabels(
                range(0, 13, 2),
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
        s="予測精度",
        x=-14.8,
        y=1,
        fontsize=42,
        fontweight="bold",
        va="center",
        rotation=90,
    )

    # タイトル
    plt.text(
        x=-4.2,
        y=2.46,
        s=model_titles.get(model_key, f"{model_key}での内部表現"),
        fontsize=42,
        fontweight="bold",
        va="center",
    )

    # ファイル名
    if model_key == "CLIP_ConvNeXt":
        filename = "encoding_lineplot_clip_36layers.png"
    else:
        filename = "encoding_lineplot_convnext_36layers.png"

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  ✓ 保存: {os.path.basename(save_path)}")
    plt.close()

# ============================================================================
# 3. CLIPとConvNeXtを重ねた折れ線グラフ
# ============================================================================
print("\n3. CLIPとConvNeXt重ね合わせグラフを生成中...")

# 両モデルのデータを準備
model_data = {}
for model_key in ["CLIP_ConvNeXt", "ConvNeXt"]:
    if model_key not in encoding_results:
        continue

    layer_results = encoding_results[model_key]
    plot_data = []
    for layer_idx, (layer_name, target_results) in enumerate(layer_results.items()):
        for target_name, score in target_results.items():
            plot_data.append({
                'Layer': layer_idx,
                'target': target_name,
                'Score': score
            })

    melt_df = pd.DataFrame(plot_data)

    # 各カテゴリの3層移動平均を計算
    df_list = []
    for key, attrs in group_dict.items():
        filtered = melt_df[melt_df["target"].isin(attrs)]
        if len(filtered) == 0:
            continue
        attr_df = pd.DataFrame(
            filtered
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

    model_data[model_key] = pd.concat(df_list)

# 2x2グリッドでプロット（CLIPとConvNeXtを重ねる）
fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)

for i, attr in enumerate(category_order):
    ax = axes.flatten()[i]

    # ConvNeXt（実線）
    if "ConvNeXt" in model_data:
        d_convnext = model_data["ConvNeXt"][model_data["ConvNeXt"]["attr"] == attr]
        sns.lineplot(
            data=d_convnext.reset_index(),
            x="index",
            y="Score",
            color=colors[i],
            marker="o",
            markersize=12,
            linewidth=3,
            ax=ax,
            label="DCNN"
        )

    # CLIP（破線）
    if "CLIP_ConvNeXt" in model_data:
        d_clip = model_data["CLIP_ConvNeXt"][model_data["CLIP_ConvNeXt"]["attr"] == attr]
        sns.lineplot(
            data=d_clip.reset_index(),
            x="index",
            y="Score",
            color=colors[i],
            marker="s",
            markersize=12,
            linewidth=3,
            linestyle="--",
            ax=ax,
            label="CLIP"
        )

    ax.set_ylim(0.1, 1)

    if i > 1:
        ax.set_xlabel("層（3層平均）", fontsize=36, fontweight="bold")
        ax.set_xticklabels(
            range(0, 13, 2),
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

    # 凡例（右下のグラフのみ）
    if i == 3:
        ax.legend(fontsize=20, loc="lower right")
    else:
        ax.legend().set_visible(False)

plt.tight_layout()

# y軸ラベル（回転）
plt.text(
    s="予測精度",
    x=-14.8,
    y=1,
    fontsize=42,
    fontweight="bold",
    va="center",
    rotation=90,
)

# タイトル
plt.text(
    x=-4.2,
    y=2.46,
    s="CLIPとDCNNの比較",
    fontsize=42,
    fontweight="bold",
    va="center",
)

save_path = os.path.join(output_dir, "encoding_lineplot_clip_vs_convnext.png")
fig.savefig(save_path, bbox_inches="tight")
print(f"  ✓ 保存: {os.path.basename(save_path)}")
plt.close()

# ============================================================================
# 3.5. Pretrained vs Finetuned ConvNeXt エンコーディング比較
# ============================================================================
print("\n3.5. Pretrained vs Finetuned ConvNeXt比較グラフを生成中...")

if "ConvNeXt" in encoding_results and "ConvNeXt_Finetuned" in encoding_results:
    # 両モデルのデータを準備
    ft_model_data = {}
    for model_key in ["ConvNeXt", "ConvNeXt_Finetuned"]:
        layer_results = encoding_results[model_key]
        plot_data = []
        for layer_idx, (layer_name, target_results) in enumerate(layer_results.items()):
            for target_name, score in target_results.items():
                plot_data.append({
                    'Layer': layer_idx,
                    'target': target_name,
                    'Score': score
                })

        melt_df = pd.DataFrame(plot_data)

        # 各カテゴリの3層移動平均を計算
        df_list = []
        for key, attrs in group_dict.items():
            filtered = melt_df[melt_df["target"].isin(attrs)]
            if len(filtered) == 0:
                continue
            attr_df = pd.DataFrame(
                filtered
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

        ft_model_data[model_key] = pd.concat(df_list)

    # 2x2グリッドでプロット（PretrainedとFinetunedを重ねる）
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)

    for i, attr in enumerate(category_order):
        ax = axes.flatten()[i]

        # Pretrained（実線）
        d_pretrained = ft_model_data["ConvNeXt"][ft_model_data["ConvNeXt"]["attr"] == attr]
        sns.lineplot(
            data=d_pretrained.reset_index(),
            x="index",
            y="Score",
            color=colors[i],
            marker="o",
            markersize=12,
            linewidth=3,
            ax=ax,
            label="事前学習のみ"
        )

        # Finetuned（破線）
        d_finetuned = ft_model_data["ConvNeXt_Finetuned"][ft_model_data["ConvNeXt_Finetuned"]["attr"] == attr]
        sns.lineplot(
            data=d_finetuned.reset_index(),
            x="index",
            y="Score",
            color=colors[i],
            marker="s",
            markersize=12,
            linewidth=3,
            linestyle="--",
            ax=ax,
            label="ファインチューニング"
        )

        ax.set_ylim(0.1, 1)

        if i > 1:
            ax.set_xlabel("層（3層平均）", fontsize=36, fontweight="bold")
            ax.set_xticklabels(
                range(0, 13, 2),
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

        # 凡例（右下のグラフのみ）
        if i == 3:
            ax.legend(fontsize=20, loc="lower right")
        else:
            ax.legend().set_visible(False)

    plt.tight_layout()

    # y軸ラベル（回転）
    plt.text(
        s="予測精度",
        x=-14.8,
        y=1,
        fontsize=42,
        fontweight="bold",
        va="center",
        rotation=90,
    )

    save_path = os.path.join(output_dir, "encoding_lineplot_pretrained_vs_finetuned.png")
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  ✓ 保存: {os.path.basename(save_path)}")
    plt.close()
else:
    print("  ⚠ ConvNeXt_Finetuned のデータがありません")

# ============================================================================
# 4. ファインチューニングあり/なしの比較バーチャート
# ============================================================================
print("\n4. ファインチューニング比較バーチャートを生成中...")

# ファインチューニング結果を読み込み
finetuned_corr = {}
if os.path.exists(finetuned_v9_path):
    with open(finetuned_v9_path, 'rb') as f:
        v9_data = dill.load(f)
    for model, corr in v9_data['avg_val_corr'].items():
        if not pd.isna(corr):
            finetuned_corr[model] = float(corr)

if os.path.exists(finetuned_v10_path):
    with open(finetuned_v10_path, 'rb') as f:
        v10_data = dill.load(f)
    for model, corr in v10_data['avg_val_corr'].items():
        if not pd.isna(corr):
            finetuned_corr[model] = float(corr)

# Pretrained結果（CLIPを含む）
pretrained_corr = {
    "CLIP": cv_results["CLIP_ConvNeXt_Base"]["res_L"]["overall_score"],
    "ConvNeXt": cv_results["ConvNeXt_Pretrained"]["res_L"]["overall_score"],
    "ResNet": cv_results["ResNet152_Pretrained"]["res_L"]["overall_score"],
    "VGG": cv_results["VGG16_Pretrained"]["res_L"]["overall_score"],
}

# ファインチューニング結果のキー名を変換
finetuned_display = {
    "CLIP": None,  # CLIPはファインチューニングなし
    "ConvNeXt": finetuned_corr.get("convnext_base"),
    "ResNet": finetuned_corr.get("resnet152"),
    "VGG": finetuned_corr.get("vgg16"),
}

# データフレーム作成
plot_data = []
model_order = ["CLIP", "ConvNeXt", "ResNet", "VGG"]

for model in model_order:
    # Pretrained
    plot_data.append({
        "モデル": model,
        "条件": "事前学習のみ",
        "相関係数": pretrained_corr[model]
    })
    # Finetuned（CLIPはなし）
    if finetuned_display[model] is not None:
        plot_data.append({
            "モデル": model,
            "条件": "ファインチューニング",
            "相関係数": finetuned_display[model]
        })

df_compare = pd.DataFrame(plot_data)

# プロット作成
fig = plt.figure(figsize=(16, 9), dpi=150)

# Seaborn mutedパレット
palette = {"事前学習のみ": "#4878CF", "ファインチューニング": "#6ACC65"}

ax = sns.barplot(
    data=df_compare,
    x="モデル",
    y="相関係数",
    hue="条件",
    palette=palette,
    order=model_order,
    width=0.6
)

plt.xticks(fontsize=42, fontweight="bold")
plt.xlabel("", fontsize=36)
plt.yticks(fontsize=46, fontweight="bold")
plt.ylabel("相関係数", fontsize=46, labelpad=20, fontweight="bold")
plt.ylim(0.0, 1.0)
plt.legend(fontsize=24, loc="upper right")

plt.tight_layout()

save_path = os.path.join(output_dir, "cv_barplot_pretrained_vs_finetuned.png")
fig.savefig(save_path, bbox_inches="tight")
print(f"  ✓ 保存: {os.path.basename(save_path)}")
plt.close()

print("\n" + "="*80)
print("可視化完了！")
print("="*80)
print(f"\n生成されたグラフ: {output_dir}/")
print("\n生成されたファイル:")
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.png'):
        print(f"  - {f}")
