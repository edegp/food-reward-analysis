# Food Reward Analysis

食品報酬予測のためのDNNモデル学習と、食品画像に対する脳活動のfMRI分析を統合したプロジェクト

## 概要

このプロジェクトは、食品画像から主観的報酬価値を予測するDNNモデルの学習と、そのモデル表現と脳活動の対応関係を分析します。

### 主な分析

1. **DNN Model Training**: 食品報酬予測DNNモデルの学習（ConvNeXt, CLIP LoRA）
2. **Behavior GLM**: 食品の主観的価値（欲しさ評定）に関連する脳活動
3. **DNN GLM**: 深層学習モデルの特徴量に関連する脳活動
4. **RSA**: 表象類似性分析によるDNN-脳対応の検証

## クイックスタート

```bash
# 1. Python環境セットアップ（ルートから）
uv sync

# 2. 全タスク一覧表示
task --list

# 3. DNN学習（CV）
python scripts/dnn_model/notebooks/CV/TrainFoodRewardCV.py

# 4. fMRI分析（Hierarchical DNN GLM）
task hierarchical_full_pipeline
```

## 環境設定

### 必要なソフトウェア
| ソフトウェア | バージョン | 用途 |
|-------------|-----------|------|
| MATLAB | R2024b以降 | SPMによるGLM解析 |
| SPM25 | 最新版 | fMRI統計解析 |
| Python | 3.12+ | DNN学習・可視化・RSA分析 |
| uv | 最新版 | Pythonパッケージ管理 |
| Task | 最新版 | タスクランナー |
| CUDA | 13.0+ | GPU学習（オプション）|

### インストール
```bash
# Task (macOS)
brew install go-task/tap/go-task

# uv (Python)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Python依存関係
uv sync

# GPU学習用（オプション）
uv sync --group training
```

### SPM25設定
MATLABパスにSPM25を追加：
```matlab
addpath('/path/to/spm25');
```

## DNN Model Training

食品報酬予測のためのDNNモデル学習。

### モデルアーキテクチャ
| モデル | 説明 |
|--------|------|
| ConvNeXt Base | 高精度CNN、食品嗜好回帰でファインチューニング |
| ConvNeXt Tiny | 軽量版、高速推論用 |
| CLIP ViT-L/14 | Vision-Language Model、LoRAでファインチューニング |
| VGG16/ResNet | ベースラインモデル |

### 学習スクリプト
```bash
# Cross-validation training
python scripts/dnn_model/notebooks/CV/TrainFoodRewardCV.py

# CLIP LoRA fine-tuning
python scripts/dnn_model/notebooks/CV/TrainCLIPLoRA.py
```

### モデルコード
```
scripts/dnn_model/
├── src/
│   ├── model.py      # モデルアーキテクチャ
│   ├── train.py      # 学習ユーティリティ
│   ├── dataset.py    # データローダー
│   ├── analysis.py   # 分析ユーティリティ
│   └── const.py      # 設定・定数
└── notebooks/
    ├── CV/           # Cross-validation scripts
    ├── analysis/     # 結果分析notebooks
    └── EDA/          # 探索的データ分析
```

## fMRI Analysis

### データ要件

```
food-reward-analysis/
├── Database/                    # 食品画像 (896枚)
├── Food_Behavior/               # 行動データ
│   └── sub-XXX/
│       └── rating_data*.csv     # 評定データ・タイミング情報
├── fMRIprep/
│   ├── derivatives/             # fMRIprep出力
│   └── smoothed/                # 平滑化済み (6mm FWHM)
│       └── sub-XXX/func/*.nii
├── DNNs_model/                  # DNNモデル（学習済み）
│   └── v9/res_L/
│       └── convnext_base_regression.pth
└── data_images/
    └── dnn_pmods/               # DNN特徴量（事前抽出済み）
```

### 被験者情報
- **総被験者数**: 31名 (sub-001 〜 sub-031)
- **解析対象**: 20名
- **除外被験者**: sub-001, sub-004, sub-012, sub-017, sub-021, sub-026〜031
  - 除外理由: 頭部動き過大、データ欠損、タスク遂行不良等

### 分析パイプライン

```
1. 前処理（既に完了している場合はスキップ）
   └── fMRIprep → smoothing

2. DNN特徴量抽出
   └── dnn_features

3. Behavior GLM
   └── fmri_analysis → behavior_cluster_fwe_compute → behavior_cluster_fwe_viz

4. LSS GLM（RSA用）
   └── lss_glm

5. Hierarchical DNN GLM
   └── hierarchical_preprocess_all → hierarchical_glm → hierarchical_second_level_v4
   └── dnn_cluster_fwe_compute → cluster_fwe_viz

6. ROI分析
   └── create_harvard_oxford_rois → dnn_roi_svc → dnn_roi_viz

7. RSA分析
   └── rsa_roi_centered → rsa_roi_centered_viz
```

### 1. Behavior GLM (食品価値分析)

主観的な食品の価値（欲しさ）に関連する脳活動を分析。

```bash
task fmri_analysis              # First-level GLM
task behavior_cluster_fwe_compute  # Cluster FWE
task behavior_cluster_fwe_viz      # 可視化
```

**主要コントラスト:** `ImagexValue` (画像提示時の価値パラメトリック効果)

**期待される結果:** vmPFC, OFC, 線条体での価値表現

### 2. DNN GLM (階層的DNN分析)

DNNの層グループ（初期層・中間層・後期層・最終層）の共有成分と固有成分を分離して脳活動との関連を分析。

```bash
task hierarchical_preprocess_all   # 前処理（PCA）
task hierarchical_glm              # First-level GLM
task hierarchical_second_level_v4  # Second-level GLM
task dnn_cluster_fwe_compute       # Cluster FWE
task cluster_fwe_viz               # 可視化
```

**主要コントラスト:**
- `Initial_only`, `Middle_only`, `Late_only`, `Final_only`: 層固有成分
- `Shared_*`: 隣接層間の共有成分
- `Global`: 全層共通成分

### 3. ROI分析

Harvard-Oxfordアトラスを用いたROI分析（31領域）。

```bash
task create_harvard_oxford_rois  # ROIマスク作成
task dnn_roi_svc                 # Small Volume Correction
task dnn_roi_viz                 # 可視化
```

### 4. RSA (表象類似性分析)

ROIベースのRSA解析（Double-Centering / CKA相当）。

```bash
task rsa_roi_centered      # RSA計算
task rsa_roi_centered_viz  # 可視化
```

**比較モデル:**
| モデル | 説明 |
|--------|------|
| CLIP | OpenAI CLIP ViT-L/14 (事前学習済み) |
| ImageNet | ConvNeXt-Base (ImageNet事前学習) |
| Food | ConvNeXt-Base (食品嗜好でファインチューニング) |

## ディレクトリ構成

```
food-reward-analysis/
├── scripts/
│   ├── dnn_model/          # DNN学習（旧foodReward）
│   │   ├── src/            # model.py, train.py, etc.
│   │   └── notebooks/      # CV/, analysis/
│   ├── dnn_analysis/       # DNN-fMRI分析
│   ├── behavior_glm/       # Behavior GLMスクリプト
│   ├── common/             # 共通スクリプト
│   └── preprocess/         # 前処理
├── Database/               # 食品画像 (896枚)
├── Food_Behavior/          # 行動データ（評定・タイミング）
├── fMRIprep/               # 前処理済みfMRIデータ
├── results/                # 分析結果
├── rois/                   # ROIマスク (Harvard-Oxford)
├── DNNs_model/             # DNNモデルチェックポイント
├── paper/                  # 論文関連
│   ├── main.typ            # 本文 (Typst)
│   └── image/              # 図表出力先
├── pyproject.toml          # Python依存関係（統合）
├── Taskfile.yml            # タスク定義
└── README.md               # このファイル
```

## 主要タスク一覧

```bash
task --list  # 全タスク表示
```

### DNN学習
| タスク | 説明 |
|--------|------|
| `dnn_train_cv` | Cross-validation学習 |
| `dnn_features` | DNN特徴量抽出 |

### Behavior GLM
| タスク | 説明 |
|--------|------|
| `fmri_analysis` | First-level GLM |
| `lss_glm` | LSS GLM (試行別β推定) |
| `behavior_cluster_fwe_compute` | クラスターFWE計算 |
| `behavior_cluster_fwe_viz` | クラスターFWE可視化 |

### DNN GLM (Hierarchical)
| タスク | 説明 |
|--------|------|
| `hierarchical_preprocess_all` | 前処理一括実行 |
| `hierarchical_glm` | First-level GLM |
| `hierarchical_second_level_v4` | Second-level GLM |
| `dnn_cluster_fwe_compute` | クラスターFWE計算 |
| `hierarchical_full_pipeline` | 全パイプライン |

### ROI分析
| タスク | 説明 |
|--------|------|
| `create_harvard_oxford_rois` | ROIマスク作成 |
| `dnn_roi_svc` | Small Volume Correction |
| `dnn_roi_viz` | ROIバープロット可視化 |

### RSA
| タスク | 説明 |
|--------|------|
| `rsa_roi_centered` | ROI RSA計算 |
| `rsa_roi_centered_viz` | RSA可視化 |
| `rsa_all` | RSA一括実行 |

## 論文出力

主要な図表は `paper/image/` に出力されます。

## トラブルシューティング

### MATLABでSPMが見つからない
```matlab
addpath('/path/to/spm25');
spm('defaults', 'fmri');
```

### Python環境エラー
```bash
uv sync --force
```

### タスクが見つからない
```bash
# プロジェクトルートで実行
task --list
```

## 参考文献

- SPM: https://www.fil.ion.ucl.ac.uk/spm/
- fMRIPrep: https://fmriprep.org/
- nilearn: https://nilearn.github.io/
- CLIP: https://github.com/openai/CLIP
- ConvNeXt: https://github.com/facebookresearch/ConvNeXt

## License

MIT License
