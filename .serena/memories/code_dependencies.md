# Code Dependencies Map

## プロジェクト概要

統合リポジトリ `food-reward-analysis` は以下を含む：
- **DNN Model Training** (旧foodReward): 食品報酬予測モデル学習
- **fMRI Analysis** (旧food-brain): 脳活動分析

## ディレクトリ構造

```
scripts/
├── dnn_model/              # DNN学習 (旧foodReward)
│   ├── src/                # モデル・学習コード
│   └── notebooks/          # CV, analysis, EDA
├── dnn_analysis/           # DNN-fMRI分析
│   ├── config/             # 設定ファイル
│   ├── common/             # 共通ユーティリティ
│   ├── hierarchical/       # 階層的DNN GLM (メイン分析)
│   ├── layerwise/          # 層別DNN GLM (将来用)
│   ├── roi/                # ROI分析
│   └── rsa/                # RSA分析
├── behavior_glm/           # 行動GLM分析
└── common/                 # 共通スクリプト
```

---

## 0. DNN Model Training パイプライン

### 0.1 モデルアーキテクチャ

```
scripts/dnn_model/src/
├── model.py          # モデル定義
│   ├── ConvNeXt (Base/Tiny) - 回帰ヘッド付き
│   ├── VGG16/ResNet - ベースライン
│   └── CLIP LoRA - Vision-Language fine-tuning
├── dataset.py        # データローダー
│   └── FoodImageDataset - 食品画像 + 主観評定
├── train.py          # 学習ユーティリティ
│   ├── train_epoch() - 1エポック学習
│   ├── evaluate() - 評価
│   └── EarlyStopping - 早期終了
├── analysis.py       # 分析ユーティリティ
│   └── 特徴量抽出、可視化
├── eda.py            # 探索的データ分析
└── const.py          # 定数・設定
    ├── SUBJECTS - 被験者リスト
    ├── IMAGE_SIZE - 224
    └── モデルパス等
```

### 0.2 Cross-Validation 学習

```
scripts/dnn_model/notebooks/CV/
├── TrainFoodRewardCV.py      # メインCV学習スクリプト
│   入力: Database/ (食品画像)
│   入力: Food_Behavior/*/rating_data*.csv (主観評定)
│   出力: DNNs_model/v*/res_L/convnext_base_regression.pth
│   処理: Leave-one-subject-out CV、ConvNeXt fine-tuning
│
├── TrainCLIPLoRA.py          # CLIP LoRA fine-tuning
│   処理: CLIP ViT-L/14 + LoRA adapter学習
│
├── TrainFoodRewardCVByBMI.py # BMI別CV
└── brightnessCV.py           # 輝度ベースライン
```

### 0.3 分析・可視化

```
scripts/dnn_model/notebooks/analysis/
├── unified_cv_analysis.py       # CV結果統合分析
│   出力: paper/image/cv_barplot_pretrained_vs_finetuned.png
│
├── unified_encoding_analysis.py # エンコーディング分析
│   出力: paper/image/encoding_lineplot_*.png
│
├── RSA_and_regression_v16.ipynb # RSA + 回帰分析
├── clip_v2.ipynb                # CLIP分析
└── comparison_clip_dcnn.ipynb   # CLIP vs CNN比較
```

### 0.4 DNN特徴量抽出

```
[Task: dnn_features]
scripts/extract_dnn_features.py
  入力: Database/ (食品画像)
  入力: DNNs_model/v*/res_L/*.pth (学習済みモデル)
  出力: data_images/dnn_features/*.csv (層別特徴量)
  処理: 各層の活性化を抽出、CSV保存
```

---

## 1. Hierarchical GLM パイプライン

### 1.1 前処理 (PC抽出)

```
[Task: hierarchical_extract_global_pcs]
extract_global_pcs.py
  入力: data_images/dnn_features/*.csv (DNN特徴量)
  出力: data_images/dnn_pmods/global/*.csv (全層共通PC)
  処理: 全層を結合してPCA、共通成分を抽出

[Task: hierarchical_extract_layer_pcs]
extract_layer_pcs.py
  入力: data_images/dnn_features/*.csv, global PCs
  出力: data_images/dnn_pmods/layer/*.csv (層グループPC)
  処理: Global成分を除去後、各層グループでPCA

[Task: hierarchical_create_pcs]
create_pcs.py
  入力: global PCs, layer PCs
  出力: data_images/dnn_pmods/3level/*_3level_pcs.csv
  処理: Global + Shared + Layer-Specific の3階層PC作成
```

### 1.2 First-level GLM

```
[Task: hierarchical_glm]
run_all_subjects.sh
  └── run_glm.m (各被験者に対して実行)
        ├── 入力: fMRIprep/smoothed/sub-*/func/*.nii (平滑化fMRI)
        ├── 入力: data_images/dnn_pmods/3level/*_3level_pcs.csv
        ├── 入力: Food_Behavior/sub-*/rating_data*.csv (行動データ)
        ├── 出力: results/first_level_analysis/sub-*/glm_model/glm_dnn_pmods_designN_*/
        └── build_contrasts_sessions.m (コントラスト定義)
              出力: SPM.mat内にF/Tコントラスト追加

[Task: hierarchical_glm_daywise] (代替デザイン)
run_all_daywise.sh
  └── run_glm_daywise.m
        └── build_contrasts_daywise.m
```

### 1.3 コントラスト再構築

```
[Task: hierarchical_rerun_contrasts]
rerun_all_contrasts.m
  └── rerun_hierarchical_contrasts.m (各被験者)
        └── build_contrasts.m
              処理: 既存SPM.matにコントラストを追加/更新
```

### 1.4 Second-level GLM

```
[Task: hierarchical_second_level_v4]
create_second_level_hierarchical.m
  入力: results/first_level_analysis/sub-*/glm_model/*/con_*.nii
  出力: results/dnn_analysis/second_level/hierarchical_*/
  処理: One-sample t-test (グループ解析)
```

### 1.5 Cluster FWE補正

```
[Task: dnn_cluster_fwe_compute]
common/utils/run_cluster_fwe.m
  入力: results/dnn_analysis/second_level/hierarchical_*/SPM.mat
  出力: results/dnn_analysis/cluster_fwe_v2/hierarchical_v3_*/*_clusterFWE.nii
  処理: SPM RFTによるクラスターレベルFWE補正

[Task: cluster_fwe_viz]
common/visualization/visualize_cluster_fwe.py
  入力: results/dnn_analysis/cluster_fwe_v2/hierarchical_v3_*/*_clusterFWE.nii
  出力: paper/image/*_hierarchical_fwe_p0.05.png
  処理: Glass brain + スライス可視化
```

### 1.6 可視化

```
[Task: hierarchical_viz / hierarchical_viz_shared / hierarchical_viz_global]
hierarchical/visualization/visualize_results.py
  入力: results/dnn_analysis/second_level/hierarchical_*/
  出力: paper/image/clip_hierarchical_*.png, convnext_hierarchical_*.png
  オプション: --include-shared, --include-global
```

---

## 2. ROI分析パイプライン

```
[Task: create_harvard_oxford_rois]
roi/create_harvard_oxford_rois.py
  入力: Harvard-Oxfordアトラス (nilearn)
  出力: rois/HarvardOxford/*_mask.nii (31 ROI)
  ROI一覧:
    Visual: V1, EarlyVisual, LOC, Fusiform, IT
    Parietal: IPL, SPL, AngularGyrus, Precuneus
    Temporal: STG, MTG, TemporalPole, PHC
    Frontal: IFG, DLPFC, VLPFC, OFC, FrontalPole, Broca_L, vmPFC
    Cingulate: ACC, PCC
    Insula: Insula
    Subcortical: Hippocampus, Amygdala, Caudate, Putamen, NAcc, Thalamus, Pallidum, Striatum

[Task: dnn_roi_svc]
roi/run_svc_hierarchical.m
  入力: results/dnn_analysis/second_level/hierarchical_*/SPM.mat
  入力: rois/HarvardOxford/*_mask.nii
  出力: results/dnn_analysis/roi_analysis/hierarchical_svc/*.csv
  処理: Small Volume Correction (ROI内FWE補正)

[Task: dnn_roi_eta2p_ci]
roi/calculate_eta2p_ci.py
  入力: results/dnn_analysis/roi_analysis/hierarchical_svc/*.csv
  出力: results/dnn_analysis/roi_analysis/eta2p_ci/*.csv
  処理: η²p効果量 + 95%信頼区間計算

[Task: dnn_roi_beta_rms]
roi/extract_beta_rms.m
  入力: results/first_level_analysis/sub-*/glm_model/*/beta_*.nii
  入力: rois/HarvardOxford/*_mask.nii
  出力: results/dnn_analysis/roi_analysis/beta_rms/*.csv
  処理: ROI内βのRMS抽出

[Task: dnn_roi_viz]
roi/visualize_beta_rms.py
  入力: results/dnn_analysis/roi_analysis/beta_rms/*.csv
  出力: paper/image/roi_rms_barplot*.png
  処理: バープロット作成
```

---

## 3. RSA分析パイプライン

```
[Task: rsa_roi_centered]
rsa/calc_rsa_roi_centered.py
  入力: results/behavior_glm/lss/sub-*/beta_*.nii (LSS beta)
  入力: rois/HarvardOxford/*_mask.nii
  入力: data_images/dnn_features/*.csv (DNN特徴量)
  出力: /Volumes/Extreme Pro/hit/food-brain/results/rsa_analysis/roi_centered/
         ├── roi_centered_summary.csv
         └── roi_centered_details.csv
  処理: Double-centering RSA (CKA相当) + Noise Ceiling計算
  比較モデル:
    - CLIP: OpenAI CLIP ViT-L/14
    - ImageNet: ConvNeXt pretrained
    - Food: ConvNeXt fine-tuned on food preference

[Task: rsa_roi_centered_viz]
rsa/visualize_roi_centered.py
  入力: results/rsa_analysis/roi_centered/roi_centered_summary.csv
  出力: paper/image/rsa_roi_centered_comparison_jp.png (全ROI比較)
         paper/image/rsa_roi_centered_summary_jp.png (モデル比較サマリー)
```

---

## 4. Behavior GLM パイプライン

```
[Task: fmri_analysis]
behavior_glm/main/run_first_level.m
  入力: fMRIprep/smoothed/sub-*/func/*.nii
  入力: Food_Behavior/sub-*/rating_data*.csv
  出力: results/first_level_analysis/sub-*/glm_model/glm_*/
  処理: 主観的価値のパラメトリックGLM

[Task: lss_glm]
behavior_glm/lss_model/run_lss_glm.m
  入力: fMRIprep/smoothed/sub-*/func/*.nii
  出力: results/behavior_glm/lss/sub-*/beta_*.nii
  処理: Least Squares Separate (試行ごとβ推定)

[Task: behavior_cluster_fwe_compute]
behavior_glm/visualization/run_cluster_fwe.m
  入力: results/second_level_analysis/*/SPM.mat
  出力: results/behavior_glm/cluster_fwe/*/

[Task: behavior_cluster_fwe_viz]
behavior_glm/visualization/visualize_cluster_fwe.py
  入力: results/behavior_glm/cluster_fwe/*
  出力: paper/image/glm_rgb_nutri_ImagexValue.png
```

---

## 5. Layerwise GLM (将来用)

```
[Task: layerwise_glm]
layerwise/first_level/run_all_subjects.m
  └── run_glm.m
        └── build_contrasts.m

[Task: layerwise_second_level]
layerwise/second_level/create_second_level.m

[Task: layerwise_viz]
layerwise/visualization/visualize_results.py
```

---

## 6. 共通ユーティリティ

### Config (設定読み込み)
```
config/layer_groups.json - レイヤーグループ定義
config/load_layer_groups.py - Python用読み込み関数
config/load_layer_groups.m - MATLAB用読み込み関数
```

### Common/preprocess
```
common/preprocess/collect_images.py
  出力: data_images/used_image_ids.txt (568画像)
  処理: 実験で使用された画像IDを収集（一度実行済み）
```

### Common/utils
```
common/utils/run_cluster_fwe.m - DNN用クラスターFWE計算
common/utils/extract_cluster_fwe_results.m - 結果抽出（未使用）
```

---

## 7. 削除済みファイル（参考）

| ファイル | 理由 |
|---------|------|
| common/visualization/visualize_*.py (8個) | 未使用の可視化スクリプト |
| mvpa/ (13ファイル) | MVPA分析（Taskfileにコメントで構想残し）|
| hierarchical/first_level/run_glm_hybrid.m | 実験的デザイン |
| hierarchical/first_level/run_daywise_clip.sh | 一時的な再開用スクリプト |
| hierarchical/first_level/run_daywise_convnext.sh | 同上 |
| hierarchical/preprocess/extract_pcs.py | 3ファイルに分割済み |
| roi/create_aal2_rois.py | Harvard-Oxford使用のため不要 |
| roi/run_svc_hierarchical_subjectSE.m | 別バージョンSVC |
| roi/visualize_svc_results.py | 未使用 |
| roi/roi_effectsize_hierarchical.py | 未使用 |
