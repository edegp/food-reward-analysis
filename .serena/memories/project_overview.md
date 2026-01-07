# Food Reward Analysis プロジェクト概要

## 統合リポジトリ（2026-01-07作成）

旧リポジトリを統合:
- `foodReward` → `scripts/dnn_model/` (DNN学習)
- `food-brain` → その他すべて (fMRI分析)

### 主要分析

1. **DNN Model Training**: 食品報酬予測モデル学習（ConvNeXt, CLIP LoRA）
2. **Behavior GLM**: 食品の主観的価値に関連する脳活動
3. **DNN GLM (Hierarchical)**: 3階層PC（Global, Shared, Layer-Specific）による脳活動モデリング
4. **RSA**: ROIベースの表象類似性分析（Double-Centering / CKA相当）
5. **ROI分析**: Harvard-Oxford 31領域でのSVC解析

### スクリプト構成

#### scripts/dnn_model/ (旧foodReward)
- `src/` - モデル・学習コード
  - `model.py` - ConvNeXt, CLIP LoRA等
  - `train.py` - 学習ユーティリティ
  - `dataset.py` - データローダー
- `notebooks/CV/` - Cross-validation学習
  - `TrainFoodRewardCV.py` - メインCV
  - `TrainCLIPLoRA.py` - CLIP fine-tuning
- `notebooks/analysis/` - 分析・可視化

#### scripts/dnn_analysis/
- `hierarchical/preprocess/` - PC抽出パイプライン
  - `extract_global_pcs.py` - Step 1: Global PC抽出
  - `extract_layer_pcs.py` - Step 2: Layer PC抽出
  - `create_pcs.py` - Step 3: 3階層PC作成
- `hierarchical/first_level/` - GLM実行
- `hierarchical/second_level/` - グループ解析
- `hierarchical/visualization/` - 結果可視化
- `roi/` - ROI分析
  - `create_harvard_oxford_rois.py` - ROIマスク作成
  - `run_svc_hierarchical.m` - SVC計算
  - `visualize_beta_rms.py` - バープロット
- `rsa/` - RSA分析
  - `calc_rsa_roi_centered.py` - RSA計算
  - `visualize_roi_centered.py` - RSA可視化
- `layerwise/` - 層別分析（将来用）
- `common/` - 共通ユーティリティ

#### scripts/behavior_glm/
- `main/` - First-level GLM
- `lss_model/` - LSS GLM（トライアル単位β推定）
- `roi/` - ROI分析
- `visualization/` - 可視化

### 主要Taskfileタスク

```bash
# DNN学習
python scripts/dnn_model/notebooks/CV/TrainFoodRewardCV.py

# 前処理
task hierarchical_preprocess_all  # PC抽出全パイプライン

# GLM
task hierarchical_glm            # First-level
task hierarchical_second_level_v4  # Second-level

# ROI
task create_harvard_oxford_rois  # ROIマスク作成
task dnn_roi_svc                 # SVC計算
task dnn_roi_viz                 # 可視化

# RSA
task rsa_all                     # RSA一括実行

# 全パイプライン
task hierarchical_full_pipeline
```

### 論文画像（paper/image/）
- `cv_barplot_pretrained_vs_finetuned.png` - DNN CV結果
- `encoding_lineplot_*.png` - エンコーディング分析
- `glm_rgb_nutri_ImagexValue.png` - Behavior GLM
- `roi_effectsize_rgb_nutri_ImagexValue.png` - ROI効果量
- `roi_rms_barplot*.png` - DNN ROI効果量
- `clip/convnext_hierarchical_fwe_p0.05.png` - DNN GLM
- `rsa_roi_centered_*.png` - RSA結果

### 外部ストレージ
- `/Volumes/Extreme Pro/hit/food-brain/results/` - 大容量結果ファイル
  - `first_level_analysis/` - LSS β値
  - `rsa_analysis/roi_centered/` - RSA結果
