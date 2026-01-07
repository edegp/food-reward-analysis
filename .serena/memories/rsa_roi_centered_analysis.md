# RSA ROI-Centered Analysis

## 概要
ROIベースのRSA解析（Double-Centering / CKA相当）

## 手法
- **Double-Centering**: Pearson相関をCKA（Centered Kernel Alignment）と等価にする
- **Noise Ceiling**: Leave-One-Out法で算出
- **ROI**: Harvard-Oxford Atlas (31領域)

## 3モデル比較
1. **CLIP**: OpenAI CLIP ViT-L/14 (visual_trunk_stages_*)
2. **ImageNet**: ConvNeXt pretrained on ImageNet
3. **Food**: ConvNeXt fine-tuned on food preference

## ROI一覧 (31領域)
### Visual (5)
V1, EarlyVisual, LOC, Fusiform, IT

### Parietal (4)
IPL, SPL, AngularGyrus, Precuneus

### Temporal (4)
STG, MTG, TemporalPole, PHC

### Frontal (7)
IFG, DLPFC, VLPFC, OFC, FrontalPole, Broca_L, vmPFC

### Cingulate (2)
ACC, PCC

### Insula (1)
Insula

### Subcortical (8)
Hippocampus, Amygdala, Caudate, Putamen, NAcc, Thalamus, Pallidum, Striatum

## スクリプト
- `scripts/dnn_analysis/rsa/calc_rsa_roi_centered.py` - RSA計算
- `scripts/dnn_analysis/rsa/visualize_roi_centered.py` - 可視化

## 出力
### データファイル (外付けドライブ)
- `/Volumes/Extreme Pro/hit/food-brain/results/rsa_analysis/roi_centered/roi_centered_results.json`
- `/Volumes/Extreme Pro/hit/food-brain/results/rsa_analysis/roi_centered/roi_centered_summary.csv`

### 論文画像
- `paper/image/rsa_roi_centered_comparison_jp.png` - 全31 ROIのNC%比較
- `paper/image/rsa_roi_centered_summary_jp.png` - モデル間比較サマリー

## Taskfileタスク
```bash
task rsa_roi_centered      # RSA計算
task rsa_roi_centered_viz  # 可視化
task rsa_all               # 両方実行
```

## 結果概要
視覚野（V1, EarlyVisual）で最も高いNC%（60-70%）、非視覚野では低い（2-15%）。
CLIPが非視覚野で最も高い説明率を示す傾向。
