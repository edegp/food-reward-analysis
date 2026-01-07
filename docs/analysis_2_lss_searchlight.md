# 分析2：LSS GLM + Searchlight Encoding Model

## 概要
試行レベルの脳活動パターンとDNN特徴量を対応付け、各脳領域で各DNN層がどれだけ脳活動を予測できるかを調べる多変量パターン解析（MVPA）。

---

## データ
- **被験者**: 20名
- **刺激**: 食品画像（896枚）
- **fMRIデータ**: 各被験者約560試行（画像提示）
- **DNNモデル**:
  - CLIP (10層 → 4層グループ: Initial/Middle/Late/Final)
  - ConvNeXt (10層 → 4層グループ: Initial/Middle/Late/Final)

---

## 分析の流れ

```
【準備】
食品画像（896枚）
    ↓
DNN特徴抽出（全チャネル、PCは使わない）
    ↓
各層の生の特徴量

【First-Level: LSS GLM】
fMRI収集（各被験者約560試行）
    ↓
各画像提示ごとに独立したbeta値を推定
    ↓
画像×ボクセルのbeta値行列（20名分）

【MVPA: Searchlight分析】
各ボクセル周辺の小領域（searchlight）で
    ↓
DNN特徴量 → 脳活動パターンの予測モデル
    ↓
予測精度（R²）マップ
    ↓
層ごとのR²マップ + Winner-take-all マップ
```

---

## Step 1: LSS GLM（Least Squares Separate）

### 目的
各画像提示に対する個別のbeta値（脳活動パターン）を推定する。

### 従来のGLMとの違い

**従来のGLM**:
```
条件Aの全試行を平均 → 条件Aのbeta値
条件Bの全試行を平均 → 条件Bのbeta値
```
→ 条件ごとの平均しか得られない

**LSS GLM**:
```
画像1の試行 → 画像1のbeta値
画像2の試行 → 画像2のbeta値
...
画像560の試行 → 画像560のbeta値
```
→ 各試行（各画像）ごとのbeta値が得られる

### 手法

**デザインマトリクス**:
- 各画像提示を独立した説明変数としてモデル化
- 約560個の説明変数（画像数分）
- 共変量: 運動パラメータ（6自由度）

**推定**:
- 標準的な最小二乗法
- 各画像に対するbeta値を推定

### 利点

1. **試行レベルの分解能**
   - 条件平均ではなく、個別画像の効果を推定

2. **DNN特徴量との直接対応**
   - 各画像のbeta値 ↔ その画像のDNN特徴量

3. **MVPAに最適**
   - 多変量パターン（複数ボクセルのパターン）を利用可能
   - 予測モデルの構築が可能

### 出力
- 各被験者: 約560個のbeta画像（beta_0001.nii, beta_0002.nii, ...）
- 各beta画像は1つの画像提示に対応

**実装**: `scripts/first_level_analysis/lss_model/run_lss_glm.m`

**現在の状況**: 🔄 実行中（6/20名完了）

---

## Step 2: Searchlight Encoding Model

### 目的
各脳領域で、各DNN層がどれだけ脳活動を予測できるかを定量化する。

### Searchlightとは？

**概念**:
```
     ●  ← 中心ボクセル
   ● ● ●
  ● ● ● ●  ← 半径5mmの球（searchlight）
   ● ● ●
     ●
```

各ボクセルを中心とした小さな球状領域（searchlight）で独立に分析を行う。

**メリット**:
- 局所的な情報を保持
- 全脳をスキャン
- 多変量パターン（複数ボクセル）を考慮

### Encoding Modelとは？

**方向**:
```
DNN特徴量 → 脳活動パターン
  (説明変数)    (目的変数)
```

**質問**:
「このDNN層の表現は、この脳領域の活動をどれだけ説明できるか？」

### 手法の詳細

#### 入力データ
**X (説明変数)**: DNN特徴量
- 各層グループ（Initial/Middle/Late）の特徴量
- 全チャネル連結 → PCAで次元削減（分散70%保持）
- 標準化

**補足（Final層について）**:
- Final層グループ（ConvNeXtのclassifier、CLIPのattnpool）は全結合／グローバルプール後の単一層で、空間パターンの差分を持たない
- Searchlightは局所球内の多ボクセルパターン差を指標にするため、Final層を含めてもR²や分類精度が定義できない
- Final層の効果は分析1（GLM）や全脳F-mapで評価し、本SearchlightではInitial/Middle/Lateの3層グループに限定する

**y (目的変数)**: 脳活動パターン
- LSS GLMから得られた各画像のbeta値
- Searchlight領域内のボクセル活動の平均

#### 分析手順（各searchlight領域で）

1. **データ準備**
   ```
   画像1: DNN特徴 [f₁, f₂, ..., fₙ] → 脳活動 y₁
   画像2: DNN特徴 [f₁, f₂, ..., fₙ] → 脳活動 y₂
   ...
   画像560: DNN特徴 [f₁, f₂, ..., fₙ] → 脳活動 y₅₆₀
   ```

2. **予測モデルの構築**
   - Ridge回帰（正則化パラメータαは事前に最適化）
   - Cross-validation（12-fold）で予測

3. **評価**
   - R²（決定係数）: モデルがどれだけ分散を説明できるか
   - Pearson相関係数

4. **層グループごとに繰り返し**
   - Initial層グループのR²
   - Middle層グループのR²
   - Late層グループのR²

#### 全被験者データの統合

**アプローチ**: 被験者間でデータを統合
```
被験者1: 560試行
被験者2: 560試行
...
被験者20: 560試行
↓
統合: 11,200試行
```

**利点**:
- サンプルサイズ増加により予測精度向上
- 被験者間で一般化可能なパターンを抽出

### 出力

#### 1. 層グループごとのR²マップ
```
encoding_Initial_clip_r5mm_r2.nii.gz
encoding_Middle_clip_r5mm_r2.nii.gz
encoding_Late_clip_r5mm_r2.nii.gz
```

各ボクセルの値 = その層グループの予測R²

**解釈例**:
```
ボクセル (x, y, z) のMiddle層R² = 0.15
→ Middle層特徴量は、この領域の活動の15%を説明できる
```

#### 2. Winner-take-all マップ
```
encoding_winner_clip_r5mm.nii.gz
```

各ボクセルの値 = 最もR²が高い層グループのインデックス
- 0 = Initial
- 1 = Middle
- 2 = Late

**解釈**:
どの脳領域がどの層グループに対応するかの階層マップ

**実装**: `scripts/dnn_analysis/mvpa/encoding_searchlight_lss.py`

---

## 期待される結果

### 1. 階層的予測精度の勾配

**仮説**:
```
視覚野（V1/V2）   → Initial層のR²が高い
側頭葉（IT）      → Middle層のR²が高い
前頭葉・側頭前部  → Late層のR²が高い
```

### 2. Winner-take-all マップによる可視化

視覚的に明確な階層構造:
```
後頭葉 → 側頭葉後部 → 側頭葉前部 → 前頭葉
(Initial)  (Middle)    (Late)      (Late)
```

### 3. モデル間比較

**CLIP vs ConvNeXt**:
- 言語学習の効果が前頭葉のLate層表現に現れるか？
- 視覚のみ学習は視覚野により特化するか？

### 4. 定量的な比較

各層グループについて:
- 平均R²
- 最大R²
- 有意に予測できるボクセル数
- Winner割合（全ボクセルのうちその層が最良の割合）

---

## 分析1との違いと補完関係

### 分析1（パラメトリックGLM + Second-Level）

**アプローチ**: 単変量（各ボクセル独立）
**統計**: 被験者間での一貫性を検定
**出力**: 統計的有意性マップ（p値、Z値）
**強み**:
- 統計的に厳密（多重比較補正）
- 解釈が直感的
- 論文報告の標準

### 分析2（LSS GLM + Searchlight）

**アプローチ**: 多変量（複数ボクセルのパターン）
**統計**: 予測精度（R²）
**出力**: 予測精度マップ
**強み**:
- より細かい空間分解能
- 直接的な予測モデル
- 層間の直接比較（winner-take-all）

### 相補的な使用

**論文での報告例**:
1. **Main Results**: 分析1の結果
   - 「CLIP Middle層は側頭葉で有意な活動を示した（cluster-FWE p < 0.05）」

2. **Validation**: 分析2の結果
   - 「Searchlight分析でも、同領域でMiddle層が最高のR²を示した（R² = 0.18）」

3. **New Insights**: 分析2のwinner-take-all
   - 「脳の階層構造とDNN階層の明確な対応関係が観察された」

---

## 技術的詳細

### LSS GLM仕様
- **推定方法**: 最小二乗法（SPM実装）
- **デザイン**: 各画像 = 独立regressor
- **共変量**: 運動パラメータ6自由度
- **出力形式**: beta_XXXX.nii（非圧縮）

### Searchlight仕様
- **半径**: 5mm（約100-300ボクセル/球）
- **回帰手法**: Ridge回帰
- **正則化**: α = 自動選択（RidgeCV、2000ボクセルでサンプリング）
- **Cross-validation**: 12-fold KFold
- **特徴量前処理**:
  1. PCA（分散70%保持）
  2. 標準化（平均0、分散1）
- **評価指標**:
  - R²（主要指標、負値は0にクリップ）
  - Pearson相関係数
- **並列処理**: threading backend（全CPU使用）

---

## 現在の状況
- ✅ DNN生特徴抽出完了（全チャネル、PCAなし）
- 🔄 LSS GLM実行中（6/20名完了、推定残り時間: 約30分）
- ⏳ Searchlight分析待機中（LSS GLM完了後に実行）

---

## Taskfile コマンド

```bash
# LSS GLM実行（並列）
task lss_glm_parallel

# Searchlight分析（CLIP）
task dnn_encoding_lss_clip

# Searchlight分析（ConvNeXt）
task dnn_encoding_lss_convnext

# 結果確認
python scripts/dnn_analysis/mvpa/check_encoding_results.py
```

---

## 次のステップ

1. **LSS GLM完了待ち** → 約30分後
2. **Searchlight実行**:
   ```bash
   task dnn_encoding_lss_clip
   task dnn_encoding_lss_convnext
   ```
   各実行時間: 約2-3時間/モデル
3. **結果の可視化と解釈**

---

## 参考文献
- Mumford et al. (2012). "Deconvolving BOLD activation in event-related designs for multivoxel pattern classification analyses" (LSS method)
- Kriegeskorte et al. (2006). "Information-based functional brain mapping" (Searchlight)
- Naselaris et al. (2011). "Encoding and decoding in fMRI" (Encoding models)
- Kay et al. (2008). "Identifying natural images from human brain activity" (Voxel-wise encoding)
