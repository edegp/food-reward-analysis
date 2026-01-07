# 深層ニューラルネットワークを用いたfMRI解析パイプライン

## 概要

本パイプラインは、深層ニューラルネットワーク（DNN）特徴量を用いて食品画像の神経表現を解析する包括的なフレームワークを実装しています。最先端のDNN（ConvNeXtとCLIP）から抽出された階層的な視覚特徴量が脳活動パターンにどのように対応するかを、パラメトリックモジュレーション法を用いて調査します。パイプラインには、特徴量抽出、mass-univariate GLM解析、群レベル統計推論、サーチライト法を用いた多変量パターン解析（MVPA）が含まれます。

## パイプライン構成

解析パイプラインは5つの主要コンポーネントで構成されています：

1. **DNN特徴量抽出**: 事前学習済みConvNeXtおよびCLIPモデルから階層的な活性化ベース主成分を抽出
2. **第1レベルGLM解析**: DNN特徴量を用いた被験者レベルのパラメトリックモジュレーション解析
3. **第2レベル群解析**: Flexible Factorial Designを用いた集団レベルの統計推論
4. **層ごと統計マッピング**: 多重比較補正（FWE、FDR）を用いた個別層解析
5. **多変量パターン解析**: サーチライトベースのRSAおよび分類解析

## 方法

### 1. DNN特徴量抽出

2つの補完的なDNNアーキテクチャから活性化ベースの特徴量を抽出します：

- **ConvNeXt-Base**: 視覚回帰タスクに最適化された現代的な畳み込みアーキテクチャ（食品画像評価データでファインチューニング済み）
- **CLIP-ResNet50**: 画像-テキストペアで事前学習された視覚-言語モデル

各モデルについて、ネットワークの深さ方向に階層的に層をサンプリングします：
- **ConvNeXt**: 初期畳み込み層、中間ブロック、後期特徴層、分類器を含む37層
- **CLIP**: アテンションプーリングを含むResNet50バックボーンの27層

選択した各層で以下を実行：
1. データベース内の全896枚の食品画像の活性化マップを抽出
2. 空間平均プーリングを適用して特徴ベクトルを取得
3. 主成分分析（PCA）を実行し上位3成分を保持
4. GLM解析用の層×PC パラメトリックモジュレータを生成

**出力**: GLM解析用のPCスコア、約57,344行（896画像 × 64層）

### 2. 第1レベルGLM解析

パラメトリックモジュレーションを用いた被験者レベル一般線形モデル（GLM）を構築：

**デザイン行列**:
- **主リグレッサ**: 試行ごとの画像提示
- **パラメトリックモジュレータ**: 層固有のPC1、PC2、PC3値
- **交絡リグレッサ**: 運動パラメータ、生理ノイズ

**コントラスト**:
- **Tコントラスト**: 個別層×PC効果（例: "ConvNeXt features_3_0_block_4 PC1"）
- **Fコントラスト**: 階層的ステージ内の複数層を組み合わせた層グループ効果

**層グルーピング**（階層的組織化）:
- **Initial**: 初期視覚特徴（低レベルのエッジ、テクスチャ）
- **Middle**: 中間表現（形状、パーツ）
- **Late**: 高レベル特徴（物体、構成）
- **Final**: 抽象的・意味的表現（分類器・プーリング層）

### 3. 第2レベル群解析

SPM12のFlexible Factorial Designを用いた集団レベル推論（層ごと解析）。

#### 3.1. 層ごとデザイン（10条件 + 交互作用）

被験者内反復測定におけるランク不足に対処するため、PC平均化アプローチを実装：

**条件削減戦略**:
- PC1、PC2、PC3を1つの層ごとTコントラストに統合（各PCの重み=1）
- 30条件（10層 × 3 PC）から10条件（10層）に削減
- Subject × Layer 交互作用を含む安定した推定を可能に

**デザイン構造**:
- **因子1（Subject）**: ランダム効果、独立=Yes、分散=Equal
  - 被験者間変動をモデル化
  - 母集団からのランダムサンプリングとして20名の被験者
- **因子2（Layer）**: 固定効果、独立=No、分散=Equal
  - 被験者内因子（従属測定）
  - 被験者ごとに10層分の条件
- **交互作用**: Subject × Layer
  - 被験者固有の層効果を捉える
  - 200観測値（20×10）で誤差項推定を可能に
  - 各セルに1観測値しかない被験者内デザインで必須

**統計的根拠**:
- N被験者、K条件の場合、主効果のみはN+Kパラメータが必要
- 交互作用あり: N×Kパラメータだが、N×K観測値に分散
- 10条件の場合: 20×10交互作用項 = 200パラメータ < 200観測値 ✓
- 30条件の場合: 20×30 = 600パラメータ = 600観測値（ランク不足）✗

**層グループごとのFコントラスト**:
1. **Initial LayerGroup**: 初期視覚特徴（2層）
2. **Middle LayerGroup**: 中間表現（4層）
3. **Late LayerGroup**: 高レベル特徴（3層）
4. **Final LayerGroup**: 抽象的・意味的特徴（1層）

各Fコントラストは以下を検定: H₀: グループ内全層が効果なし（omnibus検定）

**マスキング**: Implicit mask（SPMが自動的に全被験者のデータが存在する領域を計算）

**出力**:
- 各層グループの群レベルF-map（spmF_0001.nii ～ spmF_0004.nii）
- 完全なモデル仕様とコントラストを含むSPM.mat
- 因子構造を示すデザイン行列の可視化

### 4. 層ごと統計マッピング

厳密な多重比較補正を用いた個別層解析：

**補正方法**:
- **FWE（Family-Wise Error）**: 任意の偽陽性の確率を制御（p < 0.05）
- **FDR（False Discovery Rate）**: 偽発見の期待割合を制御（q < 0.05）

**層ごとの解析**:
1. 全Layer × PC条件の第2レベルfactorial designを作成
2. 層の主効果のF統計量を計算
3. 閾値補正を適用
4. クラスタ統計を抽出（ピーク座標、広がり、強度）

**出力**: 各個別層の解剖学的クラスタテーブル付き統計マップ

### 5. 多変量パターン解析（MVPA）

局所的な脳パターンからDNN情報をデコードするサーチライトベース解析：

**サーチライト設定**:
- **半径**: 5mm球形近傍
- **特徴選択**: サーチライト内のZ標準化されたベータ推定値
- **交差検証**: 被験者レベルleave-one-run-out

**解析タイプ**:

#### 5.1. 表現類似性解析（RSA）
- ベータパターンから神経RDM（表現非類似度行列）を計算
- 層×PC距離からモデルRDMを計算
- 神経RDMとモデルRDM間のSpearman相関を算出
- 脳全体の相関マップを生成

#### 5.2. 層グループ分類
- どの層グループ（Initial/Middle/Late/Final）が活動パターンを生成したかをデコードする多クラスSVMを訓練
- one-vs-rest戦略を使用
- 分類精度マップを出力

#### 5.3. 主成分分類
- どのPC（PC1/PC2/PC3）が脳活動を調整したかをデコード
- 特定の特徴次元に対する局所的感度を評価
- PCデコード精度マップを生成

### 6. 可視化

異なる解析レベルのための複数の可視化アプローチ：

#### 6.1. 層グループGlass Brain可視化
- **レイアウト**: 4行表示（Initial、Middle、Late、Final）
- **ビュー**: 統合多方向glass brain（左、軸位、右、冠状）
- **閾値**: FDR補正 p < 0.05
- **表示**: 有意なボクセルを持つ行のみ表示
- **統計**: グループごとのボクセル数とピーク統計

#### 6.2. 個別層F-map
- **レイアウト**: 全個別層のグリッド表示
- **補正**: FWE p < 0.05
- **オーバーレイ**: MNIテンプレート上の統計マップ
- **カラーマップ**: F統計量用のhot colormap

#### 6.3. オーバーレイ可視化
- **層グループ**: グループごとに異なる色の透明オーバーレイ
- **棒グラフ**: クラスタ範囲の定量的比較
- **被験者ごと**: 個別および群レベル表示

#### 6.4. 第2レベル結果
- **柔軟な表示**: 設定可能な閾値と補正方法
- **複数ビュー**: 軸位、矢状、冠状スライス
- **Glass Brain**: 全脳投影ビュー

## 実行手順

### 前提条件

**ソフトウェア要件**:
- MATLAB R2020b以降、SPM12付き
- Python 3.10以降、uvパッケージマネージャ付き
- Task（taskfile.dev）によるワークフロー実行

**データ要件**:
- BIDS形式の前処理済みfMRIデータ
- 食品画像データベース（896画像）
- 事前学習済みDNNモデル（ConvNeXtチェックポイント、CLIP事前学習済み）

### ステップごとの実行

#### ステップ1: DNN特徴量抽出

食品画像から活性化ベースPCを抽出：

```bash
# ConvNeXtとCLIP両方から特徴量を抽出
task dnn_features
```

`data_images/dnn_pmods/`にパラメトリックモジュレータファイルを生成：
- `convnext_pcs.csv`（約57,344行）
- `clip_pcs.csv`（約57,344行）

#### ステップ2: 第1レベルGLM解析

パラメトリックモジュレーションを用いた被験者レベルGLMを実行：

```bash
# 全被験者を並列実行（推奨）
task dnn_glm_parallel

# または単一被験者を実行（テスト用）
task dnn_glm
```

**出力場所**: `results/first_level_analysis/sub-*/glm_model/glm_dnn_pmods_{convnext,clip}/`

各GLMディレクトリの内容：
- `SPM.mat`: モデル仕様と結果
- `beta_*.nii`: パラメータ推定値
- `con_*.nii`: コントラスト画像（TおよびFコントラスト）
- `spm*.nii`: 統計マップ

#### ステップ3: コントラスト更新（オプション）

モデルを再推定せずにコントラストを追加または修正する場合：

```bash
task dnn_update_contrasts
```

全被験者の既存GLM結果のFコントラストとTコントラストを更新します。

#### ステップ4: 第2レベル群解析

集団レベル統計マップを作成：

```bash
# ConvNeXtのみ
task dnn_second_level_convnext

# CLIPのみ
task dnn_second_level_clip

# 両モデル（推奨）
task dnn_second_level_both
```

**出力場所**: `results/dnn_analysis/second_level/pc_analysis_{convnext,clip}_layers/`

各ディレクトリの内容：
- `SPM.mat`: 第2レベルモデル仕様
- `spmF_*.nii`: 各層グループのF統計量マップ（Fコントラスト1-4）
- `batch/`: 再現性のためのMATLABバッチスクリプト

**層グループマッピング**:
- `spmF_0001.nii`: Initial LayerGroup
- `spmF_0002.nii`: Middle LayerGroup
- `spmF_0003.nii`: Late LayerGroup
- `spmF_0004.nii`: Final LayerGroup

#### ステップ4.1: 群結果の可視化

全層グループのglass brain可視化を生成：

```bash
# ConvNeXt層ごと結果（FDR p<0.05）
cd scripts/dnn_analysis/visualization
uv run python visualize_layerwise_results.py --source convnext --threshold 0.05 --correction fdr \
  --spm-dir ../../results/dnn_analysis/second_level/pc_analysis_convnext_layers/YYYYMMDD_HHMMSS

# CLIP層ごと結果（FDR p<0.05）
uv run python visualize_layerwise_results.py --source clip --threshold 0.05 --correction fdr \
  --spm-dir ../../results/dnn_analysis/second_level/pc_analysis_clip_layers/YYYYMMDD_HHMMSS

# FWE補正（より厳密）
uv run python visualize_layerwise_results.py --source convnext --threshold 0.05 --correction fwe
```

**出力**: `results/dnn_analysis/visualization/layerwise/{source}_layerwise_{correction}_p{threshold}.png`

4行のglass brainがInitial、Middle、Late、Final LayerGroupsをピーク統計とともに表示します。

#### ステップ4.2: クラスタ情報の抽出

解剖学的クラスタテーブルを抽出し、スライスビューを作成：

```bash
cd scripts/dnn_analysis/analysis
uv run --project ../visualization python extract_cluster_info.py --source convnext --threshold 4.91
uv run --project ../visualization python extract_cluster_info.py --source clip --threshold 4.91
```

**閾値**: F=4.91はFWE p<0.05に相当

**出力場所**: `results/dnn_analysis/cluster_analysis/`
- `{source}_{LayerGroup}_clusters.csv`: ピーク座標（MNI）、クラスタサイズ、F値
- `{source}_{LayerGroup}_slices.png`: 軸位、冠状、矢状スライスビュー

**解剖学的解釈**:
スライスビューは以下の領域の活性化を明らかにします：
- **視覚野**（後頭葉）: 初期特徴処理
- **vmPFC**（腹内側前頭前皮質）: 主観的価値コーディング
- **OFC**（眼窩前頭皮質）: 感覚-報酬統合
- **線条体**（尾状核/被殻）: 報酬予測と動機づけ
- **側頭皮質**: 物体認識と意味処理

#### ステップ5: 層ごと統計マップ

多重比較補正を用いた個別層F-mapを生成：

```bash
task dnn_layer_fmaps
```

各層の個別第2レベルモデルを作成し、以下を適用：
- FWE補正（p < 0.05）
- Enhanced Bayesian Hierarchical（e-BH）補正

**出力場所**: `results/dnn_analysis/second_level/layer_fmaps_{convnext,clip}/layer_*/`

各層ディレクトリには補正済み統計マップとクラスタテーブルが含まれます。

#### ステップ6: 可視化

##### 6.1. 層グループGlass Brain（4行レイアウト）

```bash
# 層グループオーバーレイで全被験者を可視化
task dnn_viz_overlay_all
```

**出力**: `results/dnn_analysis/visualization/all_subjects_overlay/`
- `all_subjects_{convnext,clip}_overlay_with_bars.png`

##### 6.2. 個別層F-map

```bash
# 全層のグリッド可視化
task dnn_viz
```

**出力**: `results/dnn_analysis/visualization/brain_images/`

##### 6.3. 第2レベル結果（カスタム閾値）

```bash
# ソースと閾値を指定
task dnn_viz_second_level -- --source convnext --threshold 0.001 --correction fdr

# オプション:
# --source: convnext または clip
# --threshold: α水準（デフォルト: 0.001）
# --correction: fpr（未補正）、fdr、または bonferroni
```

**出力**: `results/dnn_analysis/visualization/second_level/`

#### ステップ7: MVPAサーチライト解析（オプション）

##### 7.1. 層グループコントラストの作成

まず、必要なコントラスト画像を生成：

```bash
# CLIP用
task dnn_create_layergroup_contrasts_clip

# ConvNeXt用
task dnn_create_layergroup_contrasts_convnext
```

##### 7.2. サーチライト解析の実行

```bash
# 表現類似性解析（RSA）
task dnn_searchlight_rsa_clip
task dnn_searchlight_rsa_convnext

# 層グループ分類
task dnn_searchlight_classify_layer_clip
task dnn_searchlight_classify_layer_convnext

# PC分類
task dnn_searchlight_classify_pc_clip
task dnn_searchlight_classify_pc_convnext

# 1モデルの全解析を実行
task dnn_searchlight_all_clip
task dnn_searchlight_all_convnext
```

**パラメータ**（`Taskfile.yml`で変更可能）:
- `--radius`: サーチライト球の半径（mm）（デフォルト: 5.0）
- `--analysis`: 解析タイプ（rsa、classify_layer、classify_pc、または all）

**出力場所**: `results/dnn_analysis/mvpa/{analysis_type}_{source}/`

### 完全な解析ワークフロー

最初から完全な解析を行う場合：

```bash
# 1. DNN特徴量を抽出
task dnn_features

# 2. 第1レベルGLMを実行（全被験者）
task dnn_glm_parallel

# 3. 第2レベル群解析を作成
task dnn_second_level_both

# 4. 層ごと統計マップを生成
task dnn_layer_fmaps

# 5. 結果を可視化
task dnn_viz_overlay_all

# 6.（オプション）サーチライトMVPAを実行
task dnn_create_layergroup_contrasts_clip
task dnn_create_layergroup_contrasts_convnext
task dnn_searchlight_all_clip
task dnn_searchlight_all_convnext
```

### ユーティリティ

#### 結果の削除（再解析用）

```bash
# CLIP GLM結果を削除
task delete_dnn_glm_clip_results

# ConvNeXt GLM結果を削除
task delete_dnn_glm_convnext_results
```

### 出力構造

```
results/
├── first_level_analysis/
│   └── sub-*/
│       └── glm_model/
│           ├── glm_dnn_pmods_convnext/
│           │   └── YYYYMMDD_HHMMSS/
│           │       ├── SPM.mat
│           │       ├── beta_*.nii
│           │       └── con_*.nii
│           └── glm_dnn_pmods_clip/
│               └── YYYYMMDD_HHMMSS/
│                   └── ...
└── dnn_analysis/
    ├── second_level/
    │   ├── pc_analysis_convnext_layers/
    │   │   └── YYYYMMDD_HHMMSS/
    │   │       ├── SPM.mat
    │   │       └── spmF_*.nii
    │   ├── pc_analysis_clip_layers/
    │   │   └── ...
    │   ├── layer_fmaps_convnext/
    │   │   └── layer_*/
    │   │       ├── fwe/
    │   │       └── ebh/
    │   └── layer_fmaps_clip/
    │       └── ...
    ├── mvpa/
    │   ├── rsa_convnext/
    │   ├── rsa_clip/
    │   ├── classify_layer_convnext/
    │   ├── classify_layer_clip/
    │   ├── classify_pc_convnext/
    │   └── classify_pc_clip/
    └── visualization/
        ├── all_subjects_overlay/
        ├── brain_images/
        ├── layer_fmaps/
        └── second_level/
```

## 結果

### 層ごと群解析（FWE p < 0.05）

PC1+PC2+PC3を平均化した10条件デザイン（Subject × Layer交互作用あり）を用いて層ごと群解析を実施しました。全てのFコントラストはFWE p < 0.05（F ≥ 4.91）で閾値処理しました。

#### ConvNeXt結果

**要約統計**:

| LayerGroup | 有意ボクセル数 | ピークF値 | ピークMNI座標（X, Y, Z） |
| ---------- | -------------- | --------- | ------------------------ |
| Initial    | 250            | 11.13     | (28.5, 18.9, 13.2)       |
| Middle     | 73             | 9.34      | (-2.6, -31.4, 3.6)       |
| Late       | 48             | 7.62      | (26.1, -62.5, -25.2)     |
| Final      | 7,121          | 17.67     | (-28.9, -50.6, -10.8)    |

**主要な解剖学的所見**:

1. **Initial LayerGroup**（初期視覚特徴）:
   - **右前頭皮質**: ピーク (28.5, 18.9, 13.2)、F=11.13、688 mm³
   - **左前頭皮質**: ピーク (-36.1, -0.3, 22.8)、F=9.62、234 mm³
   - **線条体**: ピーク (-2.6, -7.5, -3.6)、F=8.66、68 mm³
   - **vmPFC**: ピーク (4.6, 35.7, -3.6)、F=7.19、165 mm³
   - **右OFC**: ピーク (35.7, -0.3, -25.2)、F=6.54、41 mm³
   - **左OFC**: ピーク (-38.5, 2.1, -32.4)、F=6.52、68 mm³

2. **Middle LayerGroup**（中間表現）:
   - **楔前部/後部帯状回**: ピーク (-2.6, -31.4, 3.6)、F=9.34、41 mm³
   - **左前頭皮質**: ピーク (-31.3, 11.7, 46.8)、F=7.38、96 mm³

3. **Late LayerGroup**（高レベル特徴）:
   - **右側頭皮質**: ピーク (26.1, -62.5, -25.2)、F=7.62、41 mm³
   - **小脳**: ピーク (-2.6, -60.1, -39.6)、F=7.41、27 mm³

4. **Final LayerGroup**（抽象的・意味的表現）:
   - **左紡錘状回**: ピーク (-28.9, -50.6, -10.8)、F=17.67、1,679 mm³
   - **右後頭側頭皮質**: ピーク (40.5, -84.1, 20.4)、F=14.95、2,643 mm³
   - **右紡錘状回**: ピーク (30.9, -48.2, -13.2)、F=14.27、2,822 mm³
   - **左頭頂皮質**: ピーク (-33.7, -43.4, 15.6)、F=11.29、1,362 mm³
   - **両側前頭前皮質**: 背外側および腹外側領域に複数のクラスタ
   - **広範な視覚野**: 後頭部および側頭領域

#### CLIP結果

**要約統計**:

| LayerGroup | 有意ボクセル数 | ピークF値 | ピークMNI座標（X, Y, Z） |
| ---------- | -------------- | --------- | ------------------------ |
| Initial    | 2,733          | 24.01     | (33.3, -96.1, 13.2)      |
| Middle     | 90             | 8.75      | (4.6, -40.9, 18.0)       |
| Late       | 11             | 6.40      | (35.7, 30.9, 15.6)       |
| Final      | 1,373          | 17.26     | (16.6, -96.1, 6.0)       |

**主要な解剖学的所見**:

1. **Initial LayerGroup**（初期視覚特徴）:
   - **右後頭皮質**: ピーク (33.3, -96.1, 13.2)、F=24.01、広範な活性化
   - **左後頭皮質**: ピーク (-24.1, -103.3, 8.4)、F=19.05
   - **両側視覚野**: V1/V2領域の支配的な活性化パターン

2. **Middle LayerGroup**（中間表現）:
   - **後部帯状回**: ピーク (4.6, -40.9, 18.0)、F=8.75、68 mm³
   - **左楔前部**: ピーク (-2.6, -57.8, 15.6)、F=6.90、27 mm³

3. **Late LayerGroup**（高レベル特徴）:
   - **右前頭皮質**: ピーク (35.7, 30.9, 15.6)、F=6.40、13 mm³

4. **Final LayerGroup**（抽象的・意味的表現）:
   - **右後頭皮質**: ピーク (16.6, -96.1, 6.0)、F=17.26
   - **左後頭側頭部**: ピーク (-16.9, -105.7, -1.2)、F=13.90
   - **広範な後部皮質**: 後頭部および頭頂領域

### モデル間比較

**階層的処理パターン**:

ConvNeXtとCLIPの両方が階層的視覚処理の証拠を示しますが、異なる活性化パターンがあります：

1. **Initial LayerGroup**:
   - **CLIP**: 支配的な視覚野活性化（2,733ボクセル）、ボトムアップ視覚特徴抽出を反映
   - **ConvNeXt**: より分散したパターン（250ボクセル）、前頭皮質、vmPFC、OFC、線条体を含み、価値と報酬処理の早期統合を示唆

2. **Middle LayerGroup**:
   - 両モデルとも楔前部/後部帯状回の活性化を示し、中レベル特徴統合を示唆
   - ConvNeXtは追加の前頭皮質関与を示す

3. **Late LayerGroup**:
   - **ConvNeXt**: 側頭皮質と小脳の活性化
   - **CLIP**: 最小限の活性化（11ボクセル）、主に前頭皮質

4. **Final LayerGroup**:
   - **ConvNeXt**: 広範な活性化（7,121ボクセル）、紡錘状回、後頭側頭部、頭頂葉、前頭前皮質全体
   - **CLIP**: 中程度の活性化（1,373ボクセル）、後頭部および後部領域に集中

**価値関連活性化**:

食品嗜好評価でファインチューニングされたConvNeXtは、Initial LayerGroupで価値コーディング領域の活性化を独自に示します：
- **vmPFC**（X=4.6、Y=35.7、Z=-3.6）: 主観的価値計算
- **OFC 両側**（X=±35-38、Y=0-2、Z=-25～-32）: 感覚-報酬統合
- **線条体**（X=-2.6、Y=-7.5、Z=-3.6）: 報酬予測と動機づけ

これらの価値関連活性化はCLIPのInitial LayerGroupでは**観察されず**、嗜好評価でのファインチューニングが早期層表現を価値関連特徴をエンコードするように形成することを示唆しています。

**視覚処理階層**:

- **CLIP**: 強い早期視覚野 → 減少した中期/後期 → 中程度の最終活性化
  - 階層的特徴抽出後の意味的プーリングと一貫したパターン
- **ConvNeXt**: 分散した早期 → 中程度の中期/後期 → 広範な最終活性化
  - 価値統合を伴う連続的な特徴変換を示唆するパターン

### 解剖学的解釈

解析はDNN-脳対応の異なるパターンを明らかにします：

1. **視覚野**（後頭部/側頭部）:
   - CLIPは支配的な早期活性化を示し、視覚-言語事前学習と一貫
   - ConvNeXtは紡錘状回と腹側側頭領域で広範な最終層活性化を示す

2. **価値システム**（vmPFC/OFC/線条体）:
   - ConvNeXtによって早期層で独自に活性化
   - 嗜好評価でのタスク特異的ファインチューニングを反映

3. **意味処理**（紡錘状回/側頭部）:
   - 両モデルとも広範な最終層活性化を示す
   - 高レベルの意味的食品表現を示唆

4. **注意/制御**（前頭前皮質/頭頂葉）:
   - ConvNeXtは階層全体で分散した活性化を示す
   - 特徴処理のトップダウン調整を反映する可能性

## 設定

### 層選択

層グループは`scripts/dnn_analysis/config/layer_groups.json`で設定されています：

- **ConvNeXt**: 4グループ（Initial: 2層、Middle: 4層、Late: 3層、Final: 1層）
- **CLIP**: 4グループ（Initial: 2層、Middle: 4層、Late: 3層、Final: 1層）

層選択を変更するには、JSON設定ファイルを編集してください。

### 統計的閾値

デフォルト閾値は各スクリプトで変更可能です：

- **第2レベルF検定**: FDR q < 0.05（`visualize_layergroup_results.py`内）
- **層ごとF-map**: FWE p < 0.05（`create_layer_fmaps_corrected.m`内）
- **サーチライト**: 未補正（順列ベース推論は別途利用可能）

## 参考文献

本パイプラインは以下の方法を実装しています：

1. **パラメトリックモジュレーション**: Büchel et al. (1998). "Characterizing stimulus-response functions using nonlinear regressors in parametric fMRI experiments." NeuroImage.

2. **Flexible factorial design**: Friston et al. (2005). "Variance components." In: Statistical Parametric Mapping.

3. **サーチライト解析**: Kriegeskorte et al. (2006). "Information-based functional brain mapping." PNAS.

4. **DNN-fMRIマッピング**: Khaligh-Razavi & Kriegeskorte (2014). "Deep supervised, but not unsupervised, models may explain IT cortical representation." PLoS Computational Biology.

## トラブルシューティング

### 問題: GLM推定が失敗する

**解決策**: 以下を確認してください：
- 前処理済みデータが`results/first_level_analysis/sub-*/`に存在する
- DNN PCファイルが`data_images/dnn_pmods/`に存在する
- SPM12が正しくインストールされ、MATLABパスに設定されている

### 問題: 第2レベル解析で被験者が表示されない

**解決策**: 以下を確認してください：
- 第1レベルGLMが正常に完了している
- GLMディレクトリに`latest_run.txt`マーカーが存在する
- 第1レベルと第2レベル間でコントラスト名が一致している

### 問題: 可視化で空のプロットが生成される

**解決策**: 以下を確認してください：
- 統計的閾値が厳しすぎない
- 第2レベルSPM.matが存在する
- Fコントラストファイル（spmF_*.nii）が存在しデータを含んでいる

### 問題: サーチライト解析が非常に遅い

**解決策**:
- 並列処理が有効になっていることを確認
- サーチライト半径を減らすことを検討
- テスト用に被験者のサブセットを使用

## 連絡先

質問や問題については、メインリポジトリのドキュメントを参照するか、解析チームに連絡してください。
