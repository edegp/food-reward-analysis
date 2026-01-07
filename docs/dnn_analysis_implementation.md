# DNN Analysis Implementation - LayerGroup-based PCA

## 概要

このドキュメントは、food-brainプロジェクトにおけるDNN特徴量抽出とPCA分析の実装について説明します。
特に、LayerGroup-based PCAアプローチの導入と、ConvNext/CLIPモデルの統合について詳述します。

## アーキテクチャ

### 使用モデル

1. **ConvNeXt Base** (`DNNs_model/v9/res_L/convnext_base_regression.pth`)
   - 食品画像に対する評価スコア予測用にファインチューニング済み
   - Layer構成: features_1, features_3, features_5, features_7, classifier

2. **CLIP** (事前学習済みモデル)
   - ResNetベースのアーキテクチャ (CLIP-ResNet)
   - Layer構成: stage0, stage1, stage2, stage3, head

### 出力形式

**LayerGroup形式のCSVファイル:**
- `data_images/dnn_pmods/convnext_pcs.csv`
- `data_images/dnn_pmods/clip_pcs.csv`

```csv
image_id,{model}.Initial,{model}.Middle,{model}.Late,{model}.Final,{model}.rating_mean
0001.jpg,PC1,PC2,...,PC1,PC2,...,PC1,PC2,...,PC1,PC2,...,rating_value
```

## 個別レイヤー形式 vs LayerGroup形式

### 個別レイヤー形式 (旧アプローチ)

**特徴:**
- 各レイヤーごとに独立してPCAを実行
- レイヤー数が多い場合、列数が膨大になる
- GLM解析スクリプトとの互換性がない

**CSV形式:**
```csv
image_id,clip.act3,clip.layer1_0_act3,clip.layer1_1_act3,clip.layer1_2_act3,...
0001.jpg,PC1,PC2,...,PC1,PC2,...,PC1,PC2,...,PC1,PC2,...
```

**問題点:**
- GLMスクリプトが期待する形式と異なる
- レイヤーごとの特徴量が分散し、段階的な表現を捉えにくい
- 不必要に高次元のデータになる

### LayerGroup形式 (現アプローチ)

**特徴:**
- 複数のレイヤーをグループ化してまとめてPCAを実行
- ネットワークの階層的な特徴表現を考慮
- GLM解析との互換性が高い

**グループ分け (layer_groups.json):**
```json
{
  "convnext": {
    "groups": [
      {"label": "Initial", "layers": ["convnext.features_1_0_block_4", ...]},
      {"label": "Middle", "layers": ["convnext.features_5_4_block_4", ...]},
      {"label": "Late", "layers": ["convnext.features_5_14_block_4", ...]},
      {"label": "Final", "layers": ["convnext.features_5_23_block_4", ...]}
    ]
  }
}
```

**利点:**
1. **階層的な特徴表現:**
   - Initial: 低レベル特徴 (エッジ、テクスチャ)
   - Middle: 中レベル特徴 (パターン、部分的なオブジェクト)
   - Late: 高レベル特徴 (複雑なオブジェクト)
   - Final: 最終的な抽象表現 + 行動評価

2. **次元削減の効率性:**
   - グループ内のレイヤー数 × 2 = PC数
   - 例: Initialグループが10レイヤー → 20 PCs

3. **GLM解析との互換性:**
   - SPMスクリプトが期待する列名と一致
   - Parametric Modulationsとして直接使用可能

## LayerGroup-based PCA実装詳細

### 処理フロー

```
1. Layer Groups定義の読み込み (layer_groups.json)
2. 画像リストのフィルタリング (568枚の評価済み画像)
3. Activations抽出 (batch処理)
   ├─ バッチごとに画像を読み込み
   ├─ 各LayerGroupの全レイヤーから特徴量抽出
   └─ 一時ファイルに保存 (.npy形式)
4. IncrementalPCAによる次元削減
   ├─ グループ内レイヤーの特徴量を連結
   ├─ n_components = num_layers_in_group × 2
   └─ メモリ効率的なバッチ学習
5. PC値の計算と保存
   └─ CSVファイルに出力
```

### コード構造

**主要関数 (`extract_dnn_pcs.py`):**

```python
def extract_clip_activations_group_pca(
    image_paths,
    out_dir,
    config,
    components_per_layer=2,
    batch_size=64,
    behavior_pcs=None,
    temp_dir=None
):
    """
    CLIPモデルからLayerGroup-based PCAで特徴量を抽出

    Args:
        image_paths: 画像パスのリスト
        out_dir: 出力ディレクトリ
        config: layer_groups.jsonから読み込んだ設定
        components_per_layer: 各レイヤーあたりのPC数
        batch_size: バッチサイズ
        behavior_pcs: 行動評価データ
        temp_dir: 一時ファイル保存先
    """
    # 1. Layer Groups定義の読み込み
    # 2. グループごとの処理
    for group_name, layers in groups.items():
        # 3. Activations抽出 (バッチ処理)
        # 4. IncrementalPCAフィッティング
        # 5. PC値の計算
    # 6. CSVファイルに保存
```

### Layer Groups定義 (`layer_groups.json`)

**ConvNeXt:**
- Initial: 10レイヤー (features_1, features_3, features_5の初期)
- Middle: 10レイヤー (features_5の中盤)
- Late: 9レイヤー (features_5の後半)
- Final: 9レイヤー (features_5の最終 + features_7 + classifier)

**CLIP:**
- Initial: 10レイヤー (stage0, stage1, stage2の初期)
- Middle: 10レイヤー (stage2の中盤)
- Late: 9レイヤー (stage2の後半)
- Final: 9レイヤー (stage2の最終 + stage3 + head)

**各グループのPC数:**
- Initial: 20 PCs (10レイヤー × 2)
- Middle: 20 PCs (10レイヤー × 2)
- Late: 18 PCs (9レイヤー × 2)
- Final: 18 PCs (9レイヤー × 2)
- rating_mean: 1列 (行動評価)

**合計列数 (1モデルあたり):** 77列 (image_id + 76 PCs + 1 rating)

## 実装の進化

### Phase 1: 個別レイヤーPCA
- 各レイヤーごとに独立してPCA
- 大量の列数 (レイヤー数 × PC数)
- GLM解析で使用できない

### Phase 2: LayerGroup-based PCA (現在)
- レイヤーをグループ化してPCA
- 階層的な特徴表現を捉える
- GLM解析と互換性あり
- メモリ効率的な実装 (IncrementalPCA)

### Phase 3: ConvNeXt Base導入
- 食品画像評価タスクでファインチューニング
- rating_mean列の追加 (行動予測値)
- 評価済み568画像のみを処理

## 使用方法

### 基本的な実行

```bash
uv run --project scripts/dnn_analysis/preprocess python scripts/dnn_analysis/preprocess/extract_dnn_pcs.py \
  --convnext-checkpoint DNNs_model/v9/res_L/convnext_base_regression.pth \
  --image-dir Database \
  --image-list data_images/used_image_ids.txt \
  --out-dir data_images/dnn_pmods \
  --use-group-pca \
  --components-per-layer 2 \
  --batch-size 64
```

### 重要なオプション

- `--use-group-pca`: LayerGroup-based PCAを使用 (**必須**)
- `--image-list`: 処理する画像リスト (**推奨**: 568画像のみ)
- `--components-per-layer`: 各レイヤーあたりのPC数 (デフォルト: 2)
- `--batch-size`: バッチサイズ (デフォルト: 64)
  - 画像数が少ない場合: 64でも安全
  - 全画像(896枚)の場合: 16-32を推奨
- `--temp-dir`: 一時ファイル保存先 (ディスク容量不足時に指定)
- `--disable-convnext`: ConvNeXtを無効化 (CLIPのみ実行)
- `--disable-clip`: CLIPを無効化 (ConvNeXtのみ実行)

### 画像フィルタリング

**568画像リストの生成:**
```bash
python scripts/dnn_analysis/preprocess/collect_used_images.py
```

**出力:** `data_images/used_image_ids.txt` (568行)

## 技術的な詳細

### IncrementalPCA

大規模な特徴量行列を扱うため、IncrementalPCAを使用:

#### 基本的な仕組み

IncrementalPCAは**単一のPCAモデルを累積的に学習**します。バッチごとに別々のPCAを作るのではありません。

```python
from sklearn.decomposition import IncrementalPCA

# Step 1: PC数を決定 (固定)
n_components = len(layers) * components_per_layer  # 例: 10レイヤー × 2 = 20 PCs

# Step 2: 単一のPCAモデルを初期化
pca_model = IncrementalPCA(n_components=n_components)  # n_components=20で固定

# Step 3: バッチごとに累積学習 (メモリ効率のため)
pca_fit_batch_size = 50  # メモリ節約のため50画像ずつ処理
for batch_start in range(0, n_images, pca_fit_batch_size):  # 568 / 50 = 12回ループ
    batch_end = min(batch_start + pca_fit_batch_size, n_images)

    # 各画像のグループ内全レイヤーの特徴量を連結
    batch_features = []
    for img_idx in range(batch_start, batch_end):
        img_features = []
        for layer in group_layers:  # 例: 10レイヤー
            img_features.append(layer_activation)  # 例: (512,)
        concatenated = np.concatenate(img_features)  # 例: (6656,)
        batch_features.append(concatenated)

    batch_array = np.array(batch_features)  # (50, 6656)
    pca_model.partial_fit(batch_array)  # 内部の共分散行列を更新

# Step 4: 全画像にPCA適用
all_pcs = pca_model.transform(all_features)  # (568, 20)
```

#### 重要なポイント

**誤解:** バッチごとに別々のPCAが作られる ❌
```
バッチ1 (50画像) → PCA → 20 PCs
バッチ2 (50画像) → PCA → 20 PCs
...
→ 合計: 12バッチ × 20 = 240 PCs ❌ 間違い!
```

**正解:** 単一のPCAモデルを累積的に学習 ✅
```
PCAモデル初期化 (n_components=20)
  ↓
バッチ1 (50画像) → partial_fit → 共分散行列更新
  ↓
バッチ2 (50画像) → partial_fit → 共分散行列更新
  ↓
...バッチ12まで繰り返す
  ↓
最終的なPCAモデル: 568画像全体から学習した20個の主成分
  ↓
transform → 各画像のPC値 (568, 20)
```

#### バッチサイズの役割

- **pca_fit_batch_size = 50**: メモリ効率のため
- **目的**: 全568画像 × 6656次元 (約30MB) を一度にメモリに載せない
- **効果**: 50画像ずつ処理 (約2.6MB) でメモリ節約
- **注意**: バッチサイズはPC数に影響しない (常に20 PCs)

#### 具体例 (Initialグループ)

```
画像数: 568
レイヤー数: 10
各レイヤーの特徴次元: 約500-1000次元
連結後の総特徴次元: 約6656次元

IncrementalPCA設定:
  n_components = 10 × 2 = 20 (固定)
  pca_fit_batch_size = 50 (メモリ節約)

処理フロー:
  1. 568画像を50ずつ12回に分けて学習
  2. 各バッチで共分散行列を累積更新
  3. 最終的なPC数: 20個 (変わらない)
  4. 出力: (568, 20) の行列
```

### 一時ファイル管理

**ファイル構造:**
```
{temp_dir}/
├── clip_group_pca_temp/
│   ├── clip.stage0_0_batch_0.npy
│   ├── clip.stage0_0_batch_1.npy
│   ├── ...
│   └── clip.rating_mean_batch_8.npy
```

**削除タイミング:**
- PCA完了後、自動的に削除される
- エラー時は手動で削除が必要

### メモリ使用量の最適化

**バッチサイズ vs メモリ:**
- batch_size=16: 約1.5GB RAM
- batch_size=32: 約3.0GB RAM
- batch_size=64: 約6.0GB RAM

**568画像の場合:**
- batch_size=64: 9バッチ (推奨)
- batch_size=32: 18バッチ
- batch_size=16: 36バッチ

## GLM解析との連携

### CSVファイルの読み込み (MATLAB)

```matlab
% scripts/dnn_analysis/analysis/first_level/load_dnn_pcs_from_csv.m
function pcs = load_dnn_pcs_from_csv(csv_file, model_name)
    % LayerGroup形式のCSVを読み込み
    % model_name: 'convnext' or 'clip'

    data = readtable(csv_file);

    % 各グループのPC値を抽出
    initial_cols = startsWith(data.Properties.VariableNames, ...
                             [model_name '.Initial']);
    middle_cols = startsWith(data.Properties.VariableNames, ...
                            [model_name '.Middle']);
    % ...

    pcs.Initial = data{:, initial_cols};
    pcs.Middle = data{:, middle_cols};
    pcs.Late = data{:, late_cols};
    pcs.Final = data{:, final_cols};
end
```

### Parametric Modulationsの設定

```matlab
% GLMモデルでの使用例
for group_idx = 1:4
    group_names = {'Initial', 'Middle', 'Late', 'Final'};
    group_name = group_names{group_idx};

    % PC値をParametric Modulationsとして設定
    matlabbatch{1}.spm.stats.fmri_spec.sess(run).regress(group_idx).name = ...
        ['DNN_' group_name];
    matlabbatch{1}.spm.stats.fmri_spec.sess(run).regress(group_idx).val = ...
        pcs.(group_name)(image_ids, :);
end
```

## 参考ファイル

### 設定ファイル
- `scripts/dnn_analysis/config/layer_groups.json` - Layer定義

### スクリプト
- `scripts/dnn_analysis/preprocess/extract_dnn_pcs.py` - PCA抽出
- `scripts/dnn_analysis/preprocess/collect_used_images.py` - 画像リスト生成
- `scripts/dnn_analysis/analysis/first_level/load_dnn_pcs_from_csv.m` - CSV読み込み
- `scripts/dnn_analysis/analysis/first_level/make_glm_dnn_run.m` - GLMモデル構築

### データファイル
- `data_images/used_image_ids.txt` - 568画像リスト
- `data_images/dnn_pmods/convnext_pcs.csv` - ConvNeXt PCs
- `data_images/dnn_pmods/clip_pcs.csv` - CLIP PCs

### モデルファイル
- `DNNs_model/v9/res_L/convnext_base_regression.pth` - ConvNeXt checkpoint

---
**作成日:** 2025-11-17
**実装環境:** Python 3.x, PyTorch, scikit-learn
**対象モデル:** ConvNeXt Base, CLIP-ResNet
