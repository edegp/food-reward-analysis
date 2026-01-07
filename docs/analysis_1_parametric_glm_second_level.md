# 分析1：パラメトリックGLM + Second-Level統計分析

## 概要
DNN各層の特徴量が脳活動に与える影響を、被験者内GLM→被験者間統計検定という標準的なfMRI分析パイプラインで調べる。

---

## データ
- **被験者**: 20名
- **刺激**: 食品画像（896枚）
- **fMRIデータ**: 各被験者3セッション、計12ラン
- **DNNモデル**:
  - CLIP (Initial 2層, Middle 4層, Late 3層, Final 1層)
  - ConvNeXt (Initial 2層, Middle 4層, Late 3層, Final 1層)

---

## 分析の流れ

```
【準備】
食品画像（896枚）
    ↓
DNN特徴抽出
    ↓
各層のPC1-3特徴量

【First-Level: 被験者内分析】
fMRI収集（各被験者12ラン）
    ↓
パラメトリックGLM
    ↓
各層×各PCのcontrast画像（20名分）

【Second-Level: 被験者間分析】
全被験者のcontrast画像を集約
    ↓
Group-level統計検定
    ↓
多重比較補正
    ↓
各層の有意な脳活動領域マップ
```

---

## Step 1: First-Level パラメトリックGLM

### 目的
各DNN層の特徴量（PC1-3）が脳活動に与える影響を、個々の被験者ごとに推定する。

### 手法
**前処理**:
- CLIP/ConvNeXt のPC（PC1-3）を一括読み込み
- 層グループ設定に基づいて解析対象レイヤーをサンプリング

**GLMのデザイン**:
1. 各画像提示をイベントとしてモデル化（HRF畳み込み、刺激継続時間0）
2. その画像に対応する各DNN層のPC1-3をパラメトリック変調として追加
3. `Question` `Response` などの事象および6軸モーションを共変量に投入

**例（CLIP layer3_0 の場合）**:
```
脳活動 = β₀ × 画像提示 + β₁ × (画像提示 × PC1)
         + β₂ × (画像提示 × PC2) + β₃ × (画像提示 × PC3)
         + 運動パラメータ + 誤差
```

### Contrast設定
各層について以下を作成（T-contrastのみ）：

1. **PC別T-contrast**: Layer × PC1, PC2, PC3
2. **PC平均T-contrast**: 同一層内のPC1-3を同重みで加算したTコントラスト
   → Second-Levelの入力用（条件数削減）

### 出力
- 各被験者×各層×各PC の T-contrast（`con_****.nii`）
- 各被験者×各層のPC平均T-contrast（PC1-3統合）
- F-contrastはFirst-Levelでは未作成（Second-Levelで作成）

---

## Step 2: Second-Level Group分析

### 目的
全被験者にわたって一貫したDNN層の脳活動パターンを特定し、統計的有意性を検定する。

### 2A. 層（PC平均）ごとのGroup-level分析

**デザイン**: Flexible Factorial (Subject × Layer)
**入力**: First-Levelで作成した各層のPC平均T-contrast

**手順**:
1. 各被験者の最新GLMディレクトリからPC平均T-contrastを収集
2. Subject × Layer の2因子モデルを構築（Layerは被験者内要因）
3. 推定後、Layer Group（Initial/Middle/Late/Final）単位でF-contrastを生成
   - 出力は層グループごとの`spmF_****.nii`
   - 個別層のOne-sample T-testや`spmT`は作成していない

### 2B. 層グループ内PC効果の詳細分析

**デザイン**: Subject × Condition（Condition=Layer × PC）
**目的**: 1つの層グループ内でPC1-3の効果差を見る

**手順**:
1. 対象グループの Layer × PC T-contrast（First-Level出力）を収集
2. Subject × Condition のFlexible Factorialを推定
3. グループ内の全条件を含むF-contrast（`F: Initial LayerGroup`など）を出力
---

## Step 3: 多重比較補正

fMRI分析では数万～数十万ボクセルで同時に検定を行うため、偽陽性を制御する補正が必須。

### 適用する補正方法

#### 1. Uncorrected (p < 0.001)
- **用途**: 探索的分析、ROI設定の参考
- **解釈**: 統計的には弱いが、視覚的な参考として有用
- **注意**: 論文報告には不適切

#### 2. Peak-level FWE (p < 0.05)
- **手法**: Random Field Theory（RFT）に基づく補正
- **特徴**: 最も厳密（保守的）
- **用途**: 強い効果の検出、確実な結果の報告
- **解釈**: 「少なくとも1つの偽陽性が含まれる確率 < 5%」

#### 3. Cluster-level FWE (p < 0.05)
- **手法**:
  1. Cluster-forming threshold: p < 0.001 (uncorrected)
  2. このthresholdを超えるクラスターを形成
  3. クラスターサイズに対してFWE補正
- **特徴**: Peak-level FWEより検出力が高い
- **用途**: 拡がりのある脳活動領域の検出
- **推奨**: 多くの論文で標準的に使用される

### 補正方法の選択指針

**論文報告の推奨順序**:
1. **主要結果**: Cluster-level FWE (p < 0.05)
2. **補足結果**: Peak-level FWE (p < 0.05)
3. **参考**: Uncorrected (p < 0.001) をSupplementary Materialに

**補足**:
- FWEはRFT閾値計算を試行し、失敗時はBonferroniにフォールバック
- Cluster-FWEのクラスター閾値kは自由度に応じた経験的固定値

---

## 期待される結果

### 1. 階層的表現の空間マップ
- **初期層（Initial）**: 後頭葉視覚野（V1/V2/V3）
  - 低次視覚特徴（エッジ、色、テクスチャ）

- **中間層（Middle）**: 側頭葉（IT皮質）
  - 中次視覚特徴（形状、物体パーツ）

- **後期層（Late）**: 前頭葉・側頭葉
  - 高次特徴（カテゴリー、意味情報）

### 2. モデル間の差異
- **CLIP vs ConvNeXt**
  同一パイプライン上で得たF-mapを比較することで、マルチモーダル学習（CLIP）と純視覚学習（ConvNeXt）の階層的表現差を議論可能

### 3. 統計的に頑健な結果
- Cluster-FWEとPeak-level FWEで偽陽性を制御
- 層グループ単位のF-contrastに基づき、再現性のある階層的マップを提示

---

## 解析結果まとめ（FWE p < 0.05）

PC1-3を平均した10条件デザイン（Subject × Layer交互作用あり）で群解析を実施し、層グループごとにFWE p < 0.05で閾値処理した結果は以下の通り。

### ConvNeXt（嗜好評価でファインチューニング）

| LayerGroup | 有意ボクセル数 | ピークF値 | ピークMNI座標 (X, Y, Z) |
| ---------- | -------------- | --------- | ----------------------- |
| Initial    | 250            | 11.13     | (28.5, 18.9, 13.2)      |
| Middle     | 73             | 9.34      | (-2.6, -31.4, 3.6)      |
| Late       | 48             | 7.62      | (26.1, -62.5, -25.2)    |
| Final      | 7,121          | 17.67     | (-28.9, -50.6, -10.8)   |

- Initial層からvmPFC/OFC/線条体が活性化し、価値コーディングとの早期結合が顕著。
- Middle層は楔前部・後部帯状回、Late層は右側頭皮質と小脳にピーク。
- Final層は紡錘状回・後頭側頭部・両側前頭前皮質など広範囲に及び、意味・価値統合が最大化。

### CLIP（視覚・言語事前学習）

| LayerGroup | 有意ボクセル数 | ピークF値 | ピークMNI座標 (X, Y, Z) |
| ---------- | -------------- | --------- | ----------------------- |
| Initial    | 2,733          | 24.01     | (33.3, -96.1, 13.2)     |
| Middle     | 90             | 8.75      | (4.6, -40.9, 18.0)      |
| Late       | 11             | 6.40      | (35.7, 30.9, 15.6)      |
| Final      | 1,373          | 17.26     | (16.6, -96.1, 6.0)      |

- Initial層は後頭葉視覚野を中心に広範囲な活性化を示し、典型的なボトムアップ処理が支配的。
- Middle層は後部帯状回と楔前部に限定的なピーク、Late層は右前頭皮質の小クラスターのみ。
- Final層は後頭から頭頂に集中し、意味的プーリング後も視覚主導の表現が支配的。

### モデル間の主な違い

1. **視覚階層**: CLIPは初期視覚野での強い反応から徐々に減衰する一方、ConvNeXtは初期から分散し最終層で大規模な活動を示す。
2. **価値関連領域**: ConvNeXtはInitial層でvmPFC/OFC/線条体が有意となり、嗜好評価タスク用ファインチューニングの影響が確認できる。CLIPでは同領域の有意クラスタは観測されない。
3. **高次視覚処理**: 両モデルともFinal層で紡錘状回・側頭葉が有意だが、ConvNeXtの方がボクセル規模・ピークF値ともに大きい。

---

## 結果の解釈例

**例: CLIP Middle層グループ（layer3_0）**

```
Cluster-level FWE corrected (p < 0.05):
- 右側頭葉 (peak: [42, -54, -12], Z = 5.2, k = 342 voxels)
- 左側頭葉 (peak: [-45, -51, -15], Z = 4.8, k = 287 voxels)

解釈:
CLIP の中間層表現は、両側側頭葉で有意に脳活動を説明する。
これは物体認識に関わるIT皮質の活動と一致する。
```

---

## 現在の状況
- DNN特徴抽出（PC1-3）完了
- First-Level GLM（全20名、CLIP/ConvNeXt）完了
- Second-Level Flexible Factorial（層PC平均 / 層グループ詳細）完了
- 多重比較補正済みF-map生成完了

**結果の保存場所**:
- First-level: `results/first_level_analysis/sub-XXX/glm_model/glm_dnn_pmods_{clip,convnext}/`
- Second-level: `results/dnn_analysis/second_level/pc_analysis_{clip,convnext}_layers/`
- F-maps: `results/dnn_analysis/individual_layer_fmaps/`

---

## Taskfile コマンド

```bash
# First-level GLM実行
task dnn_glm_parallel

# Second-level分析（CLIP）
task dnn_second_level_clip

# Second-level分析（ConvNeXt）
task dnn_second_level_convnext

# F-map生成（多重比較補正）
task dnn_layer_fmaps

# 可視化
task dnn_viz_overlay_all

```
