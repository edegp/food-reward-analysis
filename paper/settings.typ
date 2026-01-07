#import "@preview/starter-journal-article:0.4.0": author-meta

#let settings = (
  title: "食べ物の好みの脳計算過程：深層学習による表現",
  authors: (
    "青木悠飛": author-meta(
      "HIT",
      email: "dm240001@g.hit-u.ac.jp",
      cofirst: "thefirst",
    ),
    // "Yoko Mano": author-meta(
    //   "HIT",
    // ),
    // "Michiyo Sugawara": author-meta(
    //   "HIT",
    // ),
    // "Asako Toyama": author-meta(
    //   "HIT",
    // ),
    // "Taiki Kojima": author-meta(
    //   "NCNP",
    // ),
    // "Yuichi Yamashita": author-meta(
    //   "NCNP",
    // ),
    // "Shinsuke Suzuki": author-meta(
    //   "HIT",
    // ),
    // "Yuma Matsuda": author-meta(
    //   "HIT",
    // ),
    // "Koki Nakaya": author-meta(
    //   "HIT",
    // ),
  ),
  affiliations: (
    "HIT": "一橋大学",
    // "NCNP": "National Center of Neurology and Psychiatry",
  ),
  abstract: [
    日常の食事選択は、食品に付与する主観的価値に基づいて行われる。しかし、この主観的価値が脳内でどのように計算されるかについては、ほとんど解明されていない。本研究では、深層ニューラルネットワーク（DNN）モデルを用いて、食品評価の神経計算過程の解明を目指した。199名の参加者による896枚の食品画像の評価データを用いてDCNNを訓練、視覚-言語モデル（CLIP）の埋め込みを入力とする回帰モデルを学習した結果、DNNは主観的価値を有意に予測し、CLIPが最も高い精度（r = 0.78）を示した。DNN層の活性化パターンの分析から、高次属性（主観的価値、健康性、美味しさ）は後期層で強く表現される一方、低次の色情報は全層にわたって一貫して符号化されることが明らかになった。31名の参加者の機能的磁気共鳴画像法（fMRI）データを用いた表現類似性解析（RSA）では、一次視覚野（V1）においてDNNモデルが約65-69%を説明したが、高次視覚野や価値関連領域（vmPFC）では説明率が低下した。エンコーディング解析では、ConvNeXt（視覚モデル）の階層構造が視覚処理階層と対応し、初期層が一次視覚野、後期層が高次視覚野および価値関連領域と関連することが示された。一方、CLIP（視覚-言語モデル）は中間層で言語・情動・報酬関連領域（IFG・PCC・腹側線条体・島皮質・扁桃体）との対応を示し、食品の主観的価値計算では報酬だけでなく言語・情動などの情報処理が関与していることを示唆する。これらの知見は、視覚情報が階層的に処理され、価値計算と統合される脳内メカニズムを明らかにするものである。
  ],
  // keywords: (`深層学習`, `主観的価値`, `計算論的神経科学`),
  keywords: (),
)
