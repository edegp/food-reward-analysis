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
    何を食べるかという選択は、ヒトが毎日行う意思決定である。食事選択は主観的価値に基づくと考えられているが、その詳細な神経計算過程は不明である。本研究では、深層ニューラルネットワーク（DNN）とfMRIを用いて、食品を見た時の主観的価値の計算過程を明らかにすることを目的として解析を行った。199名の参加者による896枚の食品画像の評価データを用いた行動実験では、視覚-言語モデルCLIPが食品の主観的価値を最も高い精度（r = 0.77）で予測した。DNN層の活性化パターン分析から、主観的価値や美味しさなどの高次属性は後期層で、色情報は全層で符号化されることが明らかになった。31名のfMRIデータを用いた解析では、一次視覚野（V1）においてDNNがノイズ上限の65-69%を説明した一方、価値関連領域（vmPFC）では4-8%に留まった。また、ConvNeXtは視覚処理階層に沿った対応を示し、CLIPの中間層は言語・情動関連領域（IFG・島皮質・扁桃体）と対応した。これらの結果は、視覚-言語の統合が食品の価値計算に重要な役割を果たすことを示唆している。
  ],
  // keywords: (`深層学習`, `主観的価値`, `計算論的神経科学`),
  keywords: (),
)
