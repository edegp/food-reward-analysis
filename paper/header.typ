
// ------------------------------------------------------------------------------
// 表紙の設定
// ------------------------------------------------------------------------------

#let make_cover(
  thesis_type: "修士論文",
  title: none,
  etitle: none,
  author: none,
  eauthor: none,
  affiliation: none,
  eaffiliation: none,
  date: none,
  edate: none,
  show_english: false,
  show_logo: false,
  logo_path: none,
  logo2_path: none,
) = {
  set page(margin: 2cm)

  // 枠線付きページ
  block(
    width: 100%,
    height: 100%,
    stroke: (paint: rgb(102, 102, 102), thickness: 2pt),
    inset: 2em,
  )[
    #set align(center + top)

    // 上部のスペース
    #v(2fr)

    // 論文タイプ
    #text(size: 1.5em, weight: "bold", font: "BIZ UDPGothic")[#thesis_type]

    #v(6em)

    // 日本語タイトル
    #text(size: 2em, weight: "bold", font: "BIZ UDPGothic")[#title]

    #v(16em)

    // 日本語著者名
    #text(size: 1.3em, font: "BIZ UDPMincho")[#author]

    #v(2em)

    // 日本語所属
    #text(size: 1.3em, font: "BIZ UDPMincho")[#affiliation]

    #v(3em)

    // 日本語日付
    #text(size: 1.3em, font: "BIZ UDPMincho")[#date]

    // 英語版（オプション）
    #if show_english {
      v(3em)

      // 英語タイトル
      text(size: 1.8em, font: "Times New Roman")[#etitle]

      v(4em)

      // 英語著者名
      text(size: 1.3em, font: "Times New Roman")[#eauthor]

      v(1em)

      // 英語所属
      text(size: 1.3em, font: "Times New Roman")[#eaffiliation]

      v(2em)

      // 英語日付
      text(size: 1.3em, font: "Times New Roman")[#edate]
    }

    // ロゴ（オプション）
    #if show_logo and logo_path != none {
      v(2em)
      stack(
        dir: ltr,
        spacing: 1em,
        if logo_path != none { image(logo_path, height: 18mm) },
        if logo2_path != none { image(logo2_path, height: 16mm) },
      )
    }

    // 下部のスペース
    #v(0.5fr)
  ]

  pagebreak()
}

#let set_title(title) = {
  show: block.with(width: 100%)
  set align(center)
  set text(size: 1.7em, font: "BIZ UDPGothic", weight: 400)
  title
  v(10pt)
}

#let set_author(author) = {
  set text(font: ("Times New Roman", "BIZ UDPMincho"))
  text(author.name)
  super(author.insts.map(it => str(it + 1)).join(","))
  if author.email != none {
    footnote(numbering: "*")[
      #if author.address != none {
        [住所: #author.address. ]
      }
      #if author.email != none {
        [Email: #underline(author.email).]
      }
    ]
  }
  if author.cofirst == "thefirst" {
    // footnote("cofirst-author-mark", numbering: "*")
  } else if author.cofirst == "cofirst" {
    locate(loc => query(footnote.where(body: [cofirst-author-mark]), loc).last())
  }
}

#let set_affiliation(id, address) = {
  set text(size: 0.8em, font: ("Times New Roman", "BIZ UDPMincho"))
  super([#(id + 1)])
  address
}

#let set_author_info(authors, affiliations) = {
  {
    show: block.with(width: 100%)
    authors
      .keys()
      .map(key => (
        return set_author((
          name: key,
          insts: affiliations
            .keys()
            .enumerate()
            .filter(((ik, k)) => k == authors.at(key).affiliation.at(0))
            .map(((ik, k)) => ik),
          corresponding: if ("email" in authors.at(key) and "address" in authors.at(key)) { true } else { false },
          cofirst: authors.at(key).cofirst,
          address: if ("address" in authors.at(key)) { authors.at(key).address } else { none },
          email: if ("email" in authors.at(key)) { authors.at(key).email } else { none },
        )),
      ))
      .join(", ")
  }
  {
    show: block.with(width: 100%, above: 0.5em)
    affiliations
      .keys()
      .enumerate()
      .map(((ik, key)) => {
        set_affiliation(ik, affiliations.at(key))
      })
      .join(", ")
  }
}



#let set_abstract(abstract, keywords) = {
  // Abstract and keyword block
  set align(horizon)
  if abstract != [] {
    stack(
      dir: ttb,
      spacing: 1em,
      ..(
        [
          #set text(size: 1em, font: ("Times New Roman", "BIZ UDPMincho"))
          #show heading: set text(weight: "semibold")
          #show heading: set align(center)
          #heading(outlined: false)[概要]
          #set par(first-line-indent: 2em, spacing: 0.8em, leading: 0.55em)
          #block(
            above: 2em,
            below: 1em,
          )[#abstract]
        ],
        if keywords.len() > 0 {
          show raw: set text(font: ("Times New Roman", "BIZ UDPMincho"))
          set text(font: ("Times New Roman", "BIZ UDPMincho"))
          text(size: 1em, [キーワード: ])
          text([#keywords.join([, ]).])
        } else { none },
      ),
    )
  }
}

#let set_body(body) = {
  // 見出しスタイル設定
  show heading: set text(lang: "en", size: 1em, font: ("Helvetica", "BIZ UDPGothic"), tracking: 0.03em)
  // 見出し１では改ページする
  show heading.where(level: 1): it => {
    colbreak()
    block(above: 4em, below: 2em)[
      #it
    ]
  }
  show heading.where(level: 2): it => block(above: 2em, below: 1.5em)[
    #it
  ]
  show heading.where(level: 3): it => block(below: 1.3em)[
    #set text(size: 1.1em)
    #it
  ]
  set heading(numbering: "1.1")
  set par(first-line-indent: 1.5em, spacing: 1.3em, leading: 1.25em)
  set footnote(numbering: "1")
  // 図表のキャプションスタイル設定
  show figure.caption: set text(lang: "en", size: 0.8em, font: "BIZ UDPGothic", tracking: 0.03em)
  set text(lang: "en", size: 1em, font: ("Times New Roman", "BIZ UDPMincho"), tracking: 0.03em)
  body
}


#let appendix(body) = {
  // 付録用の見出し設定

  // レベル1の見出しはnumbering無し（タイトルのみ表示）
  show heading.where(level: 1): set heading(numbering: none, supplement: [付録])
  // レベル2以降の見出しは"A.a"形式の番号付き

  set heading(numbering: (first, ..nums) => numbering("A.a", ..nums))
  counter(heading).update(0)
  counter(figure).update(0)
  counter(figure.where(kind: image)).update(0)
  set figure(numbering: "A")
  body
}
