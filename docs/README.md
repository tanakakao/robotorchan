# robotorchan documentation

このディレクトリは、robotorchan の利用方法、モデル選択、理論、ベンチマーク、開発者向け情報をまとめる入口です。

## ドキュメントの読み方

目的に応じて、次の順で参照してください。

### robotorchan を使いたい

1. [Installation](getting_started/installation.md) で利用環境を準備する
2. [Quickstart](getting_started/quickstart.md) で共通 API を確認する
3. [Model selection](getting_started/model_selection.md) で問題設定に合うモデル候補を絞る
4. [モデル概要・使い所](models.md) で個別モデルの用途と制約を確認する
5. [理論ガイド](theory/README.md) でモデルの仮定と理論的背景を確認する
6. [Notebook 一覧](../examples/README.md) から対応する実行例を試す

モデルファミリー別の詳細は [Model guides](models/README.md) に整理しています。内容を削減せず、利用ガイド・Theory・Notebookを相互に辿れる構成を維持します。

### 理論を学びたい

[理論ガイド](theory/README.md) は、ベイズ最適化、Gaussian Process、獲得関数から、高次元・ロバスト・入力不確かさ・次元削減までを扱います。

### 高次元問題を扱いたい

- [高次元入力](models/high_dimensional_inputs.md)
- [高次元出力](models/high_dimensional_outputs.md)
- [高次元 Multi-task](models/high_dimensional_multitask.md)
- [モデル選択](models/high_dimensional_model_selection.md)
- [探索戦略](optimization/high_dimensional_search.md)
- [TuRBO](optimization/turbo.md)

### Robust / Noise / Input uncertainty を扱いたい

入口は [モデルファミリー別ガイド](models/README.md) です。Robust / Noise と Input uncertainty を分けて、生成機構ごとの選択基準と個別モデルへの導線を整理しています。

### 開発・設計情報を確認したい

現在は次の文書があります。

- [Architecture](development/architecture.md)
- [Model design guidelines](development/model_design_guidelines.md)
- [Training API](development/training_api.md)
- [Release procedure](development/releasing.md)

開発文書には現在も有効な設計原則・公開契約・リリース手順だけを残し、Phase記録や解消済みauditはGit/PR履歴で追跡します。

## 情報設計

ドキュメントは次の責務で整理しています。

| 区分 | 責務 |
|---|---|
| Getting Started | インストール、Quickstart、最初のモデル選択 |
| Models | 何を使うか、使い所、API、制約、関連モデル |
| Optimization | 探索戦略、獲得・最適化側の実務ガイド |
| Theory | なぜ使えるか、数式、統計的仮定 |
| Benchmarks | 比較条件、評価方法、実験結果 |
| Development | architecture、設計判断、audit、release |

## 網羅性に関する方針

再編では「短くすること」を目的にしません。

- 公開モデルの説明を削除しない
- 数式・仮定・制約・注意事項を要約によって失わない
- Mixed、Multi-task、Robust、Reduced などの派生モデルも対象にする
- モデルファミリーへ統合する場合も、個別モデルの差異を明示する
- User guide / Theory / Notebook の3方向から各公開モデルへ到達できる状態を維持する
- 実装を source of truth として documentation coverage を検証する

現在の機械可読な対応表は [model_coverage.json](model_coverage.json) です。

## 現在の構造

```text
docs/
├── README.md
├── getting_started/
├── models/
├── optimization/
├── theory/
├── benchmarks/
└── development/
```

旧パス維持のための互換文書や旧名エイリアスは作らず、参照側を現在の構造へ完全移行します。
