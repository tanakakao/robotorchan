# Level-set / Straddle：理論から実装へ

## この章で解決する問い

目的は最適値でなくf(x)=tの境界・閾値上下の分類。posterior平均が境界近くても不確実性が小さければ追加測定価値は限られる。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/10_level_set.md)が正本であり、
この章は**数式の計算対象 → クラス構造 → 内部処理 →近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

`Straddle`のスコアは`beta*sigma-abs(mu-t)`、`RandomizedStraddle`は確率的beta、
`BoundaryVariance`は別の境界近接heuristic。すべてlocal Implemented。

## 採用時の確認と注意事項

q=1と出力選択の制限、Straddle betaとUCBのsqrt(beta)慣行の違い、RandomizedStraddleのresample境界を確認。

## Straddleで閾値付近の候補を評価する

`Straddle`は予測平均と閾値の距離を、不確実性による探索ボーナスから引く。`beta`は標準偏差に直接掛かるので、UCBの`sqrt(beta)`と取り違えない。

このスコアは境界探索のheuristicであり、閾値の上下を表す分類確率ではない。また`q=1`の候補別評価であり、複数候補のjoint utilityを求めたものでもない。

### BoTorch / robotorchanとの対応

robotorchan は level-set / boundary learning 用に Straddle、RandomizedStraddle、BoundaryVariance を提供します。

これらは現在、scalarized q=1 regression acquisition として明示的な適用範囲を持ち、
ensemble posterior など未定義の reduction semantics は拒否します。

Theory は将来の拡張可能性を説明できますが、現在の利用可能範囲については次を参照してください。

- [Level-set learning guide](../../optimization/level_set_learning.md)
- [BoundaryVariance guide](../../optimization/boundary-variance.md)
- [統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)

## 実装ごとのscoreと由来

| acquisition | score・係数の意味 | 由来 |
| --- | --- | --- |
| `Straddle` | `beta * std - abs(mean - target)` | 既存のlevel-set / boundary-learning criterion |
| `RandomizedStraddle` | 正のrandom variableから探索係数を生成 | 文献に基づくrandomized straddle family |
| `BoundaryVariance` | varianceとstandardized target proximityを組み合わせる | robotorchan固有のheuristic |

`RandomizedStraddle`はselection round内で1つのrandom coefficientを固定する。候補ごとの再評価で係数を無関係に再抽選すると、
同じ獲得関数を最適化する契約にならない。`BoundaryVariance`にはStraddleのような`beta` parameterを設けていない。APIの対称性だけを理由に係数を追加すると、
別のheuristicを定義することになる。

Straddleの固定`beta`は実装上の探索係数であり、理論の時変confidence scheduleと自動的に同一にはならない。数学的背景と文献は
[対応する理論章](../../theory/acquisition/10_level_set.md)を参照する。
`BoundaryVariance`を文献上の名前付き手法として引用しない。

## 適用範囲と他の意思決定との違い

現在のlocal acquisitionはscalarized regressionの`q=1`を対象とする。
multi-outputやstructured outputでは境界を定めるscalar quantityを定義し、ensembleなど未定義のreduction semanticsへ拡張しない。
pointwise scoreの上位q点を選ぶことは、joint batch utilityの最適化ではない。

```text
PosteriorVariance → 関数全体の不確実性を減らす
Straddle / BoundaryVariance → 閾値付近を学ぶ
```

level setとexcursion set、level-set estimationとconstrained BO、分類Active Learningは、境界を扱っていても意思決定目的が異なる。
現在のAPI定義と統合範囲は
[Level-set guide](../../optimization/level_set_learning.md)、
[BoundaryVariance guide](../../optimization/boundary-variance.md)、
[統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)
を参照する。

## スコアを構成するクラスとメソッド

[straddle.py](../../../src/robotorchan/acquisition/active_learning/straddle.py)の
`Straddle`はBoTorchの`AcquisitionFunction`を継承します。`forward()`はposteriorからmeanとvarianceを取り、
`_select_output()`で出力軸を解決し、標準偏差と閾値からの距離を組み合わせます。`BoundaryVariance`はこれを継承し、
`forward()`でvarianceを閾値近傍の重みへ掛けます。Gaussian回帰のposteriorを取得する部分と、独自のheuristicを計算する部分は別です。

[randomized_straddle.py][randomized]の`RandomizedStraddle`も`Straddle`を継承します。
`_beta_for()`は一様乱数Uから`-2 * log1p(-U)`を作ってbufferへ保存し、`forward()`はその平方根を標準偏差の係数にします。
`resample()`がbufferを無効化するまで同じ探索曲面を保つ設計です。

[randomized]: ../../../src/robotorchan/acquisition/active_learning/randomized_straddle.py

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/28_active_learning_acquisition.ipynb)を参照してくださ
い。
