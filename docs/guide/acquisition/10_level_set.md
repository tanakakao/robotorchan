# Level-set / qStraddle：理論から実装へ

## この章で解決する問い

目的は最適値でなくf(x)=tの境界・閾値上下の分類。posterior平均が境界近くても不確実性が小さければ追加測定価値は限られる。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/10_level_set.md)が正本であり、
この章は**数式の計算対象 → クラス構造 → 内部処理 →近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

`qStraddle`のスコアは`beta*sigma-abs(mu-t)`、`qRandomizedStraddle`は確率的beta、
`qBoundaryVariance`は別の境界近接heuristic。すべてlocal Implemented。

## 採用時の確認と注意事項

q-batchの周辺スコア集約と出力選択の制限、qStraddle betaとUCBのsqrt(beta)慣行の違い、qRandomizedStraddleのresample境界を確認。

## qStraddleで閾値付近の候補を評価する

`qStraddle`は予測平均と閾値の距離を、不確実性による探索ボーナスから引く。`beta`は標準偏差に直接掛かるので、UCBの`sqrt(beta)`と取り違えない。

このスコアは境界探索のheuristicであり、閾値の上下を表す分類確率ではない。q>1では各候補のスコアを平均する。候補間の相関を評価する真のjoint utilityではない。

### BoTorch / robotorchanとの対応

robotorchan は level-set / boundary learning 用に qStraddle、qRandomizedStraddle、qBoundaryVariance を提供します。

これらは現在、scalarized regressionのq>=1に対応し、
ensemble posterior など未定義の reduction semantics は拒否します。

Theory は将来の拡張可能性を説明できますが、現在の利用可能範囲については次を参照してください。

- [Level-set learning guide](../../optimization/level_set_learning.md)
- [qBoundaryVariance guide](../../optimization/boundary-variance.md)
- [統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)

## 実装ごとのscoreと由来

| acquisition | score・係数の意味 | 由来 |
| --- | --- | --- |
| `qStraddle` | `beta * std - abs(mean - target)` | 既存のlevel-set / boundary-learning criterion |
| `qRandomizedStraddle` | 正のrandom variableから探索係数を生成 | 文献に基づくrandomized straddle family |
| `qBoundaryVariance` | varianceとstandardized target proximityを組み合わせる | robotorchan固有のheuristic |

`qRandomizedStraddle`はselection round内で1つのrandom coefficientを固定する。候補ごとの再評価で係数を無関係に再抽選すると、
同じ獲得関数を最適化する契約にならない。`qBoundaryVariance`にはqStraddleのような`beta` parameterを設けていない。APIの対称性だけを理由に係数を追加すると、
別のheuristicを定義することになる。

qStraddleの固定`beta`は実装上の探索係数であり、理論の時変confidence scheduleと自動的に同一にはならない。数学的背景と文献は
[対応する理論章](../../theory/acquisition/10_level_set.md)を参照する。
`qBoundaryVariance`を文献上の名前付き手法として引用しない。

## 適用範囲と他の意思決定との違い

現在のlocal acquisitionはscalarized regressionのq>=1を対象とし、q軸の周辺スコアを平均する。
multi-outputやstructured outputでは境界を定めるscalar quantityを定義し、ensembleなど未定義のreduction semanticsへ拡張しない。
pointwise scoreの上位q点を選ぶことは、joint batch utilityの最適化ではない。

```text
qPosteriorVariance → 関数全体の不確実性を減らす
qStraddle / qBoundaryVariance → 閾値付近を学ぶ
```

level setとexcursion set、level-set estimationとconstrained BO、分類Active Learningは、境界を扱っていても意思決定目的が異なる。
現在のAPI定義と統合範囲は
[Level-set guide](../../optimization/level_set_learning.md)、
[qBoundaryVariance guide](../../optimization/boundary-variance.md)、
[統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)
を参照する。

## スコアを構成するクラスとメソッド

[straddle.py](../../../src/robotorchan/acquisition/active_learning/straddle.py)の
`qStraddle`はBoTorchの`AcquisitionFunction`を継承します。`forward()`はposteriorからmeanとvarianceを取り、
`_select_output()`で出力軸を解決し、標準偏差と閾値からの距離を組み合わせます。`qBoundaryVariance`はこれを継承し、
`forward()`でvarianceを閾値近傍の重みへ掛けます。Gaussian回帰のposteriorを取得する部分と、独自のheuristicを計算する部分は別です。

[randomized_straddle.py][randomized]の`qRandomizedStraddle`も`qStraddle`を継承します。
`_beta_for()`は一様乱数Uから`-2 * log1p(-U)`を作ってbufferへ保存し、`forward()`はその平方根を標準偏差の係数にします。
`resample()`がbufferを無効化するまで同じ探索曲面を保つ設計です。

[randomized]: ../../../src/robotorchan/acquisition/active_learning/randomized_straddle.py

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/28_active_learning_acquisition.ipynb)を参照してくださ
い。
