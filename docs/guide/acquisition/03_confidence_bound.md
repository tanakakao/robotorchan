# Confidence Bound / UCB：理論から実装へ

## この章で解決する問い

UCBは予測平均と不確実性へのoptimism bonusを組み合わせる。固定betaとGP-UCB理論の時変beta scheduleを区別する。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/03_confidence_bound.md)が正本であり、
この章は**数式の計算対象 → クラス構造 →内部処理 → 近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

`UpperConfidenceBound`・`qUpperConfidenceBound`はBoTorch-native（Delegated）。
robotorchanはBoTorchクラスを再実装せずposterior互換modelを提供する。

## 採用時の確認と注意事項

最小化目的の符号、`beta`の慣行、analytic q=1とMC batchの数値的差異に注意。

### BoTorch / robotorchan との対応

BoTorch は analytic UCB と MC batch variant を提供しています。robotorchan は標準 UCB をローカルに再実装せず、
BoTorch native path を利用します。

利用方法は [Standard acquisition](../../optimization/standard_acquisition.md)、
現在の統合範囲は
[統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)
を参照してください。

### beta の意味を API と混同しない

文献によって

$$
\mu(x)+\sqrt{\beta}\sigma(x)
$$

と書く場合と、

$$
\mu(x)+\beta\sigma(x)
$$

と書く場合があります。

したがって parameter 名だけで探索強度を比較せず、**係数が式のどこへ入る定義なのか**を確認する必要があります。BoTorch API を使う場合も、そのバージョンの定義を確認します。

#### この章で覚えておくこと：実装上の補足

- UCBはposterior meanにuncertainty由来のoptimism bonusを加える考え方
- betaはexploration strengthに関係するが、定義のparameterizationを確認する
- 固定betaの実用UCBとtheorem用GP-UCB scheduleを同一視しない
- LCBと符号反転によるmaximize conventionを混同しない
- qUCBはpointwise UCB上位q点ではなくjoint batch acquisition
- q=1 analytic UCBとMC qUCBではposterior requirementが異なり得る
- uncertainty scaleの意味とcalibrationはmodel familyに依存する
- robotorchanは標準UCB/qUCBを再実装せずBoTorch-native pathを使う

## 平均と標準偏差からscoreを組み立てる

BoTorchの`UpperConfidenceBound`は`AnalyticAcquisitionFunction`を継承します。
`forward()`は`_mean_and_sigma()`でmarginal momentsを求め、最大化では平均に`beta.sqrt() * sigma`を加えます。
Theoryで係数を $\sqrt{\beta}$ と書く場合に対応し、Straddleの`beta * std`と同じparameterizationではありません。
最小化向きは平均の符号へ反映されます。

MC版の`qUpperConfidenceBound`は`SampleReducingMCAcquisitionFunction`を継承し、
`_sample_forward()`がposterior sampleと平均からsample-wiseなUCB項を構成します。候補軸とsample軸の集約は上流の基底クラスが担当するため、
pointwiseな上位q点の選択とは異なります。

式の計算はBoTorchの[analytic.py][analytic]と[monte_carlo.py][mc]、
robotorchan側の条件照合は[registry](../../../src/robotorchan/acquisition/registry.py)にあります。
ローカルのUCBソルバへ置き換えているわけではありません。

[analytic]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/analytic.py
[mc]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/monte_carlo.py

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/26_standard_acquisition.ipynb)を参照してください。
