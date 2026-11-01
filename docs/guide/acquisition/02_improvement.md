# Improvement / LogEI / PI：理論から実装へ

## この章で解決する問い

PIは改善する確率、EIは正の改善量の期待値。log-space formは目的自体を変えるのではなく数値的安定性を改善する。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/02_improvement.md)が正本であり、
この章は**数式の計算対象 → クラス構造 → 内部処理 → 近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

標準EI・LogEI・LogPIはBoTorch-native（Delegated）。
`make_qei_acquisition`はsemantic objective bridge（Extended）で、独立したEIアルゴリズムではない。

## 採用時の確認と注意事項

`best_f`の向き、標準化済みposteriorとの尺度、zero posterior variance、legacy EIとLogEIの数値安定性を照合する。

## 補足となる実装上の設計

#### References：実装上の補足

- Ament, S. et al. (2023), *Unexpected Improvements to Expected Improvement for Bayesian
Optimization*. arXiv:2310.20708. BoTorch's log-EI family follows this numerically robust
formulation.
- BoTorch documentation, *Acquisition Functions*, for the current analytic / Monte Carlo
acquisition interfaces and log-space EI variants.

## EIの解析式を計算するメソッド

BoTorchの`LogExpectedImprovement`は`AnalyticAcquisitionFunction`を継承します。
`forward()`は`_mean_and_sigma()`でGaussian marginalの平均・標準偏差を取得し、`_scaled_improvement()`で最良値との差を標準化します。
TheoryのEI式で $z=(\mu-f_{best})/\sigma$ に当たる量です。

`_log_ei_helper(z)`が標準正規密度・CDFに基づく改善量の対数を計算し、`log(sigma)`を加えて元のスケールを反映します。
小さなEIを通常の浮動小数点値として作ってからlogへ変換せず、負の大きなzでも勾配を保ちやすい形に分岐します。
`LogProbabilityOfImprovement.forward()`は同じ標準化差から正規log-CDFを使います。
この解析計算は[BoTorchのanalytic.py][analytic]が担当します。

robotorchanの[composition.py](../../../src/robotorchan/acquisition/composition.py)にある
`make_qei_acquisition()`は目的の方向や出力を解決してBoTorchのMC acquisitionを構成します。解析式とMC計算は別経路ですが、
いずれのEI式もこのbuilderが独自に計算してはいません。

[analytic]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/analytic.py

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/26_standard_acquisition.ipynb)を参照してください。
