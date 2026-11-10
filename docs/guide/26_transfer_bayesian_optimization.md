# Transfer Bayesian Optimization：理論と実装の対応

[理論章](../theory/26_transfer_bayesian_optimization.md)で扱うWarm-start、
RGPE、TAFの処理経路を説明する。使用コードと実験結果はNotebookに集約する。

## 実装の対応表

| 理論上の役割 | robotorchan実装 | BoTorchとの関係 |
| --- | --- | --- |
| Warm-start初期点選択 | `optim.transfer.warm_start.select_warm_start_points` | ソース観測点の選択とBoTorch Sobol補完 |
| RGPEランキング重み | `optim.transfer.rgpe.rgpe_weights` | ソースとターゲットの損失比較を独自実装 |
| ターゲットLOO | `optim.transfer.loo.target_loo_predictions` | `SingleTaskGP`を分割ごとに再学習 |
| RGPE予測分布 | `optim.transfer.rgpe_model.RGPEModel` | BoTorch `Model`のposteriorを実装 |
| RGPE逐次候補 | `optim.transfer.rgpe_sequential.propose_rgpe_candidate` | BoTorch EIと`optimize_acqf` |
| TAF獲得関数 | `optim.transfer.taf.TransferAcquisitionFunction` | BoTorch `AnalyticAcquisitionFunction`を継承 |
| TAF逐次候補 | `optim.transfer.taf_sequential.propose_taf_candidate` | `SingleTaskGP`学習と`optimize_acqf` |

## Warm-startの内部経路

`select_warm_start_points`はソース観測またはモデルの予測に基づいて
ターゲットの初期評価候補を選択し、不足分をSobol点で補う。
ソースの目的値はターゲット観測として流用せず、候補点をターゲットで評価する。

## RGPEの内部経路

ターゲット観測からGPを学習し、LOO予測を生成する。
`rgpe_weights`はソースモデルの順位損失とターゲットのLOO順位損失を
比較して重みを求める。`RGPEModel`は各posteriorの平均・共分散と
モデル間の平均差から、モーメント整合したGaussian posteriorを返す。
このposteriorをEIに渡し、BoTorchの候補最適化を利用する。

## TAFの内部経路

`TransferAcquisitionFunction`はターゲットモデルのEIと、
ソースモデルのposterior meanから計算する正の改善量を加重和にする。
重みは呼び出し側が明示的に指定し、内部でRGPE重みを自動推定しない。
`propose_taf_candidate`はターゲットGPを毎回再学習し、
獲得関数最適化によって1点を返す。

## 実装上の境界

- 逐次関数は**目的関数を評価しない**。新しい観測の追加は呼び出し側が行う。
- 対象は連続変数・単目的・`q=1`。Mixed、複数候補同時提案は対象外。
- RGPEではLOOモデルの繰り返し学習に計算コストがかかる。
- TAFは固定重みのため、負の転移を自動検知・抑制するとは限らない。
- 性能の比較は`examples/benchmarks/transfer_bo.py`で行う。
