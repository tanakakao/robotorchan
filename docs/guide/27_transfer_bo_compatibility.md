# Transfer Bayesian Optimization：最終対応表（Phase 9）

この表は公開実装の機能範囲を整理する。実行例は
[Notebook 48](../../examples/notebooks/48_transfer_bayesian_optimization.ipynb)、
理論は[theory 26](../theory/26_transfer_bayesian_optimization.md)、
実装との対応は[guide 26](../guide/26_transfer_bayesian_optimization.md)を参照する。

## 対応マトリクス

| 手法 | Continuous | Mixed | q=1 | Sequential q>1 | Joint q>1 |
| --- | --- | --- | --- | --- | --- |
| Warm-start初期点選択 | 対応 | 対応 | 初期点1点 | 複数初期点選択（BOの逐次greedyとは異なる） | 非該当 |
| RGPE候補提案 | 対応 | 対応 | 解析的EI | posterior mean fantasyによるgreedy | moment-matched ensemble MC qEI |
| TAF候補提案 | 対応 | 対応 | 解析的TAF | posterior mean fantasyによるgreedy | target MC qEI + source mean最大改善量 |

- **Continuous**: BoTorch `optimize_acqf`。
- **Mixed**: `cat_dims` と `fixed_features_list` を併用し、
  BoTorch `optimize_acqf_mixed` を利用する。Warm-startでは
  `fixed_features_list` を用いる。
- **q=1**: `batch_strategy="joint"` を指定しても従来の解析的獲得関数を使う。
- **Sequential q>1**: 1点ずつ提案し、ターゲットGPを事後平均の仮想観測で条件付ける。
  ソースモデルと転移重みはバッチ内で固定する。
- **Joint q>1**: 同時最適化。RGPEはアンサンブルの近似Gaussian posterior、
  TAFはターゲットのMC qEIとソースの決定論的平均改善量を利用する。

## 公開エントリーポイント

| 目的 | エントリーポイント |
| --- | --- |
| 初期点選択 | `optim.transfer.warm_start.select_warm_start_points` |
| RGPE重み | `optim.transfer.rgpe.rgpe_weights` |
| RGPE候補 | `optim.transfer.rgpe_sequential.propose_rgpe_candidate` |
| TAF候補 | `optim.transfer.taf_sequential.propose_taf_candidate` |
| Joint TAF獲得関数 | `optim.transfer.taf_joint.qTransferAcquisitionFunction` |

## 検証範囲と残る制限

- RGPE・TAFのContinuous q=1、Mixed q=1、Continuous q=2、
  Mixed q=2について、既存の焦点を絞ったテストがある。
- Mixed × q=2のPhase 7テストは最適化器を置き換えた結合テストであり、
  **実最適化の性能・数値安定性を保証するものではない**。
- 対象はスカラー出力の単目的最適化。多目的・多出力転移BOは対象外。
- RGPEのランキング重み推定にはターゲットLOOの再学習コストがかかる。
- TAFの重みは明示指定で、負の転移を自動的に排除しない。
- Mixedのカテゴリは許可された割当を指定する必要がある。
- 候補提案は目的関数を評価しない。評価・観測追加は利用者の責任。
- Joint TAFのソース項は確率的qEIではなく平均値による近似。
- バッチ内の候補重複回避を一般的に保証するAPIではない。

このPhaseでは新しい最適化アルゴリズムや横断監査を追加せず、
実装・テスト・利用ガイドの対応範囲を確定する。
