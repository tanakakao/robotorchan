# Optimizer：探索解と数学的検証の違い

対応ガイド：[最適化基礎](../optimization/01_foundations.md)、[Candidate constraints](../optimization/05_constraints.md)、[高次元探索](../20_high_dimensional_search.md)、[TuRBO](../optimization/10_trust_region.md)。実装の正本：[旧Optimization Mapping](../../optimization.md)、独立数値参照の正本：[Verification Mapping](../../verification.md)。

## 検証項目の分離

| Property | 独立に検査する内容 | shape / smoke testだけで足りない理由 |
| --- | --- | --- |
| Candidate feasibility | bounds、fixed features、linear/nonlinear inequality/equality、許容tol | 候補が返っても制約違反の可能性がある |
| Joint q-batch | candidate shape `[q,d]`、acquisition valueがjoint utilityを表す | pointwise top-qはjoint optimumと異なる |
| Integer / categorical | legal integersとcategorical valueの保存 | dtypeだけではカテゴリの離散性を表さない |
| Embedding / reconstruction | 元座標のbounds、写像・inverse/latent契約、seed | clippingやhashingが最適値を変えうる |
| TuRBO state | success/failure、length、restart、batch単位の更新 | stateが存在するだけではtransitionの正しさは言えない |
| Hybrid handoff | global `[q,d]`をlocal initial conditionsへ引き渡す | 一点ずつの上位候補とは意味が違う |

Phase 9相当の旧検証記録にはoptimizerの複数の独立参照・契約テストが記述されている。現段階ではパス・数値tol・assertionを再照合していない項目を、verifiedの網羅表として掲載しない。対応節は[旧Verification Mapping](../../verification.md)へ残す。

各探索手法が数値的に候補を生成することと、その候補が大域最適解であることは同義ではない。NSGA-IIがvector Pareto目的を扱う一方、EHVI等はscalar acquisitionである点も区別する。


## 実装テスト本文との照合：候補の可行性

`tests/optim/integration/test_nonlinear_candidate_constraints_e2e.py`には、実際にモデル・獲得関数・optimizerを接続して最適化した候補を検査するテストがある。旧[Optimization Mapping](../../optimization.md)に記載された代表テスト名を実ソースと照合した。

| 対象 | 具体的なassertion | 得られる証拠と限界 |
| --- | --- | --- |
| `qLogExpectedImprovement`、q=1 | 制約`0.65 - x[0] >= 0`に対し返却点が`>= -1e-6`、返却値はfinite | 指定した1次元・float64・明示的3 restartでの可行性 |
| linear + nonlinear + fixed features | `x[0] >= 0.25 - 1e-6`、`0.8 - x[0] >= -1e-6`、`x[1]`は固定値0.4と一致 | 異なる入力制約の組合せを一例で確認 |
| q=2 interpoint | `(X[0]-X[1])²`の和が0.04以上、`candidate.shape == [2,1]` | **同時候補間**の距離制約を評価；独立のq=1を2回選ぶテストではない |
| `qLogNParEGO` | 2目的のmodel listをscalarized acquisitionへ接続し、非線形candidate制約を適用 | 指定した多目的scalarization経路のE2E証拠に限る |

最初の3例は明示的初期条件を与えている。合格する候補を返したことは、制約付き大域最適化の数学的証明ではない。また、同テスト群の`1e-6`は**テストが採用した可行性許容誤差**であり、全backendに共通の厳密保証値ではない。

## 実装テスト本文との照合：TuRBO state

`tests/optim/trust_region/test_turbo_strategy.py`は、数値的に意味のある改善に基づくstate transitionを次のように検査する。

- `TuRBOState(dim=20,batch_size=4)`の自動`failure_tolerance=5`、`dim=5,batch_size=2`では3。
- `length=0.4`、`success_tolerance=2`のstateへ連続成功を渡すと`length=0.8`、カウンタ0。
- `length=0.2`、`length_min=0.15`、`failure_tolerance=2`で2回失敗すると`length=0.1`となり`restart_triggered=True`。
- `best_value=100.0`、`relative_improvement=1e-3`では候補100.05を成功と数えない。数値best更新と**region拡大用の成功判定**が同義ではない。
- 局所boundsはpublic bounds内にclipされ、ARD重みの幾何平均正規化について別のassertionがある。

これらは特定した入力に対する**状態不変条件と実行経路の証拠**である。探索性能、ノイズ下の較正、すべての戦略による最適値の発見率まで検証したものではない。詳しくは[Trust-region実装ガイド](../optimization/10_trust_region.md)と元[Verification Mapping](../../verification.md)を参照する。
