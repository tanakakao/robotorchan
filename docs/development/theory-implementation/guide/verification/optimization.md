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


## Phase 9独立参照：Sampling optimizerのpool内最大値

`tests/optim/backends/test_phase9_sampling_optimizer_reference.py`を直接照合した。`optimizer="random"`または`"sobol"`、`seed=29`、`num_samples=128`、`q=2`で、負の二次形式`-(X-target)^2`を候補2点・2次元のbatchに対して合計したscoreを用いる。公開optimizerが返す`[2,2]`候補と**同じseedで独立生成したpool中のargmax候補**は完全一致（`atol=rtol=0`）、valueは`1e-12`で一致し、候補はboundsに収まる。真の連続最大値0への収束を証明するものではない。

## Phase 9独立参照：TuRBOのARD幾何

`tests/optim/trust_region/test_phase9_turbo_geometry_reference.py`は3次元の非等幅bounds、境界付近のcenter、`weights=[0.5,1.0,4.0]`と`length=0.2`に対して、次のscalar referenceを組み立てる。

```text
w_normalized = w / geometric_mean(w)
half_width = 0.5 * length * (upper - lower) * w_normalized
lower_TR = max(lower, center - half_width)
upper_TR = min(upper, center + half_width)
```

`turbo_trust_region_bounds`がこの値に`atol=rtol=1e-12`で一致し、境界でclippingされることと、全weightsを7倍しても同じ領域になることをassertする。**既学習GPのARD lengthscaleの妥当性**や最適化性能は確認していない。

## Phase 9独立参照：Constraint residualとfeasibility-first

`tests/optim/constraints/test_phase9_constraint_violation_reference.py`は3つの`q=2`、`d=2`候補batchに対し、intrapoint/interpointの線形不等式、線形等式、per-point/jointの非線形不等式を組み合わせる。`equality_tolerance=0.05`として、**個々のPython scalar演算で独立に計算した**非負の違反量合計を`candidate_constraint_violation`と`1e-12`で比較する。intrapoint線形不等式はq軸の最小残差を使い、per-point非線形違反はq内で加算する。`candidate_is_feasible`は違反量ゼロと一致することを確認する。

別テストではobjective値`[0.5,100,0.9,200]`、違反量`[0,0.2,0,0.1]`に対する`feasibility_first_ranks=[3,1,4,2]`をassertする。高いobjective値を持つ不可行点より、可行点を上位にする**順位付け算術**の証拠であり、数値optimizer自体が制約可行な候補を必ず返す保証ではない。

これらの独立参照は旧[Verification Mapping](../../verification.md)のPhase 9区間と照合したものだが、一般のglobal optimum、学習済みARDの正しさ、全制約backendの厳密可行性は対象外とする。


## Phase 9独立参照：TuRBO state transition

`tests/optim/trust_region/test_phase9_turbo_state_reference.py`はfloat64の固定観測値で状態更新を照合する。独立したGP fittingや乱数samplingは含まない。

- `length=0.9`、`length_max=1.6`、`success_tolerance=3`の状態に`[0.1,0.2,0.3]`を順に渡す。最初の2成功で長さは0.9のまま、3回目は2倍にするものの上限1.6にclipされ、success counterは0へ戻る。
- `length=0.15`、`length_min=0.1`、`failure_tolerance=2`で`[1.0,1.5]`を更新すると、最良値2.0を超えないため2連続失敗となり、長さは0.075に半減、`restart_triggered=True`。続く`restart_turbo_state(length=0.8)`ではrestart回数が1増え、best valueとobserved best valueは2.0のまま保存される。
- `best_value=1.0`の状態に観測3.0と`state_values=0.5`を同時に渡すと、decision utility上は失敗扱い（failure counter=1、best value=1.0）だが、生の`observed_best_value`は3.0となる。**意思決定の改善判定と観測最大値の記録を分離する**契約を明示する。

これは状態遷移の限定的なreferenceであり、ノイズ付き探索の性能やrestart centerの生成分布を立証しない。

## Phase 9 negative contract：PyTorch勾配optimizerとcandidate制約

`tests/optim/backends/test_phase9_torch_constraint_contract.py`は`optim/backends/torch.py::optimize_acqf_torch`を直接呼び、`optimizer`が`adam`、`adamw`、`sgd`の各値について、線形不等式・線形等式・非線形不等式の計9組合せが`ValueError("does not support candidate constraints")`で拒否されることをassertする。入力はfloat64、`q=1`、`num_restarts=1`、`raw_samples=4`である。

この拒否は**未対応のcandidate constraintを無視して最適化する事故を防ぐnegative contract**であり、制約付きAdam等が実行できる、またはその候補の可行性を保証する、という主張とは正反対である。optimizerの勾配収束自体もこのテストからは評価できない。

今回の直接照合は、旧[Verification Mapping](../../verification.md)のPhase 9独立reference五群を対象とする。歴史的なPhase 9 exit判定は**代表的propertyに限定**される。旧文書の他の断定や全backendの数学的同値性まで検収済みとはしない。
