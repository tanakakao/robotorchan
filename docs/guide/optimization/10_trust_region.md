# TuRBO Trust Region：状態遷移と局所探索

## 理論・実装の対応

[理論のTrust region](../../theory/optimization/10_trust_region.md)と[高次元探索のガイド](../20_high_dimensional_search.md)に対応する。TuRBOは`optimizer="turbo"`という12番目のbackendではなく、状態を保持して局所boundsを更新する**search strategy**である。実装は`src/robotorchan/optim/trust_region/turbo.py`の`TuRBOState`、`update_turbo_state`、`restart_turbo_state`、`TuRBOStrategy`へ分解される。

## 1. 更新の入力：acquisitionではなく完了観測

`update_turbo_state(state, values, *, state_values=None, relative_improvement=1e-3)`は完了した1 batchの観測値`values`を受け取る。値の個数は`state.batch_size`と一致しなければならず、入力shapeは`[q]`または`[q,1]`である。未完了の`X_pending`や獲得関数scoreを観測Yとして入力しない。

オプションの`state_values`はdenoised posteriorなど、局所領域を広げる判断に使う別のscalar utilityである。このとき`best_value`は`state_values`の最大値を追跡し、`observed_best_value`は生の`values`の最大値を追跡する。両者を同じ観測量として扱わない。

## 2. 成功・失敗とregion lengthの数式

意思決定に使うbatch最大値を`b`、stateの最良判断値を`b_best`とし、`relative_improvement=r`とする。初回など`b_best`が有限でなければ成功とみなす。それ以外は

```text
success ⇔ b > b_best + r * max(1, abs(b_best))
```

を評価する。成功時は`success_counter += 1`、`failure_counter = 0`、失敗時は逆に`failure_counter += 1`、`success_counter = 0`。成功回数が`success_tolerance`に達すると`length = min(2 * length, length_max)`、失敗回数が`failure_tolerance`に達すると`length = length / 2`。閾値到達後、該当counterは0へ戻る。

| `TuRBOState`の主要項目 | 初期値・意味 |
| --- | --- |
| `length`、`length_min`、`length_max` | 0.8、`0.5**7`、1.6。局所幅の上下限 |
| `success_tolerance` | 3回の成功を閾値とする |
| `failure_tolerance` | `ceil(max(4 / batch_size, dim / batch_size))`が自動既定値 |
| `best_value`、`observed_best_value` | 両者の初期値は`-inf` |
| `restart_triggered` | 更新で`length < length_min`になった場合にTrue |
| `restart_count` | 実際のrestart時に増加する |

この条件は`update_turbo_state`の実装で定義される。posterior varianceが下がったからという理由だけで成功と判定しない。

## 3. Restartと局所bounds

`restart_turbo_state`は`restart_triggered=True`が必要で、成功/失敗counter、flagをリセットし`restart_count`を1増やす。既存の`best_value`と`observed_best_value`は保持されるので、「global bestごとゼロに戻る」わけではない。

`generate_turbo_restart_center(bounds,seed=...)`はpublic-spaceのbounds上にSobol centerを生成し、`[2,d]`の有限boundsかつ各成分`lower < upper`を検証する。ARD由来の`dimension_weights`を使う場合は正で有限の重みを幾何平均1に正規化し、局所boundsはglobal boundsにclipする。特定のモデルのlengthscale抽出と幾何計算は別の責務である。

## 4. 検証すべきレイヤー

| レイヤー | テストで確認する性質 | 言えないこと |
| --- | --- | --- |
| State transition | 成功でexpand、失敗でshrink、restart trigger、batch-sizeチェック | 全探索履歴の最適性 |
| Coordinate geometry | ARDの正規化と公開空間の局所bounds clip | 効果次元を正しく発見した保証 |
| Candidate proposal | `SearchResult.candidates`のq/d shape、bounds・feasibility、mixed dispatch | ベンチマークregretの改善 |
| Observation update | 実評価値と`state_values`の責務分離 | noise不確実性の厳密な最適制御 |

代表的な契約テストは`tests/optim/trust_region/test_turbo_strategy.py`。旧[Optimization Mapping](../../../.github/project-records/theory-guide/mapping/optimization.md)のassertion evidenceと[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)に既存の数値参照記録がある。CI成功と特定state invariantの検査から、TuRBOの総合的な収束や一般化性能を証明したとは扱わない。旧原文は台帳が検収されるまで保全する。

## 5. 観測結果から局所領域を拡大・縮小する実行例

次は`TuRBOState`を作成し、**完了した観測値**に基づいて長さを更新する。
`success_tolerance=2`で2回成功すると幅を2倍にし、
`failure_tolerance=2`で2回失敗すると幅を半分にする。

```python
import torch
from robotorchan.optim.trust_region.turbo import TuRBOState, update_turbo_state

state = TuRBOState(
    dim=2,
    batch_size=1,
    length=0.4,
    success_tolerance=2,
    failure_tolerance=2,
)
state = update_turbo_state(state, torch.tensor([0.5], dtype=torch.double))
assert state.length == 0.4
assert state.success_counter == 1
state = update_turbo_state(state, torch.tensor([0.7], dtype=torch.double))
assert state.length == 0.8
assert state.success_counter == 0
state = update_turbo_state(state, torch.tensor([0.6], dtype=torch.double))
assert state.length == 0.8
assert state.failure_counter == 1
state = update_turbo_state(state, torch.tensor([0.6], dtype=torch.double))
assert state.length == 0.4
assert state.failure_counter == 0
assert state.best_value == 0.7
assert state.observed_best_value == 0.7
```

`update_turbo_state`が受け取る`values`は獲得関数のscoreではなく、
実際に評価した目的値である。ここでは閾値に対する成功・失敗の
状態遷移のみを確認し、探索戦略を走らせたregretやrestart性能は検証しない。

## 理論章から分離した実装上の条件

#### 10.8 Center update：実装上の補足

`TuRBOStrategy.update_state(..., candidates=X)` に candidate を渡した場合、
meaningful improvement を生んだ batch 内 candidate が新しい center になります。

#### 10.20 Restart center：実装上の補足

`TuRBOStrategy.restart()` は明示的な center を受け取るか、
指定がなければ global bounds 内の scrambled Sobol point を新しい center として生成します。

#### 10.21 Stateful BO loop：実装上の補足

    fit surrogate
      -> build acquisition
      -> strategy.optimize(...) or strategy.thompson_sample(...)
      -> evaluate objective
      -> strategy.update_state(...)
      -> append observations
      -> refit surrogate

#### 10.21 Stateful BO loop：実装上の補足

    strategy.restart(...)

## 数値 API と利用上の補足

default failure tolerance は

default perturbation probability は

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

`TuRBOState` は少なくとも次を保持します。
