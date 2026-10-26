# TuRBO Trust Region：状態遷移と局所探索

## 理論・実装の対応

[理論のTrust region](../../../../theory/optimization/10_trust_region.md)と[高次元探索のガイド](../20_high_dimensional_search.md)に対応する。TuRBOは`optimizer="turbo"`という12番目のbackendではなく、状態を保持して局所boundsを更新する**search strategy**である。実装は`src/robotorchan/optim/trust_region/turbo.py`の`TuRBOState`、`update_turbo_state`、`restart_turbo_state`、`TuRBOStrategy`へ分解される。

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

代表的な契約テストは`tests/optim/trust_region/test_turbo_strategy.py`。旧[Optimization Mapping](../../optimization.md)のassertion evidenceと[Verification Mapping](../../verification.md)に既存の数値参照記録がある。CI成功と特定state invariantの検査から、TuRBOの総合的な収束や一般化性能を証明したとは扱わない。旧原文は台帳が検収されるまで保全する。
