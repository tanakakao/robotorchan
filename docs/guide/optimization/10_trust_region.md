# TuRBO Trust Region：状態遷移と局所探索

## 理論・実装の対応

[理論のTrust region](../../theory/optimization/10_trust_region.md)と
[高次元探索のガイド](../20_high_dimensional_search.md)に対応する。TuRBOは`optimizer="turbo"`という12番目のbackendではなく、状態を保
持して局所boundsを更新する**search strategy**である。
実装は`src/robotorchan/optim/trust_region/turbo.py`の`TuRBOState`、`update_turbo_state`、
`restart_turbo_state`、`TuRBOStrategy`へ分解される。

## 1. 更新の入力：acquisitionではなく完了観測

`update_turbo_state(state, values, *, state_values=None, relative_improvement=1e-3)`は完了した1
batchの観測値`values`を受け取る。値の個数は`state.batch_size`と一致しなければならず、入力shapeは`[q]`または`[q,1]`である。
未完了の`X_pending`や獲得関数scoreを観測Yとして入力しない。

オプションの`state_values`はdenoised posteriorなど、局所領域を広げる判断に使う別のscalar utilityである。
このとき`best_value`は`state_values`の最大値を追跡し、`observed_best_value`は生の`values`の最大値を追跡する。両者を同じ観測量として扱わない。

## 2. 成功・失敗とregion lengthの数式

意思決定に使うbatch最大値を`b`、stateの最良判断値を`b_best`とし、`relative_improvement=r`とする。初回など`b_best`が有限でなければ成功とみなす。
それ以外は

```text
success ⇔ b > b_best + r * max(1, abs(b_best))
```

を評価する。成功時は`success_counter += 1`、`failure_counter = 0`、失敗時は逆に`failure_counter += 1`、
`success_counter = 0`。成功回数が`success_tolerance`に達すると`length = min(2 * length, length_max)`、
失敗回数が`failure_tolerance`に達すると`length = length / 2`。閾値到達後、該当counterは0へ戻る。

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

`restart_turbo_state`は`restart_triggered=True`が必要で、成功/失敗counter、flagをリセットし`restart_count`を1増やす。
既存の`best_value`と`observed_best_value`は保持されるので、「global bestごとゼロに戻る」わけではない。

`generate_turbo_restart_center(bounds,seed=...)`はpublic-spaceのbounds上にSobol centerを生成し、
`[2,d]`の有限boundsかつ各成分`lower < upper`を検証する。ARD由来の`dimension_weights`を使う場合は正で有限の重みを幾何平均1に正規化し、
局所boundsはglobal boundsにclipする。特定のモデルのlengthscale抽出と幾何計算は別の責務である。

## 4. 状態・幾何・候補生成の責務

- **State transition**
  テストで確認する性質：成功でexpand、失敗でshrink、restart trigger、batch-sizeチェック
  言えないこと：全探索履歴の最適性

- **Coordinate geometry**
  テストで確認する性質：ARDの正規化と公開空間の局所bounds clip
  言えないこと：効果次元を正しく発見した保証

- **Candidate proposal**
  テストで確認する性質：`SearchResult.candidates`のq/d shape、bounds・feasibility、mixed dispatch
  言えないこと：ベンチマークregretの改善

- **Observation update**
  テストで確認する性質：実評価値と`state_values`の責務分離
  言えないこと：noise不確実性の厳密な最適制御

## 補足となる実装上の設計

#### 10.8 Center update：実装上の補足

`TuRBOStrategy.update_state(..., candidates=X)` に candidate を渡した場合、
meaningful improvement を生んだ batch 内 candidate が新しい center になります。

#### 10.20 Restart center：実装上の補足

`TuRBOStrategy.restart()` は明示的な center を受け取るか、
指定がなければ global bounds 内の scrambled Sobol point を新しい center として生成します。

#### 10.21 Stateful BO loop：実装上の補足

fit surrogate -> build acquisition -> strategy.optimize(...) or strategy.thompson_sample(...) ->
evaluate objective -> strategy.update_state(...) -> append observations -> refit surrogate

#### 10.21 Stateful BO loop：実装上の補足

strategy.restart(...)

## 5. 既定値と呼び出し側の責務

failure toleranceの自動既定値は第2節の`ceil(max(4 / batch_size, dim / batch_size))`である。
Thompson candidate poolでの既定perturbation probabilityは`min(20 / dim, 1)`で、全次元が未変更となった候補は少なくとも1次元を変更する。
これらは異なる役割の設定であり、成功・失敗判定の閾値と混同しない。

`TuRBOState`は第2節の長さ・counter・best値・restart情報と、入力次元・batch sizeを保持する。モデルの学習、実験の実行、観測の追加は呼び出し側が担当する。
restartは通常の観測追加とは別の状態遷移として扱う。詳細は[TuRBO利用ガイド](../../optimization/turbo.md)を参照する。

## 状態と数値探索を分けるクラス構造

[turbo.py](../../../src/robotorchan/optim/trust_region/turbo.py)の
`TuRBOStrategy`は`SearchStrategy`を継承し、`TuRBOState`を保持します。`optimize()`が局所bounds内のdispatcherへ候補探索を委譲し、
`update_state()`が完了観測を`update_turbo_state()`へ渡します。`turbo_trust_region_bounds()`が領域の幾何を、
`generate_turbo_thompson_choices()`が局所poolの生成を担当します。

状態遷移・幾何・pool生成はローカル実装であり、GP fittingの内部ではありません。勾配による候補生成を選んだときの数値探索は、指定されたbackendが担当します。

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/34_turbo_trust_region.ipynb)を参照してください。
