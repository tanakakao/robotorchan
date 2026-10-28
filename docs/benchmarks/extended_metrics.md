# 追加の評価指標

[基本指標](metrics.md)の regret・hypervolume・実行可能性を補完します。

| 指標 | API | 良い方向 |
| --- | --- | --- |
| 台形則による曲線面積 | `area_under_curve(curve, budgets)` | 曲線に依存 |
| 現時点の最良実行可能値 | `best_feasible_value_curve(trajectory, maximize=True)` | 最大化では高い方 |
| 最初の joint 実行可能評価 | `first_feasible_evaluation(constraints)` | 小さい方 |
| 二値負対数尤度 | `log_loss(labels, probabilities)` | 小さい方 |
| 正例確率の較正誤差 | `probability_calibration_error(labels, probabilities)` | 小さい方 |

面積は評価数または累積費用の厳密に増加する軸を受け取ります。無限大は積分へ伝播するため、実行可能点がない run の扱いを比較前に決めてください。

最良実行可能値は最初の実行可能点まで NaN です。初回実行可能性は初期設計を含む 1 始まりの index で、見つからなければ `None` です。

log loss の確率 clamp は数値安定化のためです。較正誤差は正例確率を [0,1] の等幅 bin に分け、1 は最後に含めます。多 class の confidence-based ECE ではありません。

Active Learning の比較では固定 reference 集合、予算、予測時点を揃え、[公平性契約](../../docs-internal/contracts/benchmarks/comparative-fairness-protocol.md)と初期・新規評価の軸を一致させます。これらの関数は学習・観測獲得を実行しません。
