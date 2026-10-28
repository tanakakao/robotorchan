# ノイズのある benchmark 問題

2 次元・単一目的の最大化問題で、決定的真値とノイズ観測を分離します。

| 問題 | 真値 | 観測ノイズ |
| --- | --- | --- |
| noisy_quadratic | (0.35,0.35) からの二乗距離の負 | Gaussian、標準偏差 0.1 |
| heteroscedastic_quadratic | 同じ二次関数 | Gaussian、標準偏差 0.02 + 0.18*x0 |

ともに最適点 (0.35,0.35)、最適値 0 です。評価指標には `evaluate_truth(X)`、学習には `evaluate_observation(X)` を使います。runner は `Y_truth`・`Y_observed` を保存し、候補生成には観測値のみを渡します。

factory は隔離したローカル seed の観測 generator を作ります。新規 instance は同じノイズ系列を再現し、実験 seed ごとに新しい問題を作って状態の漏れを防ぎます。組み込み factory の観測 seed は固定で、候補設計の run seed とは別です。

別の明示的な `SeededObservationNoise` は run-local の系列と `paired`・`independent` の戦略間対応を提供します。組み込み factory に暗黙には適用されません。使用した観測機構を記録し、[実測契約](../../docs-internal/contracts/benchmarks/empirical-execution-contract.md)に従ってください。対応乱数系列でも、適応的に異なる候補経路間の点ごとの対応は保証しません。

`register_noisy_problems(registry)` で登録します。
