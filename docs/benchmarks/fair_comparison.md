# 公平比較の手順

`FairComparisonProtocol(first, second)` で戦略を対にして比較します。問題 ID、seed 集合と順序、初期設計数、追加評価予算、q、dtype、device を一致させ、戦略名は異なるものにします。

実行前は設定の公平性を確認します。両戦略を `run_benchmark` で実行後、`validate_trajectories(first_runs, second_runs)` で seed 順、総評価数、初期 X・観測 Y・真値 Y・制約・費用の厳密一致を検証します。

```python
protocol = FairComparisonProtocol(random_config, sobol_config)
random_runs = run_benchmark(random_config, random_candidates, registry=registry)
sobol_runs = run_benchmark(sobol_config, sobol_candidates, registry=registry)
protocol.validate_trajectories(random_runs, sobol_runs)
```


初期設計や予算の取り違えを防ぐもので、学習時間、ハードウェア利用率、獲得関数最適化の労力、初期設計後のノイズ系列、hyperparameter 予算の同一性は保証しません。それらは別途報告してください。検証済み軌跡から指標を抽出し、`compare_paired_curves` で対応差を分析できます。
