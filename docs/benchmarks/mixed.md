# 混合変数 benchmark 問題

カテゴリは整数符号の名義ラベルであり、順序距離を表しません。候補 generator は bounds 内の合法整数符号を返す必要があります。問題契約は one-hot を要求しません。

| 問題 | 入力 | 目的 | 制約 |
| --- | --- | --- | --- |
| mixed_quadratic | 連続・整数・カテゴリ | 最小化、最適値 0 | なし |
| categorical_switch | 連続・カテゴリ | 最大化、最適値 1 | なし |
| mixed_process_yield | 温度・整数 cycle 数・recipe | 最大化、最適値 1 | recipe 依存の温度上限、cycles <= 4 |
| mixed_category_interaction | 連続・カテゴリ 2 個 | 最小化、最適値 0 | なし |

Process yield は recipe ごとに応答中心と温度上限が異なり、実行可能最適点は `(0.7, 3, 1)` です。カテゴリと連続の出力制約の相互作用を調べます。

Category interaction は連続応答中心とカテゴリ罰則の非加算的な表を持ち、最適点は `(0.85, 1, 1)` です。2 つの名義変数が共同で目的の形を決めます。

基準法の `sobol_initial_design` と `random_candidates` は離散座標を丸めます。専用の混合最適化には同じ `(q, d)` 契約の互換 generator を使ってください。`register_mixed_problems(registry)` で登録します。

## 実行例

```python
from robotorchan.benchmarks import (
    BenchmarkExperimentConfig,
    BenchmarkProblemRegistry,
    random_candidates,
    register_mixed_problems,
    run_benchmark,
)

registry = BenchmarkProblemRegistry()
register_mixed_problems(registry)
config = BenchmarkExperimentConfig(
    problem="mixed_quadratic",
    strategy="random",
    seeds=(0, 1, 2),
    initial_points=8,
    evaluation_budget=24,
    q=3,
)
trajectories = run_benchmark(config, random_candidates, registry=registry)
```


`mixed_quadratic` の最適点は `(0.8, 2, 1)`、`categorical_switch` は `(0.7, 1)`、最適値はそれぞれ 0（最小化）と 1（最大化）です。0・1・2 などの符号に順序距離はありません。基準法の丸めは動作確認用で、離散水準の一様抽出や専用 optimizer の品質を保証しません。
