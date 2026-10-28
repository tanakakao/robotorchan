# 単一目的 benchmark

標準 registry は決定的な最小化問題を提供します。

| 問題 | 次元 | Bounds | 既知の最適値 |
| --- | ---: | --- | ---: |
| Ackley2 | 2 | [-5,5] | 0 |
| Branin | 2 | [-5,10] × [0,15] | 約 0.397887 |
| Hartmann6 | 6 | [0,1] | 約 -3.322368 |
| Rosenbrock2 | 2 | [-2,2] | 0 |
| Sphere3 | 3 | [-5,5] | 0 |

元の最小化値を公開し、`to_maximization` で下流の獲得関数・指標用に方向を変換します。解析最小値と regret のテストは最適化性能とは独立です。

`register_standard_problems(registry)` で登録し、比較では `BenchmarkExperimentConfig` の seed と初期設計を揃えます。統計的性能の主張には公平比較と多 seed の評価が必要です。

## 最小実行例

factory は新しい `BenchmarkProblem` を返し、明示的な登録により import の global 副作用を避けます。決定的・制約なし・連続問題の例です。

```python
from robotorchan.benchmarks import (
    BenchmarkExperimentConfig,
    BenchmarkProblemRegistry,
    random_candidates,
    register_standard_problems,
    run_benchmark,
)

registry = BenchmarkProblemRegistry()
register_standard_problems(registry)
config = BenchmarkExperimentConfig(
    problem="branin",
    strategy="random",
    seeds=(0, 1, 2),
    initial_points=8,
    evaluation_budget=20,
    q=3,
)
trajectories = run_benchmark(config, random_candidates, registry=registry)
```


実行例であり、最適化器の優越性の証拠ではありません。
