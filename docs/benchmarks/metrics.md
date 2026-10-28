# Benchmark 指標

ノイズ観測ではなく決定的な真値を使います。出力制約は `g(X) >= 0` です。

| 指標 | 対象 | 意味 |
| --- | --- | --- |
| `feasibility_rate` | 任意の目的数 | 実行可能な評価の割合 |
| `cumulative_feasibility_rate` | 任意の目的数 | prefix ごとの割合 |
| `simple_regret_curve` | 単一目的・既知の最適値 | 最良の実行可能真値の regret |
| `hypervolume_2d` | 最大化 2 目的 | 支配された領域の厳密な和集合面積 |
| `hypervolume_curve` | 2 目的 | 各評価後の実行可能 hypervolume |

最初の実行可能点までは `simple_regret_curve` が `+inf` を返します。制約なしでは全点を実行可能とします。

hypervolume は目的と reference point を最大化の向きへ揃えてから計算し、支配点・重複点を扱います。3 目的以上、seed 間統計、解析的 Pareto front に対する hypervolume regret はこの 2D 実装の対象外です。

```python
from robotorchan.benchmarks import (
    hypervolume_curve,
    simple_regret_curve,
)

# Compute one curve per BenchmarkTrajectory and its BenchmarkProblem.
```
