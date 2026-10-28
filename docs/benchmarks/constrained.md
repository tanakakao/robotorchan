# 制約付き benchmark 問題

すべての制約は `g(X) >= 0` を実行可能とし、残差の形状は `(..., q, c)` です。負は違反です。候補で評価する出力制約であり、獲得関数 optimizer の入力領域制約とは異なります。

| 問題 | 実行可能領域 | 制約数 | 既知の最適値 |
| --- | --- | ---: | ---: |
| constrained_quadratic | x0+x1 <= 1 | 1 | -0.18（最大化） |
| constrained_annulus | 0.5 <= radius <= 1 | 2 | 0.25（最小化） |
| constrained_disconnected | 中心 (+/-0.75, 0)、半径 0.25 の 2 円盤 | 1 | 0（最大化） |
| constrained_narrow_band | 0.95 <= x0+x1 <= 1.05 | 2 | -0.15125（最大化） |

disconnected は孤立した領域間の探索、narrow band は単位正方形の狭い部分の発見を調べます。いずれも既知の実行可能最適値を持ち、真値による simple regret を確認できます。実行可能な評価がなければ regret は無限大です。

`register_constrained_problems(registry)` で 4 問題を登録し、選んだ戦略で `run_benchmark` を実行します。

## 使用例と評価の意味

```python
from robotorchan.benchmarks import (
    BenchmarkExperimentConfig,
    BenchmarkProblemRegistry,
    random_candidates,
    register_constrained_problems,
    run_benchmark,
)

registry = BenchmarkProblemRegistry()
register_constrained_problems(registry)
config = BenchmarkExperimentConfig(
    problem="constrained_quadratic",
    strategy="random",
    evaluation_budget=20,
    q=3,
)
results = run_benchmark(config, random_candidates, registry=registry)
```


すべて決定的な出力制約問題です。違反候補も提案・評価でき、履歴に残りますが、実行可能 regret の計算からは除外します。実行可能点がなければ `+inf` です。`g(X) >= 0` は optimizer の候補空間を直接制限しません。
