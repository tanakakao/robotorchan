# Batch・非同期 benchmark simulation

同期基準法は `run_benchmark` です。`run_async_benchmark` で逐次（`max_concurrency=1`）と並列数を制限した非同期（`max_concurrency>1`）を比較します。

## 条件

- `BenchmarkExperimentConfig.q` は generator 1 呼び出し当たりの候補数、`max_concurrency` は同時 pending 評価数の上限です。
- callback は `(problem, completed_X, completed_Y, pending_X, q, generator)` を受け取り、学習に使えるのは完了観測だけです。
- duration callback は候補 `(d,)` から有限・厳密に正の**論理シミュレーション時間**を返します。
- 初期設計は時刻 0 で完了済みとし、新規評価予算を消費しません。
- event は投入順に記録し、`submission_index`、`completion_index`、`submitted_at`、`completed_at` を保持します。同時完了は投入順で解決します。
- `simulated_makespan` は実測の実時間ではありません。実 worker、cancel、worker 障害、checkpoint・再開は対象外です。

## 使用例

```python
import torch
from robotorchan.benchmarks import (
    BenchmarkExperimentConfig,
    run_async_benchmark,
)
from robotorchan.benchmarks.registry import BenchmarkProblemRegistry
from robotorchan.benchmarks.standard_problems import register_standard_problems

# Register the standard problems before resolving the experiment config.
registry = BenchmarkProblemRegistry()
register_standard_problems(registry)


def propose(problem, X, Y, pending, q, generator):
    bounds = problem.bounds.to(dtype=X.dtype, device=X.device)
    unit = torch.rand((q, problem.dimension), dtype=X.dtype, generator=generator)
    return bounds[0] + unit * (bounds[1] - bounds[0])


config = BenchmarkExperimentConfig(
    problem="branin",
    strategy="random-async",
    q=2,
    initial_points=4,
    evaluation_budget=10,
    seeds=(42,),
)
result = run_async_benchmark(
    config,
    propose,
    lambda candidate: 1.0,
    max_concurrency=2,
    registry=registry,
)[0]
print(result.completion_order, result.simulated_makespan)
```


比較では初期設計、予算、seed、duration model、必要に応じ concurrency を揃えてください。これは scheduling の意味をモデル化する基準法であり、実 worker や実際の遅延を再現するものではありません。
