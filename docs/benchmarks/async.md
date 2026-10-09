# Phase 14 — Batch / Async benchmark simulation

The existing `run_benchmark` remains the synchronous baseline. Use
`run_async_benchmark` to compare sequential (`max_concurrency=1`) and
capacity-limited asynchronous evaluations (`max_concurrency>1`).

## Contract

- `BenchmarkExperimentConfig.q` limits the number of candidates requested per
  generator call. `max_concurrency` limits simultaneous pending evaluations.
- The candidate callback receives `(problem, completed_X, completed_Y,
  pending_X, q, generator)`. Only completed observations are available for fitting.
- A duration callback receives one candidate `(d,)` and returns a finite,
  strictly positive duration in **simulated logical-time units**.
- Initial design observations are assumed complete at simulated time zero.
  They do not consume the new-evaluation budget.
- Events are recorded in submission order, with `submission_index`,
  `completion_index`, `submitted_at` and `completed_at`.
  Tied completion times are resolved in submission order.
- `simulated_makespan` is **not** measured wall-clock time. Real execution,
  cancellations, worker failures and checkpoint/resume are out of scope.

## Example

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
    problem="branin", strategy="random-async", q=2,
    initial_points=4, evaluation_budget=10, seeds=(42,),
)
result = run_async_benchmark(
    config, propose, lambda candidate: 1.0,
    max_concurrency=2, registry=registry,
)[0]
print(result.completion_order, result.simulated_makespan)
```

For fair comparisons, hold the initial design, budget, seed, duration model
and concurrency fixed as appropriate. The baseline models the scheduling
contract; it does not yet simulate true worker execution or real-time latency.
