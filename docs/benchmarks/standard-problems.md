# Standard analytic problems (Phase 6)

The following benchmark factories return fresh `BenchmarkProblem` instances.
All use **native minimization** directions; `to_maximization` and
`simple_regret` handle sign conversion for comparison metrics.

| Name | Dimension | Domain | Known minimum |
| --- | --- | --- | --- |
| `branin` | 2 | [-5, 10] × [0, 15] | ≈ 0.397887 |
| `hartmann6` | 6 | [0, 1]^6 | ≈ -3.322368 |
| `sphere3` | 3 | [-5, 5]^3 | 0 |

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

Explicit registration avoids hidden global registry side effects.
Problems are deterministic, unconstrained and continuous in this phase.
Constrained, mixed-variable, multiobjective and heterogeneous benchmark
families will be introduced separately.
