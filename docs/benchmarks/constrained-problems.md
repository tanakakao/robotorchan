# Constrained benchmark problems (Phase 7)

These problems exercise outcome constraints in the `BenchmarkProblem` contract.
Feasibility is **g(X) >= 0**, and infeasible observations are not eligible for
truth-based simple regret. If no evaluated point is feasible, regret is `+inf`.

| Problem | Direction | Feasible region | Known optimum |
| --- | --- | --- | --- |
| `constrained_quadratic` | Maximize | x0 + x1 <= 1 on [0,1]^2 | -0.18 at (0.5, 0.5) |
| `constrained_annulus` | Minimize | 0.25 <= x0² + x1² <= 1 | 0.25 on inner boundary |

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

These are deterministic **outcome constraints**, not candidate-space
restrictions. Candidate generation may propose infeasible points; those
evaluations remain in the history, but must be excluded from feasible regret.
