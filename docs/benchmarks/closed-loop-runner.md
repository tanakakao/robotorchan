# Closed-loop Benchmark Runner (Phase 5)

The runner connects the Phase 2 problem contract, Phase 3 registry, and
Phase 4 experiment configuration. It evaluates an initial Sobol design, then
repeatedly requests candidates until the **additional evaluation** budget is
exhausted. The final batch may be smaller than `q`.

```python
from robotorchan.benchmarks import (
    BenchmarkExperimentConfig,
    random_candidates,
    run_benchmark,
)

config = BenchmarkExperimentConfig(
    problem="quadratic",
    strategy="random",
    seeds=(0, 1, 2),
    initial_points=8,
    evaluation_budget=20,
    q=3,
)
# Register "quadratic" with register_problem(...) before running.
trajectories = run_benchmark(config, random_candidates)
```

The callback signature is `(problem, X, Y_observed, q, generator) -> candidates`.
A user-defined callback can connect any BoTorch acquisition/optimizer stack
without changing the runner. The `strategy` string records experiment intent;
the caller supplies the matching callback.

Each `BenchmarkTrajectory` stores evaluated X, observed Y, deterministic
truth Y, constraint residuals, evaluation costs, and the initial-design
boundary. `evaluation_count` excludes the initial design, while
`cumulative_cost` includes it. Truth values must not be supplied to the
candidate callback; only observed values are passed.

Initial Sobol designs are paired by problem and seed across strategies.
Random-search candidates use a per-seed torch generator, avoiding mutation
of the global torch RNG. The runner does not yet serialize trajectories or
compute regret curves; those belong to subsequent phases.

Observation callbacks using global randomness remain the responsibility of
the caller. Full reproducibility of noisy observations requires a seeded
observation model; this will be addressed in a later phase.
