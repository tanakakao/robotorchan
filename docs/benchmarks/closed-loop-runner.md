# Closed-loop benchmark runner

The runner connects the problem contract, registry, and experiment
configuration. It evaluates an initial Sobol design, then
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
of the global torch RNG. Trajectory persistence and regret metrics are
available through separate benchmark modules; see
[Result storage](storage.md) and [Benchmark metrics](metrics.md).

Observation callbacks using global randomness remain the responsibility of
the caller. For reproducible noisy experiments, use an explicit observation-
noise policy and record its seed semantics; see the
[Comparative fairness protocol](comparative-fairness-protocol.md).
For candidate callbacks that must not access truth-bearing problem fields,
use `restrict_candidate_problem=True` and verify the restricted-view
contract before certifying a comparison.
