# Mixed-variable benchmark problems (Phase 9)

Two deterministic single-objective problems test integer and categorical
coordinates without one-hot encoding.

| Problem | Variables | Direction | Known optimum |
| --- | --- | --- | --- |
| `mixed_quadratic` | continuous x, integer n, category c | minimize | (0.8, 2, 1), value 0 |
| `categorical_switch` | continuous x, category c | maximize | (0.7, 1), value 1 |

The category labels `0, 1, 2` index independent response curves. They do not
encode a numerical distance or order between categories. Both problems use
integer-coded categories, consistent with the benchmark domain validator.
No one-hot encoding is performed.

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

The baseline Sobol and random candidate generators round integer and
categorical coordinates to valid codes. This provides an end-to-end smoke
test; it does not benchmark specialized mixed-variable acquisition
optimization or prove uniform sampling over discrete levels.
