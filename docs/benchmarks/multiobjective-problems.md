# Multiobjective benchmark problems (Phase 8)

Phase 8 introduces deterministic, unconstrained multiobjective test functions.

| Name | Variables | Objectives | Native direction | Pareto front |
| --- | --- | --- | --- | --- |
| `biobjective_linear` | 1 | 2 | maximize | (x, 1-x) |
| `zdt1` | 6 | 2 | minimize | (u, 1-sqrt(u)) |
| `dtlz2` | 7 | 3 | minimize | positive unit sphere |

The reference fronts are finite **samples** of the analytic fronts, not exact
continuous representations. Reference points use the **native objective
directions**: (-0.1,-0.1) for maximization and (1.1,...) for minimization.
Call `problem.to_maximization(...)` on both objectives and reference points
before calculating hypervolume with a maximization-based implementation.

```python
from robotorchan.benchmarks import (
    BenchmarkProblemRegistry,
    register_multiobjective_problems,
)

registry = BenchmarkProblemRegistry()
register_multiobjective_problems(registry)
problem = registry.create("zdt1")
print(problem.reference_front.shape)
```

These functions provide ground truth for subsequent hypervolume-regret and
qEHVI/qNEHVI comparisons; this phase does not implement those metrics or
acquisition strategies.
