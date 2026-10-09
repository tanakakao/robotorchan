# Benchmark metrics (Phase 11)

All metrics use deterministic **ground-truth** responses rather than noisy
observations. Outcome constraints follow `g(X) >= 0`.

| Metric | Scope | Interpretation |
| --- | --- | --- |
| `feasibility_rate` | any objective count | fraction of feasible evaluations |
| `cumulative_feasibility_rate` | any objective count | prefix feasibility fraction |
| `simple_regret_curve` | single objective, known optimum | best feasible truth regret |
| `hypervolume_2d` | two maximization objectives | exact dominated union area |
| `hypervolume_curve` | two objectives | feasible hypervolume after each evaluation |

`simple_regret_curve` yields `+inf` until the first feasible point is
observed. For problems without constraints, all evaluations are feasible.

Hypervolume converts both objectives and the reference point to maximization
orientation before calculation. The 2D implementation handles dominated and
duplicate points. It does not yet cover 3+ objectives, statistical aggregation
over seeds, or hypervolume regret against an analytic Pareto front.

```python
from robotorchan.benchmarks import (
    hypervolume_curve,
    simple_regret_curve,
)

# Compute one curve per BenchmarkTrajectory and its BenchmarkProblem.
```
