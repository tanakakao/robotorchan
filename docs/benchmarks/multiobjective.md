# Multi-objective benchmark suite

The suite now includes:

| Problem | Variables | Objectives | Directions | Pareto metadata |
| --- | ---: | ---: | --- | --- |
| Biobjective linear | 1 | 2 | maximize/maximize | sampled analytic front |
| Branin-Currin | 2 | 2 | maximize/maximize | reference point only |
| DTLZ2 | 7 | 3 | minimize/minimize/minimize | sampled analytic front |
| ZDT1 | 6 | 2 | minimize/minimize | sampled analytic front |

Branin-Currin uses the standard negative-Branin/Currin convention,
with normalized inputs on [0, 1]^2 and a conservative maximization
reference point (-310, -1). Its true Pareto front is not claimed to
be analytically known; `reference_front` is intentionally absent.
At the Currin boundary x2=0, the exponential term is defined by its
continuous limit, avoiding NaNs.

`register_multiobjective_problems(registry)` registers all four
problems. Hypervolume comparisons should use a common fixed reference
point, and transform minimization objectives consistently before
computing hypervolume. This phase extends benchmark problem coverage,
not the multiobjective BO acquisition strategy.

## Reference-front and direction contract

For `biobjective_linear`, the analytic Pareto front is `(x, 1 - x)`.
For `zdt1`, the front is `(u, 1 - sqrt(u))`. For `dtlz2`, the
three-objective front lies on the positive unit sphere. Stored
`reference_front` tensors are finite **samples** of these analytic
fronts, not exact continuous representations. `branin_currin` has a
reference point but no analytically certified stored Pareto front.

Reference points are specified in each problem's **native objective
direction**. For example, the biobjective maximization problem uses
`(-0.1, -0.1)` while the minimization examples use `(1.1, ...)`.
Transform both observed objectives and reference points consistently
with `problem.to_maximization(...)` before computing hypervolume with a
maximization-based implementation. Do not compare untransformed values.

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

These problem definitions alone do not establish qEHVI/qNEHVI
integration correctness or optimization performance.
