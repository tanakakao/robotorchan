# Multi-fidelity benchmark problems

Both benchmarks represent the design variable as `x = X[..., 0]` and
the continuous fidelity as `s = X[..., 1]`, with `x, s in [0, 1]`.
The highest fidelity is `s=1`; cheaper approximations have `s<1`.

| Problem | High-fidelity objective | Low-fidelity approximation |
| --- | --- | --- |
| multifidelity_quadratic | (x - 0.7)^2 | add (1-s)*(0.25 + 0.2*(x-0.2)^2) |
| multifidelity_oscillatory | (x-0.65)^2 + 0.05*sin(6*pi*x)^2 | add (1-s)*(0.1 + 0.3*(x-0.1)^2) |

The shared evaluation cost is `0.1 + 0.9*s^2`, so fidelity zero costs
0.1 and fidelity one costs 1.0. This enables cumulative-cost comparisons
using the existing `BenchmarkTrajectory.cumulative_cost` property.

The quadratic benchmark has known global minimum zero at `(x,s)=(0.7,1)`.
The oscillatory benchmark deliberately leaves `optimal_value` unset
rather than reporting an unverified analytic optimum.

**Important:** the final coordinate is an explicit fidelity variable,
not an additional physical design coordinate. For high-fidelity-only
evaluation and comparisons, hold `s=1`. The current generic random
candidate generator samples both coordinates, and is a baseline rather
than a fidelity-aware acquisition strategy. The runner counts evaluations
and records costs but does not enforce a monetary cost budget.

Register using `register_multifidelity_problems(registry)`.
