# Constrained benchmark problems

All benchmark constraints use the convention **g(X) >= 0**.
Constraint residuals are returned with shape `(..., q, c)`.
A negative residual indicates infeasibility. These are outcome
constraints evaluated on candidates, not constraints on the acquisition
optimizer's search space.

| Problem | Feasible region | Constraints | Known optimal value |
| --- | --- | ---: | ---: |
| constrained_quadratic | x0+x1 <= 1 | 1 | -0.18 (maximize) |
| constrained_annulus | 0.5 <= radius <= 1 | 2 | 0.25 (minimize) |
| constrained_disconnected | two radius-0.25 disks centered at (+/-0.75, 0) | 1 | 0 (maximize) |
| constrained_narrow_band | 0.95 <= x0+x1 <= 1.05 | 2 | -0.15125 (maximize) |

The disconnected problem tests optimization across isolated feasible
components. The narrow band tests feasibility discovery in a small
portion of the unit square. Both have known feasible optima for
checking truth-based simple regret. A run with no feasible evaluations
has infinite simple regret, as specified by the problem contract.

Use `register_constrained_problems(registry)` to register these four
problems, then `run_benchmark` with the desired candidate strategy.
