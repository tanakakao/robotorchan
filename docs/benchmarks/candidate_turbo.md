# Candidate constraints and trust-region benchmark scenarios

This module adds two deterministic **candidate-domain** scenarios:

- `candidate_linear_region`: `x0 + x1 <= 1`.
- `candidate_nonlinear_region`: a disk of radius `0.2` centered at `(0.5, 0.5)`.

Both optimize a two-dimensional quadratic over `[0, 1]^2`.
Use `CandidateConstraintScenario.validate(X)` to reject infeasible
proposals. These are **not** observed outcome constraints:
`BenchmarkProblem.evaluate_constraints(X)` returns zero constraint
columns. Register objective definitions with
`register_candidate_constraint_problems(registry)`; the separate
feasibility callbacks must also be supplied to the candidate optimizer.

`TrustRegionState` provides a small deterministic diagnostic for
TuRBO-style length expansion, shrinkage and restart thresholds.
`bounds(domain_bounds)` clips an axis-aligned region around the center.
`update(improved)` tracks consecutive successes and failures.

This phase supplies reusable scenarios and state-transition tests,
**not** a complete TuRBO optimizer or an automatic integration between
the candidate constraint callback and `run_benchmark`. Existing
optimizer-side nonlinear candidate constraint and TuRBO components
remain the production execution paths; an end-to-end strategy adapter
is a separate integration task.
