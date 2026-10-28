# Comparative benchmark reference and metric validation

The reference cases in
`tests/benchmarks/test_comparative_reference_metrics.py` provide
independent arithmetic checks for minimization/maximization regret,
truth-versus-observation separation, feasible regret, two-dimensional
hypervolume, and evaluation-order metric curves.

## Reference cases

| Contract | Reference | Expected |
| --- | --- | --- |
| Minimize x² on [0, 2] | X = [2, 1, 0] | Regret [4, 1, 0] |
| Maximize x² on [0, 2] | X = [2, 1, 0] | Regret [0, 0, 0] |
| Maximize x on [0, 1] | X = [0.2, 0.5, 0.8] | Regret [0.8, 0.5, 0.2] |
| Feasible maximize x | Constraint x - 0.5 >= 0 | Regret [inf, inf, 0.5, 0.1] |
| 2D rectangle union | (2,1), (1,2), reference (0,0) | HV 3 |
| Feasible 2D HV | First point infeasible | Curve [0, 1, 1.5] |
| 3D HV | Three objective values | Explicit unsupported error |

## Evaluation-axis convention

The existing metric functions return a value **after each observation**,
including initial observations. They do not automatically collapse the
initial design into one evaluation-zero point. Comparative reports must
map the last initial observation to **0 new evaluations**, then use
the following observations for steps 1 through the evaluation budget.
For q>1, acquisition decisions occur at batch boundaries even if
metrics are recorded after each member of a completed batch.

## Outstanding validation gates

- **Oracle access:** `run_benchmark` can restrict candidate callbacks
  to `CandidateProblemView` with `restrict_candidate_problem=True`,
  but this is not the default. Without that option, callbacks may receive
  the truth-bearing `BenchmarkProblem`. Metric correctness alone does
  not prevent leakage. Verify the restricted-view execution contract
  before certification; see the
  [Empirical execution contract](../../contracts/benchmarks/empirical-execution-contract.md).
- **Observed classification labels:** Current callback inputs include
  observed objective Y but not observed Pass/Fail labels. A separate
  history interface is required before classification-constrained
  benchmark cells can be certified without constraint-oracle leakage.
- **Reference HV:** BraninCurrin has a reference point but no
  independent reference HV. Do not publish HV regret until defined.
- **Three-objective HV:** The current `hypervolume_curve` rejects
  DTLZ2. Extend and independently verify before enabling it.
- **Constrained optimum:** Feasible regret needs a certified feasible
  optimum; `+inf` before first feasibility is intentional.
- **Eligibility enforcement:** A machine-checkable eligibility matrix
  and configuration validator are available; validation alone is not
  evidence that a strategy executed successfully. See
  [Empirical eligibility](empirical-eligibility.md) and verify the
  relevant runtime evidence before promoting a benchmark cell.

## Verification

Run `pytest -q tests/benchmarks/test_comparative_reference_metrics.py`.
The tests assert exact reference cases but do not constitute
multi-seed optimization performance evidence.
