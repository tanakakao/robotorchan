# Comparative optimization benchmark: execution and eligibility matrix

This document converts the [fairness protocol](comparative-fairness-protocol.md)
into an executable planning matrix. The statuses below are **static
eligibility assessments**, not results of executed benchmarks.

## Experiment profiles

| Comparison tier | Seeds | Initial points | New evaluations | q | Precision | Device |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Smoke | 1 | 4 | 4 | 1 | float64 | CPU |
| Standard | 5 | 20 | 40 | 1, 3 | float64 | CPU |
| Full | 20 | 30 | 100 | 1, 3, 5 | float64 | CPU |
| Full CUDA | 20 | 30 | 100 | 1, 3, 5 | float64 | CUDA, separately |

These are comparison tiers, **not** the existing `get_execution_profile`
presets. Each q value is a separate stratum. Use seeds 0 through N-1
for each tier; record the actual seed mapping. The evaluation budget
counts newly evaluated points, not initial points or optimizer calls.

## Method eligibility by problem family

Status key: **baseline** = candidate generator exists, but requires
oracle-access review; **integration** = component-level integration
exists, but no verified benchmark strategy; **blocked** = a required
metric or reproducibility contract is missing; **not applicable** =
the acquisition does not match the problem's objective structure.

| Problem | Random | Sobol | qEI | qNEI | qEHVI | qNEHVI |
| --- | --- | --- | --- | --- | --- | --- |
| branin | baseline | integration | baseline | baseline | not applicable | not applicable |
| hartmann6 | baseline | integration | baseline | baseline | not applicable | not applicable |
| noisy_quadratic | blocked | blocked | blocked | blocked | not applicable | not applicable |
| branin_currin | baseline | integration | not applicable | not applicable | integration | integration |
| dtlz2 (3 objectives) | blocked | blocked | not applicable | not applicable | blocked | blocked |
| strength_conductivity_pass | baseline | integration | not applicable | not applicable | integration | integration |

The `noisy_quadratic` case is a registered noisy single-objective
alternative. No `noisy_branin` problem was verified in the inspected
source; adding it requires a separate implementation and tests.

`strength_conductivity_pass` is a two-regression-objective plus
classification-feasibility scenario. Its classification-constrained
qEHVI/qNEHVI integrations are *marginal PoF-weighted approximations*,
not exact joint-MC constrained hypervolume improvement. Do not pool
them with exact constrained methods.

## Independent classification and continuous constraints

The single-objective Strength + Pass benchmark and continuous-constrained
Hartmann must be explicitly registered and verified before being
scheduled. The currently inspected `strength_conductivity_pass`
problem is multiobjective and does not substitute for Strength + Pass.
Treat classification labels as 0/1 observations, never as signed
constraint margins used to train the classifier.

## Mandatory gate: oracle access

The existing `run_benchmark` passes a full `BenchmarkProblem` to
each candidate callback, along with X and observed Y. The problem
object exposes truth evaluators, known optimum, reference information
and constraint evaluators. Therefore **none** of the rows above is
certified oracle-isolated merely by running the existing callback.

Before enabling a row for publishable comparison, implement a
restricted strategy-facing problem view **or** audit and test every
candidate callback to ensure it accesses only approved fields
(bounds, dimension, variable types and declared public metadata).
No strategy may inspect `evaluate_truth`, `objective`,
`optimal_value`, future observations, or hidden feasibility margins.
Reference points may be provided to multiobjective methods only
under a predeclared policy applied identically to comparable methods.

## Mandatory gate: execution validation

For each scheduled cell:

1. Verify registration and strategy callback construction on the
   pinned commit, including objective direction and q support.
2. Verify same initial X per seed across strategies, equal number
   of new evaluations and correct final partial batch.
3. Verify no oracle leakage and no unintended observation-noise
   leakage, including direct `BenchmarkProblem` attribute access.
4. Verify reproducible independent seed streams, and paired noise
   policy for noisy cases.
5. Verify truth-only metric, reference point, feasibility sign and
   metric curve alignment at 0 through evaluation budget.
6. Verify fit, acquisition and optimizer settings, failure policy,
   timing and package/hardware metadata are persisted.
7. For 3+ objectives, validate hypervolume against an independent
   implementation; for HV regret, fix an independent reference HV.
8. Run a smoke test before standard/full execution; mark failures
   and unsupported combinations rather than silently skipping.

## Execution sequence

- **Smoke:** exercise candidate callback, budget accounting,
  objective/constraint semantics and storage for each eligible cell.
  One seed cannot support performance claims.
- **Standard:** five paired seeds, q=1 and q=3, fixed acquisition
  settings, and seed-resampled uncertainty intervals.
- **Full:** twenty paired seeds and q=1/3/5, after standard acceptance;
  CPU and CUDA results must not be pooled for timing.

Keep a machine-readable manifest of each cell's problem, method,
tier, q, seed, configuration, eligibility decision, reason,
commit SHA, execution status and artifact path. A manifest is
a follow-up implementation deliverable, not asserted to exist here.

## Phase 3 acceptance

The experiment matrix, statuses, unsupported cases, and oracle-access
gate are documented. **No cell is yet certified for scientific
comparison.** Phase 4 must validate reference metrics and oracle
isolation before benchmarking results can be interpreted.
