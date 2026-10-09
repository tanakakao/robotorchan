# Comparative optimization benchmark: fair comparison protocol

This document defines the **comparison contract** for Random, Sobol, qEI,
qNEI, qEHVI, qNEHVI, and heterogeneous constrained strategies. It is a
protocol, not an executed experiment or a claim of superior performance.

## Baseline and experimental profiles

The existing `get_execution_profile` presets are unchanged. Their
`standard` profile uses three seeds, eight initial points, 40 new
evaluations, and q=1; `extended` uses five seeds, 16 initial points,
100 new evaluations, and q=4. The comparative experiment matrix below
is separate from those built-in presets.

| Profile | Seeds | Initial observations | New evaluations | q | Device | Dtype |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Smoke | 1 | 4 | 4 | 1 | CPU | float64 |
| Standard comparison | 5 | 20 | 40 | 1, 3 | CPU | float64 |
| Full comparison | 20 | 30 | 100 | 1, 3, 5 | CPU; CUDA separately | float64 |

The same `(problem, seed, initial_points, evaluation_budget, q)` tuple
must be used for each eligible strategy. Each q value is a separate
comparison stratum, not an independent replicate. The final q-batch
may be shorter than requested; count actual objective evaluations,
not candidate-generation iterations.

## Initial observations and observation noise

Use `sobol_initial_design` with the same seed and bounds across
strategies, and verify the initial X tensors are identical. The
strategy receives only X and `Y_observed`; never expose `Y_truth`,
known optima, future evaluations or evaluation-time constraints
unless that knowledge is explicitly part of the method definition.

For noisy problems, use independent, seeded observation-noise streams
that are reproducible for each experiment. A problem factory's
private RNG seeded identically on each run does **not** establish
paired noise at different adaptive X locations. Document whether
comparisons use common random numbers, independent observation noise,
or deterministic precomputed noise keyed to a sample identity.
Do not use global RNG state as an implicit reproducibility guarantee.
Current `run_benchmark` has no observation-noise generator argument;
an extension or explicit noise adapter is needed before strict noisy
paired comparisons can be certified.

## Strategy and optimizer budgets

- Random and Sobol must share the same evaluation budget as BO.
- BO must declare surrogate family, fitting policy, transforms, known
  observation noise assumptions, acquisition type, MC sampler and
  sample count, optimizer, restarts, raw samples, stopping conditions,
  and candidate constraints.
- Fix and report optimizer and MC settings within each comparable
  strategy family. Algorithm-specific settings may differ, but must
  be predeclared rather than tuned against test outcomes.
- Separate seed streams for initial designs, observation noise,
  surrogate fitting, acquisition MC, and acquisition optimization.
  Preserve the mapping from top-level run seed to substream seeds.
- Report failed fits and invalid candidate proposals. Do not
  silently substitute random candidates without labeling the fallback.

## Outcomes and metric alignment

Use truth-only simple regret for single-objective cases and feasible
hypervolume for two-objective cases, with a fixed reference point
and direction convention. Do not derive reference points or known
optima from one strategy's observed results. For three-objective
DTLZ2, extend the documented 2D-only hypervolume implementation
before inclusion. HV regret requires a fixed, independently
specified reference HV; do not infer it from an individual run.

Record the entire trajectory including initial observations. The
horizontal axis for cross-strategy comparisons is **number of new
evaluations**, from 0 through the declared budget. At zero new
evaluations, compute the incumbent from all initial observations.
The existing reporting CSV index starts at the first initial
observation; transform and validate alignment explicitly before
plotting or computing paired differences. q>1 trajectories should
use a declared within-batch evaluation convention or compare at
shared completed-batch checkpoints, without pretending a q-batch
was selected sequentially.

For constrained problems, use `g(X) >= 0` as truth feasibility.
Record feasible incumbent, feasibility rate and first-feasible
evaluation; keep classification predictive probability, hard
labels and true signed margins distinct. Undefined regret before
the first feasible observation must be represented and reported,
not silently treated as a finite zero.

## Statistical analysis

Pair strategy results by exact problem, seed, q, budget, initial
design, and noise policy. Use seed-level paired differences and
paired bootstrap intervals; resample **seeds**, not individual
evaluations. Predeclare a primary metric at the final common
budget, confidence interval, bootstrap seed and resample count
(default 2,000). Show trajectories as secondary diagnostics.
Avoid claims of significance from a one-seed smoke run.
If failures or missing seeds occur, disclose exclusions and
report the common paired sample size.

## Timing and environment

Record model fitting, acquisition construction, acquisition
optimization, objective evaluation, and end-to-end wall time
separately where instrumentation permits. GPU timings require
synchronization and must be reported independently from CPU.
Pin repository commit, Python, PyTorch, BoTorch, GPyTorch, device,
dtype, processor/GPU, OS, threads, and relevant configuration.
Runtime comparisons require a shared hardware/software setup.

## Comparability classes and exclusions

1. **Exact supported**: methods with matching objective and
   feasibility semantics and validated metric implementation.
2. **Approximation**: classification feasibility weighting by
   marginal PoF. Existing qNEI/qNEHVI integration paths do not
   fully correct baseline feasibility; label them approximate
   and report separately from exact joint-MC constrained methods.
3. **Not yet eligible**: three-objective HV until validated,
   noisy paired comparisons without a reproducible noise contract,
   or strategies without a closed-loop candidate callback.

## Acceptance criteria

- The same initial X, evaluation count, q, dtype and device are
  verifiably applied across strategies within a comparison stratum.
- Observation/truth separation is maintained.
- Noise and random substreams are explicit and reproducible.
- Metric direction, feasibility and reference values are fixed.
- Paired analysis operates on seed-aligned evaluation axes.
- Approximate constrained acquisitions are not misrepresented
  as exact methods.
- Execution profiles and reporting indices are not conflated.
- Missing runtime instrumentation is documented, not fabricated.

## Implementation follow-up

Phase 3 should translate this protocol into an explicit eligibility
matrix and machine-checkable configuration validation. Later phases
should implement any missing noise control, metrics, strategy
callbacks and timing hooks before reporting numerical comparisons.
