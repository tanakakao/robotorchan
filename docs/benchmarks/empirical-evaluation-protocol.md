# Frozen empirical execution protocol and acceptance gates

Status: **protocol frozen; empirical results not yet certified**.

This document defines the decision rules for Phase 26–30. It extends
[the fairness contract](empirical-execution-contract.md) without changing
the benchmark runner or claiming that a completed smoke test establishes
optimization performance.

## Frozen comparison population

Use `ComparativeExperimentCell(tier="empirical_standard", ...)` and the
repository's eligibility matrix. The eligible measured strata are:

| Problem | Eligible strategies |
| --- | --- |
| branin | random, sobol, qEI, qNEI |
| hartmann6 | random, sobol, qEI, qNEI |
| branin_currin | random, sobol, qEHVI, qNEHVI |

The `strength_pass` and `strength_conductivity_pass` classification
cells remain blocked for GP acquisitions until their observed-label
history and acquisition constraints are supported and separately audited.
Do not silently treat them as comparable to unconstrained GP methods.

For each eligible problem/strategy/q combination, use:

- Seeds: `tuple(range(10))`, paired across strategies.
- Initial evaluations: 20, with the same seed-derived Sobol design.
- Additional evaluations: 50, excluding initial evaluations.
- Batch sizes: `q=1` and `q=3` as separate strata. The final `q=3`
  batch contains two candidates.
- Compute: CPU, float64. CUDA is a separately labeled experiment.
- Noisy experiments: explicit noise implementation, scale, and pairing
  policy. Never pool noiseless and noisy results.
- Candidate generation: restricted public problem view, independent
  random streams, and isolated PyTorch strategy RNG.
- Objective orientation: maximize internally, retain native directions
  and truth values in raw artifacts.
- Multiobjective hypervolume: use the benchmark's *predeclared*
  reference point, not one inferred from observed or truth outcomes.

## Pre-execution manifest

Before any measured run, persist a manifest with the git SHA, dirty
state, package versions, Python/Torch/BoTorch/GPyTorch versions,
hardware, selected problem/strategy, all ten seeds, q, initial/budget
counts, dtype/device, declared reference point, objective directions,
noise policy, model factory and fit settings, acquisition class and MC
sampler configuration, optimizer restarts/raw samples, and failure
handling policy. A missing required field is a **manifest failure**,
not an invitation to reconstruct settings from results.

Record the exact source and settings for every candidate strategy.
A method-specific optimizer setting is allowed but must be disclosed;
do not describe such runs as optimizer-budget matched unless the
budgets were actually aligned.

## Execution gates

1. **Eligibility** — `ComparativeExperimentCell` validates and the
   strategy/problem pairing is not blocked.
2. **Isolation** — callback access is limited to
   `CandidateProblemView`; no oracle evaluator, optimal value,
   reference front, future observations, or truth-derived reference.
3. **Pairing** — compare identical initial X for each seed; require
   identical initial observed Y only when the noise policy guarantees it.
4. **Budget** — each successful run has 70 evaluations total and
   `completed_batches[-1] == 50`. For `q=3`, expect 17 completed
   batches and a final increment of two.
5. **Validity** — candidates lie within bounds, objective outputs are
   finite, metrics use truth only after evaluation, and no failed
   trajectories are silently discarded.
6. **Provenance** — store per-seed trajectory, timing, error record,
   manifest and summary before calculating aggregate metrics.
7. **Review** — distinguish `config_valid`, `oracle_isolated`,
   `runtime_validated`, and `reviewed` certifications. A smoke pass
   is not an empirical performance certification.

Any failed gate blocks a performance claim for the affected cell.

## Metrics and comparison policy

- Single-objective: best-so-far truth value in the maximization
  direction and simple regret where the benchmark supplies a known
  optimum. Never infer an optimum from the evaluated candidates.
- Multiobjective: truth-based dominated hypervolume using the fixed
  reference point, and its trajectory as a function of *additional
  evaluations*, not wall time.
- Constraints: feasible-rate and feasible best/HV only for eligible
  constrained methods; report zero-feasible cases explicitly.
- Compute: separate candidate-generation, evaluation, and total
  elapsed times; report CPU and CUDA separately.
- Stability: report failed runs, exception types, successful-seed count,
  and run-to-run variability. Do not replace failures with favorable
  values or omit them from denominators.

Use seed-paired differences against Random and Sobol, with a 95%
paired bootstrap confidence interval. Keep all paired seed IDs in the
comparison; if any run fails, mark the paired result incomplete and
report failure counts separately rather than silently analyzing only
survivors. Plot mean/median and uncertainty with the seed count
clearly shown. Do not declare superiority solely from overlapping or
nonoverlapping marginal confidence intervals.

## Stop, rerun and amendment rules

- Numerical failures, exceptions, nonfinite candidates, budget
  violations, pairing mismatches, and oracle exposure are failures.
- Preserve the failing seed, configuration, traceback, partial
  trajectory, and environment. Never substitute another seed.
- A rerun must use the same source commit and frozen manifest; record
  the attempt number and outcome. Do not select the best attempt.
- Any change to acquisition hyperparameters, optimizer budgets, noise,
  reference point, seed set, or evaluation budget requires a versioned
  protocol amendment and new comparison cohort. Do not overwrite
  previous results or mix cohorts in the same aggregate.
- Report negative, inconclusive, and unsupported results explicitly.

## Phase 26 handoff

Phase 26 executes paired noiseless single-objective comparisons under
this protocol, verifies all gates, and stores raw artifacts and
failure evidence. Phase 27 handles noisy comparisons; Phase 28 handles
multiobjective comparisons. Phase 29 addresses constrained eligibility
and evidence; Phase 30 assembles the measured report.

No performance result or statistical conclusion is claimed by this
protocol document.
