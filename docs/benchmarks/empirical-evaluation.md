# Empirical optimization performance — coverage and execution gates

> **Historical audit snapshot:** This document freezes a historical empirical-cycle audit and provisional Phase 22–50 plan. The listed pending CI/merge state and proposed future gates are historical, not the current repository status or a new work plan.

Audit date: 2026-10-11
Pinned main: `4d492df655e38cdab91acee689898dcebc74a4fd`
Scope: static repository and documentation inspection only. No fresh benchmark run or CI verification.

## Phase 1–20 reconciliation

The empirical cycle's merged Phase 20 is PR #1113 (anytime visualization).
The empirical cycle already includes environment provenance (#1095),
standard comparison profiles (#1096), independent RNG streams (#1098),
aggregate fairness reporting (#1103), aligned metric curves (#1104),
paired strategy comparison (#1106), fairness-gated reports (#1108),
raw trajectory persistence (#1110), report reconstruction (#1111),
and summary CSV (#1112). These are selected verified PR records, not
an exhaustive PR-by-PR reconciliation of all twenty phases.

The repository also has older comparative benchmark cycles with their
own Phase numbers. Do not treat their Phase 21 or Phase 22 as the current
empirical cycle. Existing capability matrix and validation documents
must be reused, not reimplemented.

## Coverage and reuse decisions

| Area | Evidence | Decision |
| --- | --- | --- |
| Registry and standard problems | `docs/benchmarks/comparative-capability-matrix.md` | Reuse |
| Runner and paired initial design | `docs/benchmarks/end_to_end.md` | Reuse, audit oracle access |
| Random, Sobol, qEI, qNEI | `docs/benchmarks/comparative-experiment-matrix.md` | Verify eligibility |
| qEHVI/qNEHVI | Same experiment matrix | Integration validation required |
| Truth-based regret and 2D HV | `docs/benchmarks/metrics.md` | Reuse; validate 3D HV |
| Seed-paired comparisons | `docs/benchmarks/comparison.md` | Reuse existing statistics |
| Persistence, CSV and plotting | `docs/benchmarks/end_to_end.md`, PR #1112, #1113 | Reuse |
| Resume and RNG | `docs/benchmarks/end_to_end.md` | Document limitations |

## Release-blocking scientific gaps

1. The candidate callback receives a `BenchmarkProblem` with truth
   evaluators. Audit and enforce no oracle leakage before claims.
2. Pin fit, acquisition MC samples, optimizer restarts/raw samples,
   timing boundaries, device, dtype, software and seed streams.
3. Confirm qEHVI/qNEHVI strategy wiring and distinguish marginal
   PoF-weighted approximations from joint-MC constrained acquisitions.
4. Independently validate 3+ objective HV and fixed reference HV.
5. Validate observed classification-label history, noisy observation
   streams, and partial q=3 final batches.
6. Confirm paired bootstrap, multiple-testing correction and policies
   for undefined feasible regret.
7. CI smoke and adapter parity are not independent optimization
   performance evidence.

## Frozen baseline for first measured comparison

Empirical Standard: 10 paired seeds, 20 initial evaluations, 50 new
evaluations, q=1 and q=3 as separate strata, float64 CPU. The existing Smoke tier uses exactly one seed, as enforced by
`comparative_matrix.py` and the fairness protocol. Exploratory runs
with 2–3 seeds must use an explicitly defined separate profile and
must not be labeled as the existing Smoke tier. Identical initial designs, truth-only
evaluation metrics, equal budgets, explicit method configurations and
JSON/CSV/curves/timing artifacts are required. For q=3, the final
batch contains two candidates.

## Revised phase gates

- 22: certify capability matrix and supported experiment cells.
- 23: freeze fairness, RNG, oracle isolation and execution manifest.
- 24: smoke-validate actual qEI/qNEI/qEHVI/qNEHVI candidate paths.
- 25: freeze protocols and acceptance criteria.
- 26: execute paired noiseless single-objective comparisons.
- 27: execute paired noisy single-objective comparisons.
- 28: execute paired multiobjective comparisons.
- 29: execute paired constrained comparisons.
- 30: publish the first measured report with uncertainty and failures.
- 31–40: evaluate mixed, high-dimensional, TuRBO, heterogeneous,
  classification and active-learning extensions only where eligible.
- 41–50: sensitivity, runtime, reproducibility, statistical robustness,
  regression protection and final release audit.

Phase 22–50 remains a provisional roadmap pending capability gates.
Never advance a phase based only on an unmerged PR or pending CI.

## Verification and Phase 21 exit

Static inspection: performed. No fresh tests or benchmark runs performed
by this audit. CI: pending. PR review and merge: pending.
The phase is complete only after the audit PR passes required checks,
is reviewed, and is merged.
