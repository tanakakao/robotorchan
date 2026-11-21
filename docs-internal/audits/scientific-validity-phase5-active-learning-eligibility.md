# Scientific validity audit — Phase 5: classification active-learning empirical eligibility

Status: **protocol and capability audit; no new empirical measurements** (2026-11-21).

The [all-strategy BO trajectory audit](scientific-validity-phase5-all-strategy-pairing.md) confirms the initial-design and evaluation-budget invariants for existing Branin/Hartmann6 comparisons. It does not validate active-learning acquisition performance. This note defines the smallest meaningful next empirical comparison using **existing** benchmark components.

## Existing reusable implementation

- `src/robotorchan/benchmarks/active_learning_problems.py` defines deterministic `circle_boundary` and `wave_boundary`, canonical binary labels, classification accuracy, and Brier score. Their dummy BO objective is not an AL performance measure.
- `src/robotorchan/benchmarks/empirical_active_learning.py` implements `run_active_learning_baselines(destination, boundary, git_sha)`. It records 10 paired seeds, 20 initial points, 50 additional labels, a 1024-point fixed Sobol reference set, and full accuracy/Brier learning curves for Random and Sobol.
- Its predictor is `kernel_label_probabilities`, a **fixed-bandwidth kernel smoother, not a GP classifier**. Both candidate policies are pre-generated and **do not adapt to labels**.
- `tests/benchmarks/test_empirical_active_learning.py` validates basic predictor/metric behavior; the existence of tests is not evidence of multi-seed model superiority.
- `docs/benchmarks/active_learning.md` explicitly distinguishes the problem definitions and smoke baseline from a genuine acquisition-driven AL runner.

## Eligibility matrix

| Candidate policy | Existing empirical runner | Learns from acquired labels to choose next query? | Eligible for claim of acquisition-driven AL benefit? |
| --- | --- | --- | --- |
| Random | Yes | No | Baseline only |
| Sobol | Yes | No | Baseline only |
| GP classification + predictive entropy | Not implemented in this runner | To be verified in a new closed loop | **No**, until executed |
| GP classification + BALD | Not implemented in this runner | To be verified in a new closed loop | **No**, until executed |
| GP classification + boundary straddle | Not implemented in this runner | To be verified in a new closed loop | **No**, until executed |

Existing acquisition classes elsewhere in the library are **not** sufficient evidence of end-to-end empirical eligibility: their candidate generation, retraining, and evaluation must be connected and validated.

## Proposed minimal measured experiment (not yet run)

1. Select **one** boundary (prefer `wave_boundary` for nonlinear separation) and reuse the same 10 seed indices, 20 initial labels, 50 additional label queries, CPU float64, and independently fixed 1024-point reference set.
2. Compare Random and Sobol against **one** existing GP-classification acquisition (predictive entropy first). Use the library's public model, fit, posterior and acquisition APIs rather than introducing a benchmark-specific surrogate or acquisition.
3. Ensure candidate generation sees only acquired X and observed 0/1 labels; the true margin, reference labels and future outcomes are held exclusively by the evaluation oracle.
4. Freeze model fit settings, optimizer backend, q=1, acquisition budget, failure handling, and RNG streams **before** running. Refit after each acquired label; never train on reference labels.
5. Primary endpoint: per-seed final reference Brier score (lower is better). Secondary: reference accuracy and learning curves. Compare seed-paired differences with a confidence interval; report all failures and wall time.
6. Save raw per-seed queried X, observed labels, predictions/metrics, complete manifest, source SHA, dependency versions, and execution errors. Do not certify a comparison from summary CSV alone.

## Acceptance gates

- **Leakage:** assert no ground-truth margin or reference labels reach model fitting or candidate selection.
- **Fairness:** verify exact initial X and labels by paired seed, identical label budget, q and reference set.
- **Correctness:** confirm predictive probabilities are finite in [0,1] and metrics are evaluated only on the independent reference set.
- **Evidence:** require actual archived measured trajectories and paired-seed uncertainty before any performance statement.
- **Scope:** the fixed kernel smoother used by current baselines must be held constant for the passive baselines or explicitly treated as a different predictor; do not conflate differences in surrogate quality with acquisition-policy effects.

## Decision

**AL empirical superiority is not yet established.** The existing runner supports valid **passive sampling baselines**, but it does not evaluate a GP-based adaptive AL acquisition. Implement the smallest missing closed-loop integration only after confirming reuse of the public classifier, fit and acquisition interfaces; avoid duplicating the existing benchmark suite or adding broad redundant tests.
