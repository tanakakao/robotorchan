# Phase 14: notebook candidate-coverage audit

This is a development audit, not a user-facing Guide or Theory chapter.

## Goal

Ensure that each supported model family has an executable route from fitting or inference
to candidate proposal and evaluation. Do not manufacture a standard BO contract for
models with incompatible posterior semantics. No model-level `fit()` implementations
are authorized in this phase.

## Baseline

The notebook manifest currently declares 40 default, 2 slow and 4 excluded notebooks.
Manifest membership proves selection for CI, not successful execution of every
model variant. Check the current branch's CI run before recording execution evidence.

The earlier Phase 2 static inventory is a historical snapshot. In particular,
its absence-of-`candidate` findings for notebooks 24 and 25 are now outdated:
both contain candidate-pool selection and objective evaluation code.

## Targeted inspection

| Notebook | Candidate path observed in source | Verification still required |
| --- | --- | --- |
| 24 Expressive Surrogates | Posterior-mean ranking over a finite input pool for DKL, I-BNN, spectral mixture and DeepGP | Executed in PR CI; training/posterior semantics still require per-model review |
| 25 Non-GP Surrogates | qEI evaluated over a finite candidate pool for RandomForestSurrogate | Executed in PR CI; other non-GP families are not covered by this notebook |
| 13 HeterogeneousMTGP | Target-task posterior ranking, observation append and refit | Confirm current PR integration and distinguish input heterogeneity |
| 42 Heterogeneous constrained BO | Regression/classification constrained qEI, candidate evaluation, label observation and refit | Confirm Phase 13 PR is merged before assuming default-branch coverage |

## Phase 14 execution evidence

Commit `ed2177f190256261e76e6947344f5fce4214e0b3` passed the
`notebook-expressive-non-gp` CI job in run `40882058703`:
https://github.com/tanakakao/robotorchan/actions/runs/40882058703

The job runs notebooks 24 and 25 sequentially using `--only` and
`--timeout 600`. Both completed successfully, so candidate selection
and evaluation were exercised for the code paths in those notebooks.
This is **not** evidence that every expressive or non-GP model family has
candidate coverage, nor does it establish acquisition optimality.

The initial run `40880413823` failed because notebook 24 imported
`SingleTaskDeepGP` from the obsolete `robotorchan.models.deep_gp` path.
The notebook now imports it from `robotorchan.models.expressive.deep_gp`.
The corrected CI run passed all 11 jobs, and Documentation links passed.

## Next audit sequence

1. Cross-reference public model exports against notebooks, including variants
   within a family; mark `covered`, `partial`, `not applicable` or `unverified`.
2. Inspect model-specific posterior and training contracts before adding candidate
   optimization; preserve non-differentiable and mixed-variable search semantics.
3. For any genuine gap, add candidate generation and objective evaluation to
   the appropriate model notebook, without duplicating a benchmark.
4. Validate all changed notebooks with Restart Kernel + Run All, update manifest
   metadata if needed, and confirm CI. Do not conflate a green default CI with
   successful execution of excluded notebooks.

## Completion criteria

- Every public model family has an explicit candidate-coverage status and evidence.
- Gaps are either fixed and executed or documented with a concrete technical reason.
- Training support is reported separately from candidate-generation support.
- No unsupported `fit()` methods, artificial wrappers or stale phase claims
  are introduced in user-facing notebooks.
