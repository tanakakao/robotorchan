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
| 24 Expressive Surrogates | Posterior-mean ranking over a finite input pool for DKL, I-BNN, spectral mixture and DeepGP | Execute excluded notebook; inspect each model's training and posterior contract |
| 25 Non-GP Surrogates | qEI evaluated over a finite candidate pool for RandomForestSurrogate | Execute excluded notebook; check posterior-sampling semantics and non-GP family breadth |
| 13 HeterogeneousMTGP | Target-task posterior ranking, observation append and refit | Confirm current PR integration and distinguish input heterogeneity |
| 42 Heterogeneous constrained BO | Regression/classification constrained qEI, candidate evaluation, label observation and refit | Confirm Phase 13 PR is merged before assuming default-branch coverage |

## Next audit sequence

1. Execute excluded notebooks 24 and 25 in isolation; record job links or local
   command output and failures without inferring success from static inspection.
2. Cross-reference public model exports against notebooks, including variants
   within a family; mark `covered`, `partial`, `not applicable` or `unverified`.
3. Inspect model-specific posterior and training contracts before adding candidate
   optimization; preserve non-differentiable and mixed-variable search semantics.
4. For any genuine gap, add candidate generation and objective evaluation to
   the appropriate model notebook, without duplicating a benchmark.
5. Validate all changed notebooks with Restart Kernel + Run All, update manifest
   metadata if needed, and confirm CI. Do not conflate a green default CI with
   successful execution of excluded notebooks.

## Completion criteria

- Every public model family has an explicit candidate-coverage status and evidence.
- Gaps are either fixed and executed or documented with a concrete technical reason.
- Training support is reported separately from candidate-generation support.
- No unsupported `fit()` methods, artificial wrappers or stale phase claims
  are introduced in user-facing notebooks.
