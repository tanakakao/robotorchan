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

## Non-GP notebook execution evidence

CI run `40977583875` (commit `507818e4c05a341c74deeb1ac123a9e4d2bf8398`)
passed all 11 jobs, including `notebook-expressive-non-gp` and
`ngboost-extra`. The notebook `25_non_gp_surrogates.ipynb` now executes
candidate proposals and objective evaluations for Random Forest, Extra Trees,
Gradient Boosting, Histogram Gradient Boosting, and NGBoost.

The standard notebook job runs without the optional NGBoost dependency and
skips that section; the `ngboost-extra` job installs the dependency and
executes the complete notebook. NGBoost's custom
`GaussianDistributionPosterior` requires an explicit `IIDNormalSampler`
for BoTorch MC acquisition because automatic sampler dispatch does not
recognize that posterior class. This is a notebook-level compatibility
solution, not a claim that BoTorch's default sampler dispatch supports the
custom posterior.

These five examples have CI execution evidence, distinct from the
registry-wide static heuristic counts. Other mapped model variants remain
subject to individual inspection.

## Registry-to-notebook static audit baseline

The Python 3.11 CI job in run `40966360099` executed
`scripts/audit_notebook_model_candidates.py --check` successfully.

| Classification | Public model registry entries |
| --- | ---: |
| `static_pipeline` | 17 |
| `partial` | 17 |
| `unverified` | 108 |
| **Total** | **142** |

These are **conservative static heuristics**, not model execution or a count of
working Bayesian optimization workflows. A constructor must appear as a direct
call in its registry-mapped notebook to count as present. The notebook-level
candidate/evaluation tokens are not attributed to individual model objects.
Therefore `static_pipeline` is *not* a verified model-level end-to-end result,
and `unverified` does *not* establish missing functionality. In particular,
notebooks may instantiate models via factories, loops, or aliases, and the
registry may point multiple variants to one representative example.

Next prioritize inspection of the 108 `unverified` entries against actual
notebook source, separating truly absent model variants from representative
family coverage. Record execution evidence per model variant before upgrading
any status to `covered`. Keep model `fit()` compatibility as an independent
dimension, without implementing new fitting methods in this phase.

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
