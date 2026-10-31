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

## Reduced GP notebook coverage (20 registry mappings)

Notebook: `examples/notebooks/19_reduced_gp.ipynb`. The CI run
`40995535501` passed on commit `1b2db43c3ad3a1c7f00b3ea1246727a41582183a`,
including the `notebook-reduced` execution job.

All 20 mapped model classes are explicitly instantiated in notebook code:

- Single-task input reduction: `PCAGP`, `PLSGP`,
  `RandomProjectionGP`, `ReducedGP`.
- Mixed single-task input reduction: `MixedPCAGP`, `MixedPLSGP`,
  `MixedRandomProjectionGP`, `MixedReducedGP`.
- Long-format multitask: `PCAMultiTaskGP`, `PLSMultiTaskGP`,
  `RandomProjectionMultiTaskGP`, `MixedReducedMultiTaskGP`,
  `ReducedMultiTaskGP`.
- Block-design multitask: `PCAKroneckerMultiTaskGP`,
  `PLSKroneckerMultiTaskGP`, `RandomProjectionKroneckerMultiTaskGP`,
  `MixedReducedKroneckerMultiTaskGP`, `ReducedKroneckerMultiTaskGP`.
- Output reduction: `OutputPCAGP`, `OutputPLSGP`.

All examples include model fitting, posterior prediction, candidate selection,
and an objective evaluation. This is one CI execution of the complete notebook,
not exhaustive configuration coverage. Some examples use a BoTorch acquisition
function, whereas multitask and output-reduction examples use posterior-mean
ranking of finite candidates. The latter do not demonstrate acquisition
optimization and should not be described as full Bayesian optimization.


## Neural reduced GP notebook coverage (29 registry mappings)

Notebook: `examples/notebooks/20_neural_reduction_gp.ipynb`. Commit
`9368b590ae16de9763cf70c91874257197fc43b0` passed CI run
`41013845874`, including the `notebook-neural-reduced` job, which executes
the complete notebook with `--timeout 600`.

All 29 registry-mapped classes are now explicitly instantiated in the notebook:

- Frozen single-task: `AutoEncoderGP`, `VAEGP`,
  `SupervisedAutoEncoderGP`, `SupervisedVAEGP`.
- Mixed frozen single-task: `MixedAutoEncoderGP`, `MixedVAEGP`,
  `MixedSupervisedAutoEncoderGP`, `MixedSupervisedVAEGP`.
- Joint single-task: `JointEncoderGP`, `HybridAutoEncoderGP`,
  `JointVAEGP`, `MixedJointEncoderGP`, `MixedHybridAutoEncoderGP`,
  `MixedJointVAEGP`.
- Long-format multitask: `AutoEncoderMultiTaskGP`, `VAEMultiTaskGP`,
  `SupervisedAutoEncoderMultiTaskGP`, `SupervisedVAEMultiTaskGP`,
  `JointEncoderMultiTaskGP`, `HybridAutoEncoderMultiTaskGP`,
  `JointVAEMultiTaskGP`, `MixedJointEncoderMultiTaskGP`.
- Block-design multitask: `AutoEncoderKroneckerMultiTaskGP`,
  `VAEKroneckerMultiTaskGP`,
  `SupervisedAutoEncoderKroneckerMultiTaskGP`,
  `SupervisedVAEKroneckerMultiTaskGP`,
  `JointEncoderKroneckerMultiTaskGP`,
  `HybridAutoEncoderKroneckerMultiTaskGP`,
  `JointVAEKroneckerMultiTaskGP`.

All examples exercise fitting, posterior inference, candidate proposal and
objective evaluation. Frozen reducers use `fit_gpytorch_mll(model.make_mll())`;
joint neural representations use `training_loss()` with an Adam optimizer.
Some single-task examples use BoTorch acquisition optimization; many mixed and
multitask examples select the highest posterior mean from a finite candidate
pool. The latter are **not acquisition-optimized BO**, and successful CI
execution does not imply statistical performance or robustness across seeds.

This closes the notebook-20 *mapped-model execution* gap, not the full
142-model registry-wide coverage audit.

## Expressive GP notebook coverage (16 registry mappings)

Notebook: `examples/notebooks/24_expressive_surrogate_gp.ipynb`.
Commit `d143b92c72649a9aec57da148767b6b479c033ef` passed
CI run `41150143395`, including the `notebook-expressive-non-gp` job.

All 16 registry-mapped expressive model classes are now explicitly instantiated:

- Single-task exact GP: `InfiniteWidthBNNGP`, `SpectralMixtureGP`,
  `MixedInfiniteWidthBNNGP`, `MixedSpectralMixtureGP`.
- Long-format multitask exact GP: `InfiniteWidthBNNMultiTaskGP`,
  `SpectralMixtureMultiTaskGP`, `MixedInfiniteWidthBNNMultiTaskGP`,
  `MixedSpectralMixtureMultiTaskGP`.
- Block-design multitask exact GP:
  `InfiniteWidthBNNKroneckerMultiTaskGP`,
  `SpectralMixtureKroneckerMultiTaskGP`,
  `MixedInfiniteWidthBNNKroneckerMultiTaskGP`,
  `MixedSpectralMixtureKroneckerMultiTaskGP`.
- Deep GP: `SingleTaskDeepGP`, `MultiTaskDeepGP`,
  `MixedSingleTaskDeepGP`, `MixedMultiTaskDeepGP`.

The notebook exercises fitting, posterior prediction, candidate selection and
objective evaluation. Deep GP models use short Adam loops on
`training_loss()`; Kronecker models use the BoTorch torch optimizer to avoid
SciPy fallback prior-resampling failures. Candidates are selected by posterior
mean from a finite pool, **not by optimizing a BO acquisition function**.
The notebook also demonstrates `JointEncoderGP`, which is mapped to notebook
20 and is not counted among these 16 classes.

The passing CI run verifies these lightweight examples only. It does not
establish multi-seed robustness, acquisition optimality or broader configurations.

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

## Registry reconciliation after expressive coverage

The registry maps 142 public model entries to 20 representative notebooks.
Executed notebook coverage is distinct from registry mapping or static-token
classification. The following notebooks have explicit, CI-executed model-level
examples in this phase:

| Notebook | Mapped entries | Evidence |
| --- | ---: | --- |
| 19 reduced GP | 20 | CI run `40995535501` |
| 20 neural reduction GP | 29 | CI run `41013845874` |
| 24 expressive GP | 16 | CI run `41150143395` |
| 25 non-GP | 5 | CI run `40977583875`, including NGBoost |\n| 16 noise models | 10 | CI run `41157626014` |

These account for 70 mapped entries. Notebook 16 adds 10 more verified mappings,\nfor 80 total. The remaining 62 mappings require
individual verification; this does not mean that all 72 are absent from
notebooks or unexecutable. Do not sum the old static heuristic statuses with
these executed counts.

## Noise-model notebook coverage (10 registry mappings)

Notebook: `examples/notebooks/16_noise_models.ipynb`. Commit
`c5adf3391bf1244015ba418843ae8416abdf74aa` passed CI run
`41157626014`, including the default `notebooks` execution job.

All 10 registry-mapped classes now have examples with fitting, posterior
prediction, finite-pool candidate proposal, and objective evaluation:

- Replicate-noise: `ReplicateNoiseSingleTaskGP`,
  `MixedReplicateNoiseSingleTaskGP`, `ReplicateNoiseMultiFidelityGP`.
- Iterative heteroskedastic: `HeteroskedasticSingleTaskGP`,
  `MixedHeteroskedasticSingleTaskGP`, `HeteroskedasticMultiTaskGP`,
  `MixedHeteroskedasticMultiTaskGP`, `HeteroskedasticMultiFidelityGP`.
- Joint variational heteroskedastic: `JointHeteroskedasticSingleTaskGP`,
  `MixedJointHeteroskedasticSingleTaskGP`.

Existing training APIs are used without adding model-level `fit()`:
`fit_heteroskedastic(iterations=1)` for iterative noise models,
`training_loss()` with Adam for joint variational models, and BoTorch
`fit_gpytorch_mll(model.make_mll())` for the replicate multi-fidelity
model, which has no `fit()` method. The initial incorrect `fit()`
call failed CI run `41156283308`; the corrected run `41156812536`
passed. Candidates are posterior-mean finite-pool selections, **not
acquisition-optimized Bayesian optimization**. One successful CI run
does not establish multi-seed stability or optimality.

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
