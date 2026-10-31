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
| 25 non-GP | 5 | CI run `40977583875`, including NGBoost |
| 16 noise models | 10 | CI run `41157626014` |
| 15 robust observation models | 8 | CI run `41159977362` |
| 09 MAP-SAAS and additive GP | 7 | CI run `41165276107` |
| 03 multi-fidelity GP | 7 | CI run `41166885750` |
| 14 contextual GP | 6 | CI run `41172687822` |
| 04 multitask GP | 5 | CI run `41177219776` |
| 18 nonstationary GP | 5 | CI run `41178611160` |
| 13 heterogeneous multitask GP | 4 | CI run `41181618304` |\n| 11 structured output GP | 4 | CI run `41183118583` |\n| 10 robust GP | 2 | CI run `41185454540`; 2 MultiTask variants blocked |\n| 08 fully Bayesian SAAS GP | 4 | CI run `41190752302` |\n| 17 uncertain-input GP | 3 | CI run `41192156012` |

The first four notebooks account for 70 mapped entries. Subsequent
notebooks add 65 verified mappings, for **135 / 142** in total.
The remaining **7** mappings include two blocked robust MultiTask variants.
These counts describe executed model examples, not global
optimization performance or mathematical correctness.

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

## Robust observation notebook coverage (8 registry mappings)

Notebook: `examples/notebooks/15_robust_observation_models.ipynb`.
Commit `1a00372c756cca7e5d890806e3dcf0731bf8532d`
passed CI run `41159977362`, including the default notebook
execution job. All eight mapped classes are instantiated and exercised:

- Student-t: `StudentTSingleTaskGP`, `MixedStudentTSingleTaskGP`,
  `StudentTMultiTaskGP`, `MixedStudentTMultiTaskGP`.
- Contamination: `ContaminatedSingleTaskGP`,
  `MixedContaminatedSingleTaskGP`, `ContaminatedMultiTaskGP`,
  `MixedContaminatedMultiTaskGP`.

The notebook trains via existing `training_loss()` and Adam,
evaluates posterior means, proposes finite-pool candidates, and
evaluates the synthetic objective. The multitask candidates fix the
task feature to task 0. These are posterior-ranking examples, **not
acquisition-optimized BO**. CI execution does not establish
multi-seed robustness or optimization performance.

## MAP-SAAS and additive notebook coverage (7 registry mappings)

Notebook: `examples/notebooks/09_map_saas_and_additive_gp.ipynb`.
Commit `bfc5fdceee767c2e97b88c821d48c99da743d0a5`
passed CI run `41165276107`, including notebook execution.

All seven registry-mapped classes have fit, acquisition optimization,
candidate proposal, and objective evaluation examples:

- `AdditiveMapSaasSingleTaskGP`,
  `EnsembleMapSaasSingleTaskGP`, `OrthogonalAdditiveGP`.
- `MixedAdditiveMapSaasSingleTaskGP`,
  `MixedEnsembleMapSaasSingleTaskGP`, `MixedOrthogonalAdditiveGP`.
- `ALEBOGP` with known observation variance.

Examples use BoTorch `qLogExpectedImprovement` and `optimize_acqf`.
Mixed-input models optimize separately with each observed categorical
value fixed, then choose the highest-acquisition candidate. This
demonstrates acquisition optimization rather than posterior-mean
ranking, but does not establish global optimality or multi-seed
performance.

## Multi-fidelity notebook coverage (7 registry mappings)

Notebook: `examples/notebooks/03_multi_fidelity_gp.ipynb`.
Commit `20543d147197e6c950cacb91b5badaa68c68d2df`
passed CI run `41166885750`, including notebook execution.

The seven mapped classes have fit, posterior, candidate proposal,
and synthetic objective evaluation examples:

- `SingleTaskMultiFidelityGP`: cost-aware
  `qMultiFidelityKnowledgeGradient` with design and fidelity
  jointly optimized; actual evaluation uses the proposed fidelity.
- `AdditiveMapSaasMultiFidelityGP`,
  `MapSaasMultiFidelityGP`, `MixedSingleTaskMultiFidelityGP`,
  `PCAMultiFidelityGP`, `PLSMultiFidelityGP`,
  `RandomProjectionMultiFidelityGP`: target-fidelity
  `qLogExpectedImprovement` optimized with fidelity fixed to 1.
  The mixed model enumerates categorical assignments.

The six additional workflows are **not** cost-aware fidelity-selection
policies. CI validates execution, not multi-seed optimization quality.

## Contextual GP notebook coverage (6 registry mappings)

Notebook: `examples/notebooks/14_contextual_gp.ipynb`.
Commit `1fb0063ac708271bcb69066f9c44b22fa41d035c`
passed CI run `41172687822`, including the notebook jobs.

The six mapped classes have fit, posterior, candidate proposal,
and synthetic objective evaluation examples:

- `SACGP` and `LCEAGP`: aggregated contextual reward;
  finite-pool posterior-mean ranking.
- `LCEMGP` and `MixedLCEMGP`: context-specific output
  selection; mixed candidates retain valid categorical values.
- `HierarchicalConditionalKernelGP` and
  `MixedHierarchicalConditionalKernelGP`: finite candidate
  pools enumerate structural branches and vary active
  parameters while fixing inactive parameters.

These are finite-pool candidate-selection examples, not
acquisition-optimized BO. CI establishes execution only.

## Multi-task GP notebook coverage (5 registry mappings)

Notebook: `examples/notebooks/04_multitask_gp.ipynb`.
Commit `6e6556c00ebbc6aea88d6572231f2326b351392f`
passed CI run `41177219776`, including the notebook job.

- `MultiTaskGP`: target-task `qLogExpectedImprovement`
  over design variables; task identity is not optimized.
- `KroneckerMultiTaskGP`: MC sample scalarization and
  `qLogExpectedImprovement`; observations update all tasks
  at each shared design point.
- `MixedMultiTaskGP`: target-task acquisition, with
  categorical assignments enumerated.
- `MixedKroneckerMultiTaskGP`: MC scalarization with
  categorical assignments enumerated and block-design evaluation.
- `ModelListGP`: independent child GPs fitted with a summed
  marginal log likelihood, then MC-scalarized acquisition.

All five paths include posterior and synthetic objective evaluation.
CI validates execution, not global candidate optimality.

## Nonstationary GP notebook coverage (5 registry mappings)

Notebook: `examples/notebooks/18_nonstationary_gp.ipynb`.
Commit `07d22182dcb3c191ee276209a36de79e03e17b04`
passed CI run `41178611160`, including the notebook job.

- `NonstationarySingleTaskGP`: existing posterior-mean
  candidate-pool ranking and objective evaluation.
- `MixedNonstationarySingleTaskGP`: categorical candidate
  values retained in a finite pool.
- `NonstationaryMultiTaskGP` and
  `MixedNonstationaryMultiTaskGP`: target-task output
  selected from posterior; task is not optimized as an input.
- `NonstationaryKroneckerMultiTaskGP`: target output
  selected from block-design posterior.

The new examples fit models with existing `make_mll()`
and evaluate synthetic objectives. Finite-pool posterior-mean
ranking is not acquisition-function optimization.

## Heterogeneous and hierarchical multi-task coverage (4 mappings)

Notebook: `examples/notebooks/13_heterogeneous_multitask_gp.ipynb`.
Commit `3af753606a37835b50f62fbe8380af2cc45e99bd`
passed CI run `41181618304`. The `notebooks` job log
explicitly records execution start and successful completion.

- `HeterogeneousMTGP`: task-specific feature subsets,
  target-task posterior, candidate evaluation and refitting.
- `MixedHeterogeneousMTGP`: global categorical feature
  index, valid category enumeration and target-task evaluation.
- `HierarchicalConditionalKernelMultiTaskGP`: branch-aware
  candidate pool with target-task posterior selection.
- `MixedHierarchicalConditionalKernelMultiTaskGP`:
  branch-aware pool with an additional categorical variable.

All four use finite-pool posterior-mean candidate selection;
this is not acquisition-function optimization.

## Structured-output GP notebook coverage (4 mappings)

Notebook: `examples/notebooks/11_structured_output_gp.ipynb`.
Commit `6b379c0404c6b78cca10b520d0211f92f8601a40`
passed CI run `41183118583`. The `notebook-structured` job
explicitly executed this notebook and completed successfully.

- `HigherOrderGP` and `LatentKroneckerGP`: existing
  training, structured posterior, scalar spectral-mean
  utility and candidate evaluation.
- `MixedHigherOrderGP` and `MixedLatentKroneckerGP`:
  categorical design dimension with finite category
  enumeration, structured posterior, spectral-mean utility
  and synthetic objective evaluation.

All four select from finite pools using posterior-mean
utilities, not acquisition-function optimization.

## Robust relevance-pursuit notebook (2 verified, 2 blocked)

Notebook: `examples/notebooks/10_robust_gp.ipynb`.
Commit `0e3c0983976a85ea543b816ce93b1799d4ed524b`
passed CI run `41185454540`; the `notebooks` job explicitly
started and completed this notebook.

- Verified: `RobustRelevancePursuitSingleTaskGP` and
  `MixedRobustRelevancePursuitSingleTaskGP` execute model
  fitting, qLogEI acquisition optimization, and candidate
  objective evaluation. Mixed search enumerates categories.
- Blocked: `RobustRelevancePursuitMultiTaskGP` and
  `MixedRobustRelevancePursuitMultiTaskGP` are not counted.
  Fitting the former raised a tensor-dimension RuntimeError
  in the GPyTorch marginal log likelihood. The attempted
  `to_standard_model()` fallback also failed. The latter
  has not been individually executed through fitting.
  Do not infer it has the identical failure without testing.

The notebook explicitly reports these limitations instead
of claiming successful robust MultiTask training. Model-level
fitting repairs remain outside the notebook-update scope.

## Fully Bayesian SAAS notebook coverage (4 mappings)

Notebook: `examples/notebooks/08_saas_gp.ipynb`.
Commit `52b3cb99ef2642533f513f5ded8df8d29c3a081b`
passed CI run `41190752302`. The `notebook-saas` job
explicitly executed this notebook to completion.

- `SaasFullyBayesianSingleTaskGP`: NUTS fitting, posterior,
  qLogEI candidate optimization and synthetic evaluation.
- `SaasFullyBayesianMultiTaskGP`: NUTS fitting with target
  task 1 selected, qLogEI over design variables and evaluation.
- `MixedSaasFullyBayesianSingleTaskGP`: NUTS fitting and
  category-enumerated `optimize_acqf_mixed`.
- `MixedSaasFullyBayesianMultiTaskGP`: NUTS fitting with
  target task 1 and categorical candidate enumeration.

All four examples use deliberately short MCMC runs for CI.
Successful execution does not establish MCMC convergence,
posterior calibration or candidate optimality.

## Uncertain-input GP notebook coverage (3 mappings)

Notebook: `examples/notebooks/17_uncertain_input_gp.ipynb`.
Commit `7a03fb1ca1b14bf9d3d201ecaf2c95fc25aef843`
passed CI run `41192156012`; the default `notebooks` job
explicitly started and completed this notebook.

- `UncertainInputSingleTaskGP`: training-input uncertainty,
  MLL fitting, posterior, qLogEI optimization and evaluation.
- `MixedUncertainInputSingleTaskGP`: continuous training-input
  uncertainty, deterministic categories, MLL fitting and
  category-enumerated qLogEI candidate evaluation.
- `UncertainCategoricalSingleTaskGP`: fitted model with
  category-probability posterior, fixed-probability
  finite-pool posterior-mean selection and evaluation.

The categorical-probability example does not implement
acquisition-function optimization. The other two treat
candidate inputs as deterministic even though training
locations are uncertain. CI execution does not establish
optimization quality or posterior calibration.

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
