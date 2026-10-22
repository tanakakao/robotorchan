# Model theory–implementation mapping: Phase 2 inventory

## Audit provenance and limits

- Baseline: `main` at `a4454c59232f6eb209f4c8cd302d87c7f2233032`.
- Sources: `MODEL_REGISTRY`, `CLASSIFICATION_MODEL_REGISTRY`,
  `docs/model_coverage.json`, public exports, and inspected implementation modules.
- This is a static inventory and representative code-path audit. It is **not**
  numerical or mathematical verification. Full per-class upstream and test-path
  provenance remains to be verified before Phase 2 can be closed.
- Model coverage is generated: never edit `docs/model_coverage.json` by hand.

## Registry boundaries

- Regression-oriented registry: **142** entries in the generated
  coverage artifact. This is the model-registry scope, not all public classes.
- Classification registry: **29** binary model IDs, defined by
  `BINARY_CLASSIFICATION_MODEL_SPECS` and composed by
  `src/robotorchan/models/classification/registry.py`.
- Classification registry IDs and regression class names use different naming
  conventions; neither list should be silently treated as a subset of the other.
- `HeterogeneousModel` is excluded from the regression coverage generator's
  public-model scope and needs separate adapter-layer treatment.
- The public classification package also exports calibration, conformal,
  likelihood, posterior, and reliability helpers; these are not all model IDs.

## Regression model inventory by existing guide

The following tables enumerate **all 142 registry names**.
Each row shares the existing guide; per-name theory and notebook paths remain
canonical in `docs/model_coverage.json` rather than duplicated here.

| Guide | Count | Model registry names |
| --- | ---: | --- |
| `docs/models/expressive.md` | 16 | `InfiniteWidthBNNGP`, `InfiniteWidthBNNKroneckerMultiTaskGP`, `InfiniteWidthBNNMultiTaskGP`, `MixedInfiniteWidthBNNGP`, `MixedInfiniteWidthBNNKroneckerMultiTaskGP`, `MixedInfiniteWidthBNNMultiTaskGP`, `MixedMultiTaskDeepGP`, `MixedSingleTaskDeepGP`, `MixedSpectralMixtureGP`, `MixedSpectralMixtureKroneckerMultiTaskGP`, `MixedSpectralMixtureMultiTaskGP`, `MultiTaskDeepGP`, `SingleTaskDeepGP`, `SpectralMixtureGP`, `SpectralMixtureKroneckerMultiTaskGP`, `SpectralMixtureMultiTaskGP` |
| `docs/models/hierarchical_contextual.md` | 6 | `HierarchicalConditionalKernelGP`, `LCEAGP`, `LCEMGP`, `MixedHierarchicalConditionalKernelGP`, `MixedLCEMGP`, `SACGP` |
| `docs/models/high_dimensional.md` | 12 | `ALEBOGP`, `AdditiveMapSaasMultiFidelityGP`, `AdditiveMapSaasSingleTaskGP`, `EnsembleMapSaasSingleTaskGP`, `MapSaasMultiFidelityGP`, `MixedAdditiveMapSaasSingleTaskGP`, `MixedEnsembleMapSaasSingleTaskGP`, `MixedOrthogonalAdditiveGP`, `MixedSaasFullyBayesianSingleTaskGP`, `OrthogonalAdditiveGP`, `SaasFullyBayesianMultiTaskGP`, `SaasFullyBayesianSingleTaskGP` |
| `docs/models/multitask_multioutput.md` | 10 | `HeterogeneousMTGP`, `HierarchicalConditionalKernelMultiTaskGP`, `KroneckerMultiTaskGP`, `MixedHeterogeneousMTGP`, `MixedHierarchicalConditionalKernelMultiTaskGP`, `MixedKroneckerMultiTaskGP`, `MixedMultiTaskGP`, `MixedSaasFullyBayesianMultiTaskGP`, `ModelListGP`, `MultiTaskGP` |
| `docs/models/non_gp.md` | 5 | `ExtraTreesSurrogate`, `GradientBoostingSurrogate`, `HistGradientBoostingSurrogate`, `NGBoostSurrogate`, `RandomForestSurrogate` |
| `docs/models/preference.md` | 1 | `PairwiseGP` |
| `docs/models/reduced.md` | 52 | `AutoEncoderGP`, `AutoEncoderKroneckerMultiTaskGP`, `AutoEncoderMultiTaskGP`, `HybridAutoEncoderGP`, `HybridAutoEncoderKroneckerMultiTaskGP`, `HybridAutoEncoderMultiTaskGP`, `JointEncoderGP`, `JointEncoderKroneckerMultiTaskGP`, `JointEncoderMultiTaskGP`, `JointVAEGP`, `JointVAEKroneckerMultiTaskGP`, `JointVAEMultiTaskGP`, `MixedAutoEncoderGP`, `MixedHybridAutoEncoderGP`, `MixedJointEncoderGP`, `MixedJointEncoderMultiTaskGP`, `MixedJointVAEGP`, `MixedPCAGP`, `MixedPLSGP`, `MixedRandomProjectionGP`, `MixedReducedGP`, `MixedReducedKroneckerMultiTaskGP`, `MixedReducedMultiTaskGP`, `MixedSupervisedAutoEncoderGP`, `MixedSupervisedVAEGP`, `MixedVAEGP`, `OutputPCAGP`, `OutputPLSGP`, `PCAGP`, `PCAKroneckerMultiTaskGP`, `PCAMultiFidelityGP`, `PCAMultiTaskGP`, `PLSGP`, `PLSKroneckerMultiTaskGP`, `PLSMultiFidelityGP`, `PLSMultiTaskGP`, `RandomProjectionGP`, `RandomProjectionKroneckerMultiTaskGP`, `RandomProjectionMultiFidelityGP`, `RandomProjectionMultiTaskGP`, `ReducedGP`, `ReducedKroneckerMultiTaskGP`, `ReducedMultiTaskGP`, `SupervisedAutoEncoderGP`, `SupervisedAutoEncoderKroneckerMultiTaskGP`, `SupervisedAutoEncoderMultiTaskGP`, `SupervisedVAEGP`, `SupervisedVAEKroneckerMultiTaskGP`, `SupervisedVAEMultiTaskGP`, `VAEGP`, `VAEKroneckerMultiTaskGP`, `VAEMultiTaskGP` |
| `docs/models/robust_noise.md` | 27 | `ContaminatedMultiTaskGP`, `ContaminatedSingleTaskGP`, `HeteroskedasticMultiFidelityGP`, `HeteroskedasticMultiTaskGP`, `HeteroskedasticSingleTaskGP`, `JointHeteroskedasticSingleTaskGP`, `MixedContaminatedMultiTaskGP`, `MixedContaminatedSingleTaskGP`, `MixedHeteroskedasticMultiTaskGP`, `MixedHeteroskedasticSingleTaskGP`, `MixedJointHeteroskedasticSingleTaskGP`, `MixedNonstationaryMultiTaskGP`, `MixedNonstationarySingleTaskGP`, `MixedReplicateNoiseSingleTaskGP`, `MixedRobustRelevancePursuitMultiTaskGP`, `MixedRobustRelevancePursuitSingleTaskGP`, `MixedStudentTMultiTaskGP`, `MixedStudentTSingleTaskGP`, `NonstationaryKroneckerMultiTaskGP`, `NonstationaryMultiTaskGP`, `NonstationarySingleTaskGP`, `ReplicateNoiseMultiFidelityGP`, `ReplicateNoiseSingleTaskGP`, `RobustRelevancePursuitMultiTaskGP`, `RobustRelevancePursuitSingleTaskGP`, `StudentTMultiTaskGP`, `StudentTSingleTaskGP` |
| `docs/models/standard.md` | 6 | `MixedSingleTaskGP`, `MixedSingleTaskMultiFidelityGP`, `MixedSingleTaskVariationalGP`, `SingleTaskGP`, `SingleTaskMultiFidelityGP`, `SingleTaskVariationalGP` |
| `docs/models/structured_output.md` | 4 | `HigherOrderGP`, `LatentKroneckerGP`, `MixedHigherOrderGP`, `MixedLatentKroneckerGP` |
| `docs/models/uncertain_input.md` | 3 | `MixedUncertainInputSingleTaskGP`, `UncertainCategoricalSingleTaskGP`, `UncertainInputSingleTaskGP` |

## Binary classification registry inventory

These are stable registry **IDs**, not public class names.

| Registry ID | Audit status |
| --- | --- |
| `binary.ensemble.gp` | Implementation/test/upstream provenance pending |
| `binary.ensemble.heterogeneous` | Implementation/test/upstream provenance pending |
| `binary.standard` | Implementation/test/upstream provenance pending |
| `binary.mixed` | Implementation/test/upstream provenance pending |
| `binary.multitask` | Implementation/test/upstream provenance pending |
| `binary.kronecker_multitask` | Implementation/test/upstream provenance pending |
| `binary.map_saas` | Implementation/test/upstream provenance pending |
| `binary.saas` | Implementation/test/upstream provenance pending |
| `binary.reduced` | Implementation/test/upstream provenance pending |
| `binary.pca` | Implementation/test/upstream provenance pending |
| `binary.pls` | Implementation/test/upstream provenance pending |
| `binary.random_projection` | Implementation/test/upstream provenance pending |
| `binary.alebo` | Implementation/test/upstream provenance pending |
| `binary.joint_encoder` | Implementation/test/upstream provenance pending |
| `binary.deep_gp` | Implementation/test/upstream provenance pending |
| `binary.uncertain.categorical_input` | Implementation/test/upstream provenance pending |
| `binary.uncertain.continuous_input` | Implementation/test/upstream provenance pending |
| `binary.robust.label_noise` | Implementation/test/upstream provenance pending |
| `binary.robust.contaminated` | Implementation/test/upstream provenance pending |
| `binary.robust.replicate_labels` | Implementation/test/upstream provenance pending |
| `binary.robust.input_dependent_label_noise` | Implementation/test/upstream provenance pending |
| `binary.robust.nonstationary` | Implementation/test/upstream provenance pending |
| `binary.robust.mixed_nonstationary` | Implementation/test/upstream provenance pending |
| `binary.robust.multitask_nonstationary` | Implementation/test/upstream provenance pending |
| `binary.non_gp.bootstrap_gradient_boosting` | Implementation/test/upstream provenance pending |
| `binary.non_gp.random_forest` | Implementation/test/upstream provenance pending |
| `binary.non_gp.extra_trees` | Implementation/test/upstream provenance pending |
| `binary.non_gp.gradient_boosting` | Implementation/test/upstream provenance pending |
| `binary.non_gp.hist_gradient_boosting` | Implementation/test/upstream provenance pending |

## Representative code-path findings

| Public model | Implementation | Confirmed upstream / local boundary | Evidence level |
| --- | --- | --- | --- |
| `SingleTaskGP` | `models/standard/single_task.py` | Subclasses BoTorch `SingleTaskGP`; local raw-data and fit conventions | Source inspected; runtime not rechecked |
| `MixedSingleTaskGP` | `models/standard/single_task.py` | Subclasses BoTorch `MixedSingleTaskGP`; local conventions | Source inspected; runtime not rechecked |
| `BinarySingleTaskGPClassifier` | `models/classification/binary/standard/single_task.py` | Extends local variational GP with GPyTorch Bernoulli likelihood and variational ELBO | Source inspected; latent/probability math not independently verified |
| `ReducedGP` | `models/high_dimensional/reduced/base.py` | Local reducer transforms around BoTorch `SingleTaskGP`; output posterior reconstruction | Source inspected; covariance restoration not independently verified |
| `RandomForestSurrogate` | `models/non_gp/random_forest.py` | Local BoTorch-compatible posterior facade over scikit-learn `RandomForestRegressor` | Source inspected; uncertainty calibration not verified |

Implementation ownership labels must be assigned **per algorithmic component**:
for example, a wrapper can be `Extended` at the public API boundary while
`Delegated` for its GP posterior. Do not mark the entire model as `Implemented`
merely because a local subclass exists.

## Existing audit machinery

`scripts/audit_model_api.py` already enumerates both registries, extracts
class ownership and constructor/method signatures, and reports static contract
mismatches. It explicitly does **not** prove runtime support. Reuse it instead
of adding a second inventory generator:

```bash
python scripts/audit_model_api.py --output /tmp/robotorchan-model-api-audit.json
python scripts/generate_model_coverage.py --check
```


## Phase 2 public-export reconciliation

Static comparison against the latest `main` at this review found:

- Regression registry: **142** entries, all present in `models.__all__`.
- Regression public exports: **154** names; the remaining
  **12** are infrastructure and heterogeneous-layer symbols,
  not unregistered regression surrogate classes.
- Classification public exports: **76** names; all
  **29** binary registry classes are publicly exported.
- Registry entries and exports are *not* a runtime or mathematical proof.

### Regression public symbols outside the registry

- `DocumentationLinks`
- `HeterogeneousModel`
- `HighDimensionalStrategy`
- `InferenceType`
- `InputType`
- `ModelCapabilities`
- `ModelRegistryEntry`
- `PosteriorSamplingType`
- `RobustnessType`
- `TaskType`
- `UnsupportedModelOperationError`
- `get_model_registry_entry`

### Binary registry IDs and exported model classes

| Registry ID | Exported class |
| --- | --- |
| `binary.ensemble.gp` | `GPBinaryClassificationEnsemble` |
| `binary.ensemble.heterogeneous` | `HeterogeneousBinaryClassificationEnsemble` |
| `binary.standard` | `BinarySingleTaskGPClassifier` |
| `binary.mixed` | `MixedBinarySingleTaskGPClassifier` |
| `binary.multitask` | `MultiTaskBinaryGPClassifier` |
| `binary.kronecker_multitask` | `KroneckerMultiTaskBinaryGPClassifier` |
| `binary.map_saas` | `MapSaasBinarySingleTaskGPClassifier` |
| `binary.saas` | `SaasBinarySingleTaskGPClassifier` |
| `binary.reduced` | `ReducedBinarySingleTaskGPClassifier` |
| `binary.pca` | `PCABinarySingleTaskGPClassifier` |
| `binary.pls` | `PLSBinarySingleTaskGPClassifier` |
| `binary.random_projection` | `RandomProjectionBinarySingleTaskGPClassifier` |
| `binary.alebo` | `ALEBOBinarySingleTaskGPClassifier` |
| `binary.joint_encoder` | `JointEncoderBinaryGPClassifier` |
| `binary.deep_gp` | `BinarySingleTaskDeepGPClassifier` |
| `binary.uncertain.categorical_input` | `UncertainCategoricalBinarySingleTaskGPClassifier` |
| `binary.uncertain.continuous_input` | `ContinuousUncertainInputBinarySingleTaskGPClassifier` |
| `binary.robust.label_noise` | `LabelNoiseBinarySingleTaskGPClassifier` |
| `binary.robust.contaminated` | `ContaminatedBinarySingleTaskGPClassifier` |
| `binary.robust.replicate_labels` | `ReplicateLabelBinarySingleTaskGPClassifier` |
| `binary.robust.input_dependent_label_noise` | `InputDependentLabelNoiseBinarySingleTaskGPClassifier` |
| `binary.robust.nonstationary` | `NonstationaryBinarySingleTaskGPClassifier` |
| `binary.robust.mixed_nonstationary` | `MixedNonstationaryBinarySingleTaskGPClassifier` |
| `binary.robust.multitask_nonstationary` | `NonstationaryMultiTaskBinaryGPClassifier` |
| `binary.non_gp.bootstrap_gradient_boosting` | `BootstrapGradientBoostingBinaryClassifier` |
| `binary.non_gp.random_forest` | `RandomForestBinaryClassifier` |
| `binary.non_gp.extra_trees` | `ExtraTreesBinaryClassifier` |
| `binary.non_gp.gradient_boosting` | `GradientBoostingBinaryClassifier` |
| `binary.non_gp.hist_gradient_boosting` | `HistGradientBoostingBinaryClassifier` |

The binary registry does not encode per-ID theory or notebook links.
These links must be audited separately; absence of metadata is not evidence
that a model has no theoretical basis or examples.


## Representative test assertion mapping

The following are **inspected test files**, not merely directory matches.
They provide evidence for the listed contracts, but do not prove all
mathematical properties or cover every registered model.

| Model | Test source | Verified assertion scope |
| --- | --- | --- |
| `MixedSingleTaskGP` | `tests/models/standard/test_mixed_single_task_gp.py` | Upstream constructor surface, wrapper contract, upstream posterior comparison |
| `BinarySingleTaskGPClassifier` | `tests/models/classification/test_binary.py` | Bernoulli likelihood, latent/probability API, ELBO construction, shapes, Bernoulli variance and entropy |
| `ReducedGP` | `tests/models/high_dimensional/reduced/test_reduced_gp.py` | Original/latent input contract, posterior q-batches, observation conditioning, output reconstruction, state round-trip |
| `RandomForestSurrogate` | `tests/models/non_gp/test_random_forest_surrogate.py` | Ensemble posterior, reproducibility, batched candidates, MC qEI integration |
| `SingleTaskGP` and mixed exact GP | `tests/models/contracts/test_exact_gp_fit.py` | Native MLL/fit behavior, categorical kernel, direct BoTorch MLL interoperability |

**Not established by these tests:** calibrated RF predictive intervals,
posterior approximation error bounds, reduction-induced covariance error,
variational classifier probability calibration, and correctness of all
142 regression / 29 binary classification models. Those claims remain
`Unverified` until a relevant numerical assertion is identified or added
in the mathematical verification track (Phases 7–9).


## Component-level implementation ownership

Source imports and class inheritance were inspected for the following
representative paths. Ownership is **per component**, not per model name.

| Model | Local responsibility | Delegated responsibility |
| --- | --- | --- |
| `SingleTaskGP` | `FittableExactGPModelMixin` fit/data contract | BoTorch `SingleTaskGP` GP and posterior |
| `MixedSingleTaskGP` | Local fit/data contract | BoTorch `MixedSingleTaskGP` categorical GP |
| `BinarySingleTaskGPClassifier` | Binary label validation and predictive probability interface | Local `SingleTaskVariationalGP`, GPyTorch `BernoulliLikelihood` and `VariationalELBO` |
| `ReducedGP` | `ReductionMixin`, input/output reducer and public posterior handling | BoTorch `SingleTaskGP` latent-space GP |
| `RandomForestSurrogate` | BoTorch `Model` facade and ensemble-posterior adapter | scikit-learn random-forest estimator (runtime import) |

These findings support `Extended` and `Delegated` **at different
boundaries**. They do not establish that the local wrapper owns the upstream
algorithm. For `RandomForestSurrogate`, the sklearn estimator is imported
within implementation code rather than from the module-level imports.

### Limits of the current evidence

The full 171-entry registry is inventoried, but these five source paths are
representative rather than exhaustive. The existing
`scripts/audit_model_api.py` produces per-class module paths and method
owners for all registry entries without fitting them; use its temporary JSON
report to finish the exhaustive ownership review. Its method-owner output
must not be interpreted as a numerical validation of a posterior or fit.


## Exhaustive static provenance export

The existing `scripts/audit_model_api.py` now emits these additional fields
for **every registered regression and classification model**:

- `implementation_module`: Python module defining the public class.
- `upstream_bases`: BoTorch/GPyTorch/sklearn classes in the static MRO.
- `provenance_verification`: `static_mro_only`, never a runtime claim.
- `documentation`: regression registry guide/theory/notebook paths,
  tagged `registry_metadata_only` (not link or execution validation).
- Classification entries explicitly say
  `not_recorded_in_classification_registry` for documentation metadata.

Run `python scripts/audit_model_api.py --output /tmp/model-api-audit.json`
to review the full 171-entry snapshot without committing generated files.
An empty `upstream_bases` list **does not** imply a fully native algorithm:
composition, runtime imports, and delegated estimators (e.g. sklearn forests)
are invisible to MRO inspection. Inspect implementation source to determine
component-level ownership. The report is a traceability aid, not proof of
correctness, training support, or mathematical equivalence.

## Open verification items (Phase 2 exit criteria)

1. Public registry/export name reconciliation completed statically above.
2. For **every** model ID, resolve the concrete class/module and any upstream
   base or delegated backend; distinguish implementation from inheritance.
3. Resolve the actual test files and test assertions, not just directory names.
4. Confirm the theory/guide/notebook links for each model; note shared notebooks
   and models without individual executable demonstrations.
5. Run the existing model API audit, coverage check, and CI; triage any gaps.
6. Document unsupported and unverified mathematical properties explicitly.

Do not advance to Phase 3 or declare Phase 2 complete while these remain open.
