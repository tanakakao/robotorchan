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


## Public export reconciliation (static)

The regression module exports **154** names through `models.__all__`.
Exactly **142** are represented in `MODEL_REGISTRY`. The remaining **12**
are public infrastructure symbols, including `HeterogeneousModel`, not missing
GP regression entries. Explicitly preserve its separate heterogeneous adapter
mapping; do not add it to the generated regression coverage just to balance counts.

The classification package exports **76** names, which include
likelihoods, mixins, calibration, conformal utilities, and registry helpers.
All **29** binary registry model classes are present in the package's
public export list. The remaining exports must not be counted as 47 unregistered
classification models.

### Classification model ID → public implementation class

| Stable model ID | Public class |
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

### Regression exports outside the regression model registry

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

## Open verification items (Phase 2 exit criteria)

1. Reconciliation of public exports and both registries: complete at the static
   name level; confirm any runtime aliases or re-exports if found.
2. For **every** model ID, resolve the concrete class/module and any upstream
   base or delegated backend; distinguish implementation from inheritance.
3. Resolve the actual test files and test assertions, not just directory names.
4. Confirm the theory/guide/notebook links for each model; note shared notebooks
   and models without individual executable demonstrations.
5. Run the existing model API audit, coverage check, and CI; triage any gaps.
6. Document unsupported and unverified mathematical properties explicitly.

Do not advance to Phase 3 or declare Phase 2 complete while these remain open.
