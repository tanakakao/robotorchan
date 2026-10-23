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

## Documentation link existence validation

The static audit now supports:

```bash
python scripts/audit_model_api.py --check-documentation-paths
python scripts/generate_model_coverage.py --check
```

The optional check validates that the guide, theory, and notebook files
referenced by **every regression registry entry** exist in the repository.
It reports missing paths and exits nonzero if any are absent. It does not
execute notebooks, validate theory content, or check classification-model
links (which are not encoded in that registry).

## Static test-reference discovery

The existing audit can now discover syntactic references to all 171 model
registry identifiers / class names in Python test files:

```bash
python scripts/audit_model_api.py --include-test-references --output model-test-references.json
```

The `test_references` result lists matching test file paths and explicitly
labels them `syntactic_reference_only`. AST matching ignores comments and
string literals but can still produce false positives (unrelated names) and
false negatives (aliases, parametrization, dynamically constructed classes).
An empty list is **not** evidence of missing tests; a nonempty list is **not**
evidence of mathematical verification. Assertions and actual executed model
paths still require human inspection and, where needed, numerical tests.
The generated JSON is a temporary review artifact, not a tracked document.

## Inspected runtime composition and mathematical ownership

The following paths were checked directly against implementation source on
`main`. These findings supplement the MRO inventory: an upstream base class
alone does not reveal internal estimators, likelihoods, or reducers.

| Model | Local responsibilities | Delegated numerical components | Verification still needed |
| --- | --- | --- | --- |
| `RandomForestSurrogate` | Validate shapes and categories; preserve raw tensors; convert per-tree predictions into a BoTorch-compatible ensemble posterior | `sklearn.ensemble.RandomForestRegressor` fits the forest and individual sklearn trees compute predictions; `make_ensemble_posterior` is a local posterior adapter | Ensemble calibration and correspondence of tree spread to predictive uncertainty |
| `BinarySingleTaskGPClassifier` | Validate binary labels; retain raw labels; expose predictive distribution and class probabilities; restore variational state | `SingleTaskVariationalGP` provides latent GP infrastructure; GPyTorch `BernoulliLikelihood`, `VariationalStrategy`, and `VariationalELBO` provide probabilistic inference components | ELBO numerical reference, probability calibration, and latent-to-label integration accuracy |
| `ReducedGP` | Fit/reuse reducers; transform input data and candidate locations; reconstruct public output posterior | BoTorch `SingleTaskGP` supplies exact GP machinery; configured `InputReducer`/`OutputReducer` implementations supply reductions | Reconstruction covariance, approximation error, and behavior under each reducer family |

**Interpretation:** `RandomForestSurrogate` has no sklearn class in its MRO,
because it composes `RandomForestRegressor` in `self._forest`. The forest is
fitted in `fit()`; `posterior()` gathers predictions from
`self._forest.estimators_`. The ensemble is an empirical approximation,
not an analytically derived GP posterior. Similarly, the classifier's
Bernoulli likelihood is constructed as an instance, and reduced GP behavior
depends on the configured reducer. None of these facts can be inferred from
`upstream_bases` alone.

These three cases have **source-inspected** component ownership, not exhaustive
runtime tracing. Other registered models remain `static_mro_only` until their
composition and method delegation have been inspected. This section does not
claim that any mathematical property has passed a numerical reference test.

## Inspected test assertions and verification boundaries

The following test modules were inspected individually, rather than inferred
from AST references. Test counts describe named test functions in the files;
they are **not** a measure of mathematical proof or model-family coverage.

| Test module | Named tests | Inspected assertion scope | Not established |
| --- | ---: | --- | --- |
| `tests/models/standard/test_mixed_single_task_gp.py` | 3 | Constructor signature compatibility with upstream BoTorch; shared raw-data/MLL contract; posterior comparison with upstream implementation | Kernel equivalence across all categorical settings, numerical optimization accuracy |
| `tests/models/non_gp/test_random_forest_surrogate.py` | 6 | Raw training-data and fit contracts; tree ensemble posterior; deterministic seed; batched candidates; MC qEI compatibility; error handling | Calibrated uncertainty or statistically consistent predictive intervals |
| `tests/models/classification/test_binary.py` | 20 | Bernoulli likelihood and ELBO API; latent/predictive shapes; label/threshold validation; probability and class outputs; Bernoulli variance/entropy identities; latent and probability sampling | Trained probability calibration, convergence of variational inference, ELBO optimum |
| `tests/models/high_dimensional/reduced/test_reduced_gp.py` | 13 | Raw versus reduced inputs; q-batch posterior; reducer reuse; conditioning; MLL contract; named reducer selection; output restoration; explicit-noise guards; serialization round-trip | Bounds on dimensionality-reduction error, covariance reconstruction accuracy across all reducers |

The assertion scopes above were derived from the named tests and inspected
source, not from filename matches alone. Some assertions check equality with
an upstream implementation or elementary probability identities, but neither
establishes a universal mathematical guarantee. In particular, the
classification tests exercise the Bernoulli variance and entropy formulas,
not empirical calibration; the forest tests exercise ensemble construction,
not calibrated epistemic uncertainty. All remaining registry models require
comparable assertion-level inspection before Phase 2's exhaustive test mapping
can be considered complete.

## Optional static upstream composition scan

Run `python scripts/audit_model_api.py --include-upstream-calls --output model-review.json`
to list imported BoTorch, GPyTorch, and scikit-learn call targets
syntactically present inside each registered model class. This supplements
`upstream_bases`, which reports inheritance only. The scan intentionally
excludes helper-function delegation, runtime aliases, inherited method bodies,
and indirect factories. Empty `upstream_calls` means **not detected**, not
that a model does not delegate. Results are marked
`static_direct_import_call_only` and must not be treated as runtime traces
or proof that any upstream method executed.

## Conditional import provenance and static scan boundaries

The Phase 2 static upstream-call scanner is now covered by regression tests for
optional imports inside `try/except ImportError`, aliased imports, and strings
that merely mention a constructor. In particular, the source of
`RandomForestSurrogate` imports `RandomForestRegressor` conditionally and
instantiates it inside the model constructor. The classifier directly
constructs a GPyTorch `BernoulliLikelihood`; these are concrete source-level
composition examples, not inherited sklearn/GPyTorch model classes.

The scanner does not resolve runtime reassignment, monkey-patched references,
indirect helper calls, or methods inherited from upstream implementations.
A syntactic match does not prove that the call executes on every code path.
The scanner's test suite checks detection behavior, **not** estimator fit,
posterior numerical accuracy, or likelihood calibration. Keep those separate
from Phase 7's mathematical verification work.

## Phase 2 evidence levels and follow-up matrix

The following evidence levels must not be conflated when reporting model
coverage. A registry entry is an inventory item, not a verified numerical
implementation. The static test-reference inventory establishes that a name
occurs in a test AST; it does not establish which assertion exercises that
model, whether the test executes, or whether the expected value is correct.

| Evidence level | Current source | What it establishes | What it cannot establish |
| --- | --- | --- | --- |
| Registry and export | Model registries and public `__all__` | Publicly advertised class identifiers | Instantiation or fit success |
| Inheritance | `upstream_bases` and `static_mro_only` | Declared BoTorch/GPyTorch ancestry | Composed estimator ownership or executed delegation |
| Direct composition | `upstream_calls` and `static_direct_import_call_only` | Syntactic calls to imported upstream objects | Indirect calls, actual execution, or numerical fidelity |
| Test reference | `test_references` and `syntactic_reference_only` | AST-level model references in test files | Assertion coverage or test success |
| Inspected assertions | The four test modules documented above | Exact assertions in selected source files | Mathematical validity outside those cases |
| Numerical verification | Future targeted reference/identity tests | Behavior for specified data, tolerances, and seeds | Universal guarantees or empirical calibration |

**Immediate Phase 2 review priorities:** inspect assertions for model families
not represented by the four source-reviewed test modules; reconcile
classification models with guide/theory/notebook references (the regression
registry's path checker does not cover them); and review delegated behavior
through local helper functions, which the direct-call scanner intentionally
omits. Record missing evidence as `Unverified` rather than treating an empty
scan as `Unsupported`. Numerical identity and statistical validation belong
to Phase 7; Phase 2 identifies the specific gaps they must address.

## Classification model documentation ownership

Unlike the regression registry, the classification registry contains no
per-model `guide`, `theory`, or `notebook` path fields. All 29 public binary
model IDs are composed from `BINARY_CLASSIFICATION_MODEL_SPECS` in
`src/robotorchan/models/classification/binary/registry.py`. Therefore the
regression-only `--check-documentation-paths` check cannot be cited as
validation of classification documentation coverage.

The shared entry point is `docs/models/classification.md`, which distinguishes
latent GP posterior, predictive class probabilities, robust observation
channels, non-GP classifiers, and ensembles. It links to
`docs/theory/24_classification.md` for GP classification theory and
`docs/theory/acquisition/classification-active-learning.md` for acquisition
semantics. These are **shared family references**, not verified per-model
mathematical derivations or proof that each model has a runnable notebook.

| Registry family | Source of model IDs | Documentation interpretation |
| --- | --- | --- |
| standard / multitask | `BINARY_CLASSIFICATION_MODEL_SPECS` | Shared classification guide and latent-versus-label contract |
| high_dimensional / expressive | `BINARY_CLASSIFICATION_MODEL_SPECS` | Family overview; reduction, embedding, and deep approximations require individual checks |
| robust / uncertain | `BINARY_CLASSIFICATION_MODEL_SPECS` | Label-noise and uncertain-input semantics require model-specific verification |
| non_gp / ensemble | `BINARY_CLASSIFICATION_MODEL_SPECS` | Probability adapters and ensemble uncertainty are not Gaussian latent inference |

**Outstanding:** resolve the relevant theory section and executable example
for each classification ID, inspect its assertion-level tests, and record
missing demonstrations explicitly. A shared guide link is not sufficient to
close these items.

## Source-inspected classification delegation: non-GP and label noise

Two additional classification paths were inspected directly on `main`:

| Implementation | Local numerical contract | Delegation and mathematical object | Unverified boundary |
| --- | --- | --- | --- |
| `classification/binary/non_gp/sklearn.py`: `SklearnBinaryClassifier` and four subclasses | Validate canonical 0/1 labels, preserve raw tensors, fit a stored estimator, convert `predict_proba` to candidate batch shape | The four subclasses assign sklearn `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, or `HistGradientBoostingClassifier` to `estimator_class`; the base class constructs and calls the estimator | Per-estimator class-order and calibration behavior; no Gaussian latent posterior follows from this adapter |
| `classification/binary/robust/label_noise.py`: `LabelNoiseBinarySingleTaskGPClassifier` | Validate flip rates, replace the base classifier likelihood, distinguish clean from observed-label predictions | GPyTorch `BernoulliLikelihood` supplies clean probabilities; `LabelNoiseBernoulliLikelihood` maps them through an explicit label-flip channel and integrates log probability by quadrature | Numerical quadrature accuracy, learned flip-rate identifiability, and calibration of clean versus corrupted predictions |

For false-positive rate `a`, false-negative rate `b`, and clean positive
probability `p`, the local label-noise transform is
`p_observed = a + (1 - a - b) * p`. The implementation applies this
transform to both latent-sample and posterior-predictive likelihood paths.
This equation is **source-inspected**, not yet checked against an independent
numerical reference. The sklearn base class constructs the estimator through
`self.estimator_class`; the current direct-import call scanner does not
resolve that class-attribute indirection. Its empty result for an individual
sklearn subclass must not be interpreted as an absence of upstream delegation.

## Source-inspected classification: heterogeneous ensembles and uncertain inputs

| Implementation | Local mathematical object | Delegated behavior | Remaining verification |
| --- | --- | --- | --- |
| `classification/binary/ensemble/heterogeneous.py` | Weighted empirical distribution over member class-probability vectors; its mean is the reported `predict_proba` and its variance measures between-member disagreement | Calls each member's `predict_proba`; delegates probability-posterior moments and entropy to `ClassificationEnsemblePosterior` | Weight normalization, entropy/MI identities, and numerical agreement across heterogeneous member backends |
| `classification/binary/uncertain/uncertain_input.py` | Gaussian candidate perturbations and Monte Carlo average of `predict_proba(X + epsilon)` | Inherits the binary GP classifier's predictive probabilities; uses PyTorch Cholesky decomposition and Gaussian draws for perturbations | Covariance validation, reproducibility, Monte Carlo convergence, and near-singular covariance behavior |

The heterogeneous ensemble intentionally raises `NotImplementedError` for
`latent_posterior` and `sample_latent`: its members do not share a single
latent space. The empirical probability posterior must not be relabeled as a
GP latent posterior. For uncertain continuous inputs, the code samples
`epsilon ~ N(0, Sigma)` with a machine-epsilon diagonal jitter and estimates
`E_epsilon[P(y | X + epsilon, D)]` by averaging member probability vectors.
These are **source-inspected algorithms**, not evidence of numerical convergence
or of calibrated epistemic uncertainty. Both require assertion-level test
mapping and independent numerical checks before verification can be claimed.

## Inspected classification test assertions: ensemble and uncertain input

These two test files were read in full on `main`, adding **12 named test
functions** to the source-inspected subset. These observations supplement,
not replace, the AST-level `test_references` inventory.

| Test module | Named tests | Verified by assertions | Not established by these tests |
| --- | ---: | --- | --- |
| `tests/models/classification/ensemble/test_heterogeneous.py` | 7 | GP-plus-fitted-forest probability composition; unweighted mean and population variance; rejection of shared latent methods; compatibility matrix for probability AL versus LatentStraddle; backend-neutral registry flags; weighted mean and normalized weights; state-dict persistence; q=1 candidate-axis preservation | Entropy and mutual-information numerical identities; calibrated probability estimates; ensembles with more than two heterogeneous backend families |
| `tests/models/classification/uncertain/test_continuous_input.py` | 5 | Uncertainty metadata; zero-input-std agreement with baseline probability (tolerance 1e-6); probability simplex and output shape for covariance input; finite candidate gradients; registry robustness capability | Monte Carlo convergence versus independent integration; multivariate correlated covariance cases; non-PSD rejection, near-singular stability, or device portability |

**Numerical scope:** The heterogeneous ensemble's mean and variance are
compared with explicitly stacked member probabilities in the test source.
The uncertain-input zero-std test provides a useful limiting case, but the
implementation adds a small diagonal jitter before Cholesky sampling; the
reported tolerance and sample count describe a specific finite-data test,
not exact equality for all settings. The nonzero-covariance test verifies
simplex preservation, not accuracy against a Gaussian integral reference.

## Inspected classification test assertions: label noise and sklearn adapters

The following two modules were inspected directly on `main`. Each contains
**five and four named test functions**, respectively; parametrization expands
the actual executed cases and must not be confused with function counts.

| Test module | Named tests | Assertion evidence | Unverified mathematical or statistical behavior |
| --- | ---: | --- | --- |
| `tests/models/classification/robust/test_label_noise.py` | 5 | Explicit asymmetric label-flip formula at three reference probabilities; symmetric flip-rate configuration and clean-to-observed positive-probability relationship; gradients for learnable flip logits; invalid-rate rejection (four parametrized rates); exclusion of simultaneous symmetric/asymmetric settings | Identifiability of flip parameters, quadrature accuracy for training likelihood, and probability calibration |
| `tests/models/classification/non_gp/test_sklearn.py` | 4 | Fitted output simplex, bounds, entropy/variance shapes for each of four sklearn classifier families (one parametrized test); rejection of unsupported epistemic probability sampling; fit-before-predict guard; registry posterior-sampling capability flags | Proper calibration, treatment of single-class training sets, class-order edge cases, and predictive performance versus sklearn references |

The label-noise test asserts the source formula
`p_observed = a + (1 - a - b) * p_clean` for
`(a, b) = (0.1, 0.2)` and `p_clean = (0, 0.25, 1)`.
This is an explicit fixed-value mathematical assertion, not evidence that
the numerical quadrature or fitted classifier is statistically calibrated.
The sklearn parametrized probability-contract test runs against four estimator
classes, but does not compare each model's predictions to a standalone
sklearn reference. These boundaries should inform targeted Phase 7 tests.

## Source and assertion audit: binary multitask and DeepGP

Three additional classification test modules were inspected in full, with
**16 named test functions** (6 long-format multitask, 4 block-design
multitask, 6 DeepGP). The DeepGP threshold test is parametrized for three
nonfinite values; named test counts do not represent executed case counts.

| Implementation and tests | Mathematical implementation and delegated operations | Confirmed assertions | Not confirmed |
| --- | --- | --- | --- |
| `classification/binary/standard/multitask.py` / `test_multitask_binary.py` | Local structural task-feature validation; data covariance from BoTorch's `get_covar_module_with_dim_scaled_prior` multiplied by GPyTorch `IndexKernel` via `ProductKernel`; variational GP and Bernoulli inference inherited from single-task classifier | Product and IndexKernel structure, negative feature index normalization, rank and task-ID validation, raw long-format data, predicted probability simplex and sample shape | Correct task-covariance values, cross-task correlation and inference quality |
| `classification/binary/standard/multitask.py` / `test_kronecker_binary.py` | Local block-to-long format reshaping, task-axis expansion, and return to all-task predictions; **uses the same variational ICM backend**, not an independently implemented Kronecker Gaussian inference routine | Preserves block-design raw X/Y, per-task output shapes and probability simplex, rejects invalid block shapes, preserves boolean labels | Equivalence with explicit long-format predictions or algebraic Kronecker structure and scaling |
| `classification/binary/expressive/deep_gp.py` / `expressive/test_deep_gp_classifier.py` | Inherits `SingleTaskDeepGP`; locally adds a GPyTorch Bernoulli observation likelihood, wraps `VariationalELBO` in `DeepApproximateMLL`, averages likelihood probabilities over latent posterior samples | Latent/probability shapes, finite scalar training loss, Bernoulli deep ELBO type, raw data contract, probability sample simplex, common classification metadata and invalid-label/threshold guards | ELBO reference equivalence, MC convergence, probability calibration, and trained predictive quality |

In the block-design classifier, `KroneckerMultiTaskBinaryGPClassifier`
expands `n x d` inputs and `n x m` labels to `(n*m) x (d+1)`
inputs with an explicit task column and `n*m` labels, then delegates
to `MultiTaskBinaryGPClassifier`. The name must not be interpreted as
evidence that exact Kronecker matrix algebra runs in this classification
model. This is an important distinction between naming and implemented
numerical algorithm.

## Source and assertion audit: high-dimensional binary classifiers

Four implementation modules and four corresponding test modules were inspected
on `main`. These contain **13 named test functions** (3 reduction, 4 SAAS,
3 MAP-SAAS, 3 ALEBO); the reduction test is parametrized over three reducers.

| Model family and tests | Local mathematical or algorithmic responsibility | Upstream delegation and checked assertions | Still unverified |
| --- | --- | --- | --- |
| `high_dimensional/reduced/base.py` / `high_dimensional/reduced/test_reduced.py` | Fit or reuse a frozen input reducer; train the latent binary GP on transformed inputs; accept original or latent candidate coordinates | PCA, PLS, and random projection use `robotorchan.reduction.input` reducers; tests compare original versus reduced probability outputs for all three, check raw tensor preservation and reject wrong dimensions | Reduction-induced predictive distortion, supervised PLS leakage in evaluation, unseen-data behavior, and numerical projection accuracy |
| `high_dimensional/saas.py` / `high_dimensional/test_saas.py` | Register Half-Cauchy shrinkage on inverse ARD lengthscales of a Matérn-5/2 kernel; use the base variational binary GP | GPyTorch kernels and prior; tests check prior targets reciprocal lengthscale, finite raw-lengthscale ELBO gradient, probability simplex and positive scale validation | Sparse feature recovery, prior posterior behavior, calibrated predictions; this is **not** a NUTS/Pyro fully Bayesian classifier |
| `high_dimensional/map_saas.py` / `high_dimensional/test_map_saas.py` | Create BoTorch ARD covariance and attach a GPyTorch Half-Cauchy inverse-lengthscale prior | Tests check registered prior name, predicted class simplex and positive `tau` guard | Whether an optimization run finds a MAP solution, active-dimension recovery, and fit convergence |
| `high_dimensional/alebo.py` / `high_dimensional/test_alebo_classifier.py` | Apply `MahalanobisRBFKernel` to embedded coordinates with an optional projection; inherit variational Bernoulli prediction | Local ALEBO kernel, `ALEBOStrategy` for training examples and BoTorch-compatible binary GP inference; tests check probability simplex, metric size, finite Mahalanobis-parameter ELBO gradient and projection shape guard | Validity of ambient-space feasibility, convergence of embedded optimization, and metric learning accuracy |

The SAAS and MAP-SAAS names denote distinct covariance construction and
prior configuration; their test assertions do **not** establish a fully
Bayesian NUTS posterior. Likewise, the reduced classifier's equality between
original- and reduced-coordinate predictions checks a code-path identity,
not lossless dimension reduction. The ALEBO test differentiates an ELBO
through its Mahalanobis parameter, but does not benchmark candidate-search
quality. These explicit verification boundaries are inputs to Phase 7,
not Phase 2 numerical pass claims.

## Open verification items (Phase 2 exit criteria)

1. Public registry/export name reconciliation completed statically above.
2. For **every** model ID, resolve the concrete class/module and any upstream
   base or delegated backend; distinguish implementation from inheritance.
3. Resolve the actual test files and test assertions, not just directory names.
4. Confirm the theory/guide/notebook links for each model; note shared notebooks
   and models without individual executable demonstrations.
5. Run the model API audit, documentation-path validation, coverage check,
   and CI; triage any gaps.
6. Document unsupported and unverified mathematical properties explicitly.

Do not advance to Phase 3 or declare Phase 2 complete while these remain open.
