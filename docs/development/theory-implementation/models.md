# Model theory–implementation mapping

## Audit provenance and limits

- Initial inventory baseline: `main` at `a4454c59232f6eb209f4c8cd302d87c7f2233032`.
  Subsequent source/test inspections are append-only audit evidence from later
  `main` revisions; rerun registry scripts against the revision under review.
- Sources: `MODEL_REGISTRY`, `CLASSIFICATION_MODEL_REGISTRY`,
  `docs/model_coverage.json`, public exports, and inspected implementation modules.
- This is a static inventory plus selected source and assertion inspections,
  **not** comprehensive numerical or mathematical verification. Static MRO,
  imported calls, syntactic test references and inspected assertions are
  distinct evidence levels; see the registry-wide evidence-summary command.
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

These are stable registry **IDs**, not public class names. The status below
refers to representative source/test inspection recorded in later sections;
**it is not per-ID exhaustive runtime validation**. Static class/module
resolution and syntactic test/upstream-call detection are available for all
29 IDs using `scripts/audit_model_api.py`.

| Registry ID | Audit status |
| --- | --- |
| `binary.ensemble.gp` | GP ensemble source and named assertions inspected |
| `binary.ensemble.heterogeneous` | Heterogeneous ensemble source and named assertions inspected |
| `binary.standard` | Variational Bernoulli baseline source and named assertions inspected |
| `binary.mixed` | Mixed classifier source and named assertions inspected |
| `binary.multitask` | Multitask variational source and named assertions inspected |
| `binary.kronecker_multitask` | Long-format variational implementation and assertions inspected; not exact Kronecker |
| `binary.map_saas` | MAP-SAAS kernel/prior and named assertions inspected |
| `binary.saas` | Shrinkage-prior variational model and named assertions inspected; not NUTS |
| `binary.reduced` | Reduced model source and representative assertions inspected |
| `binary.pca` | PCA reduction and parametrized assertions inspected |
| `binary.pls` | PLS reduction and parametrized assertions inspected |
| `binary.random_projection` | Random projection and parametrized assertions inspected |
| `binary.alebo` | ALEBO metric and named assertions inspected |
| `binary.joint_encoder` | Neural encoder and named assertions inspected |
| `binary.deep_gp` | Deep variational model and named assertions inspected |
| `binary.uncertain.categorical_input` | Finite-category marginalization and named assertions inspected |
| `binary.uncertain.continuous_input` | Gaussian input perturbation and named assertions inspected |
| `binary.robust.label_noise` | Flip-channel likelihood and named assertions inspected |
| `binary.robust.contaminated` | Contamination mixture and named assertions inspected |
| `binary.robust.replicate_labels` | Replicate preservation and named assertions inspected |
| `binary.robust.input_dependent_label_noise` | Input-dependent flip channel and named assertions inspected |
| `binary.robust.nonstationary` | Gibbs nonstationary kernel and named assertions inspected |
| `binary.robust.mixed_nonstationary` | Mixed nonstationary source and structural assertions inspected |
| `binary.robust.multitask_nonstationary` | Multitask nonstationary source and structural assertions inspected |
| `binary.non_gp.bootstrap_gradient_boosting` | Bootstrap sklearn ensemble and named assertions inspected |
| `binary.non_gp.random_forest` | sklearn adapter and named assertions inspected |
| `binary.non_gp.extra_trees` | sklearn adapter and named assertions inspected |
| `binary.non_gp.gradient_boosting` | sklearn adapter and named assertions inspected |
| `binary.non_gp.hist_gradient_boosting` | sklearn adapter and named assertions inspected |

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

## Source and assertion audit: neural, ensemble, robust and categorical models

Five further implementation and corresponding test modules were inspected
on `main`. Together they contain **27 named test functions**: joint neural
encoder (4), homogeneous GP ensemble (6), input-dependent label noise (6),
nonstationary binary GP (5), and uncertain categorical input (6).

| Model and test module | Implemented / delegated numerical object | Specific assertions observed | Remaining verification |
| --- | --- | --- | --- |
| `high_dimensional/reduced/joint_neural.py` / `high_dimensional/reduced/test_joint_encoder_classifier.py` | Local neural encoder maps standardized original inputs to GP features; `training_loss` evaluates the binary variational ELBO through the encoder | Class probability shape and simplex, latent feature dimension, finite gradients for all encoder parameters, raw-data preservation, incorrect input dimension rejection | End-to-end trained representation quality, encoder/GP optimization stability and extrapolation |
| `ensemble/gp.py` / `ensemble/test_gp.py` | Empirical class-probability posterior over separate binary GP classifiers, via local `ClassificationEnsemblePosterior` adapter | Mean and population variance against stacked member predictions, separate member latent posteriors, probability sample simplex and nonnegative MI, matching-training-data guard, registry/acquisition compatibility | Independent MI/entropy identity, statistical calibration, robustness to correlated ensemble members |
| `robust/input_dependent_noise.py` / `robust/test_input_dependent_noise.py` | GPyTorch Bernoulli likelihood plus local input-dependent false-positive/negative flip channel; rate module or callable | Input-dependent observed probabilities, clean vs observed predictions, learnable rate parameter registration, callable rates, shape and bound validation | Quadrature accuracy, noisy-label parameter identifiability, calibrated prediction across input space |
| `robust/nonstationary.py` / `robust/test_nonstationary.py` | Local `GibbsKernel` changes latent covariance with location, while the Bernoulli observation process remains unchanged | Gibbs/ScaleKernel structure, positive input-dependent local lengthscale, predictive class simplex, latent-process (not label-noise) metadata, registry capability | Positive-definite kernel numerical tests across settings, spatial adaptation and training quality; mixed/multitask variants not covered by this module |
| `uncertain/uncertain_categorical.py` / `uncertain/test_uncertain_categorical.py` | Analytic finite-support marginalization: a weighted sum of mixed-GP class probabilities over candidate categorical values | One-hot category identity, soft-weighted sum against explicit alternatives, finite gradients in continuous candidate coordinates, categorical-dimension rejection and metadata | Larger category supports, multi-batch edge cases, invalid probability distributions and runtime performance |

These assertions include meaningful identities for ensemble moments and
finite-support categorical marginalization. They are not equivalent to
statistical calibration or convergence tests. Notably, joint neural encoder
training checks that gradients reach encoder parameters, but does not verify
that a fitted encoder learns a useful representation. The nonstationary test
covers the single-task classifier only; it must not be generalized to the
mixed and multitask nonstationary implementations.

## Source and assertion audit: contamination, replicates, bootstrap and robust composition

Four additional implementation/test paths were inspected on `main`, together
with the mixed/multitask nonstationary implementations in
`classification/binary/robust/nonstationary.py`. Their four inspected test
modules contain **18 named test functions**: contamination (5), repeated
labels (4), bootstrap gradient boosting (4), and robust cross-family
composition (5). Parametrized tests cover additional execution cases.

| Implementation and test module | Local algorithm and delegation | Asserted checks | Not verified |
| --- | --- | --- | --- |
| `robust/contaminated.py` / `robust/test_contaminated.py` | A GPyTorch Bernoulli likelihood is mixed with an independent contaminant source using `(1-c)*p_clean+c*p_contaminant`; the classifier retains the clean latent GP | Explicit three-point mixture values; clean-to-observed relation; learnable contamination gradient; probability bounds and finite-value validation | Quadrature error, identifiability of contamination and clean signal, fitted probability calibration |
| `robust/replicate.py` / `robust/test_replicate.py` | Keeps every repeated observation for inherited variational Bernoulli inference; locally computes counts, positive rates and disagreement `2*p*(1-p)` without replacing training labels with aggregated rates | Six raw observations retained, per-condition counts and rates, disagreement at mixed vs unanimous replicates, minimum two replicates guard, column-label support | Effect of replication on predictive uncertainty, large replicate groups, and consistency of posterior learning |
| `non_gp/gradient_boosting.py` and `non_gp/bootstrap.py` / `non_gp/test_bootstrap.py` | Constructs individual sklearn gradient-boosting estimators over bootstrap samples; reports empirical member probability moments | Fit/sample probability shapes and class simplex; mean and population variance against member predictions; nonnegative mutual information; minimum member count; registry ensemble sampling flags | Bootstrap uncertainty calibration, independent numerical MI reference, sensitivity to resampling and rare classes |
| `robust/nonstationary.py` / `robust/test_cross_family.py` | Mixed model puts Gibbs kernels on continuous features and separate categorical kernels; multitask model excludes task identity from data-lengthscale input | Continuous-only local-lengthscale shapes in mixed and task-aware models, heterogeneous robust `ClassificationModelList` composition, capability flags, standard nonstationary compatibility | Numerical mixed/task kernel identities, positive definiteness across wider parameter ranges, cross-task recovery and training quality |

The contamination mixture formula is verified against fixed expected values
for `c=0.2`, `p_contaminant=0.75` and `p_clean=(0,0.25,1)`. In the
replicate model, `2*p*(1-p)` is a diagnostic of observed-label disagreement,
not a replacement likelihood or a GP posterior variance. Bootstrap member
variance describes variability across resampled estimators; it is not by
itself a guarantee of calibrated epistemic uncertainty. Robust cross-family
tests establish structural shapes and composition contracts, not covariance
matrix numerical equivalence.

## Repeatable registry-wide static evidence summary

Run the following command to produce one temporary JSON review report for
all regression and classification registry entries:

```bash
python scripts/audit_model_api.py --include-evidence-summary --output model-evidence.json
```

The `evidence_summary.families` result reports registry counts, identifiers
without syntactic test references, and identifiers without direct upstream
constructor calls, separately for regression and classification. It includes
the complete `test_references` and `upstream_calls` evidence for follow-up
review. Both are **static detection only**: absent matches do not prove missing
tests or delegation, and detected matches do not establish assertion execution,
mathematical correctness, or exercised runtime pathways. Do not commit the
point-in-time JSON snapshot; rerun it on the relevant `main` revision.

## Inspected regression non-GP implementations: NGBoost and ExtraTrees

Two additional non-GP regression implementations and their test modules were
inspected on `main`, covering **8 named tests** (NGBoost: 3; ExtraTrees: 5).
This evidence extends the classification-focused review to regression models.

| Model and inspected tests | Implementation and numerical delegation | Assertions inspected | Verification limits |
| --- | --- | --- | --- |
| `NGBoostSurrogate` in `models/non_gp/ngboost.py`, `tests/models/non_gp/test_ngboost_surrogate.py` | Composes optional `ngboost.NGBRegressor(Dist=Normal)`, converts NGBoost's `loc` and `scale` to a local `GaussianDistributionPosterior` with variance `scale**2`; no MLL contract | Posterior mean/variance and sampling shapes, finite nonnegative variance, raw data, MC qUCB and tree-ensemble strategy integration, posterior standard-deviation squared equals variance | NGBoost-specific tests are skipped when optional `ngboost` is unavailable; numerical agreement with the source distribution and predictive calibration are not established |
| `ExtraTreesSurrogate` in `models/non_gp/extra_trees.py`, `tests/models/non_gp/test_extra_trees_surrogate.py` | Composes sklearn `ExtraTreesRegressor`, gathers per-tree predictions into BoTorch `EnsemblePosterior` through local `make_ensemble_posterior` | Raw-data/fit and unsupported MLL contracts, ensemble values and batch shapes, finite MC qEI, fit-before-posterior and single-output guards | Per-tree prediction equality to independent sklearn results, uncertainty calibration, and statistical performance remain unverified |

**Distinct uncertainty semantics:** NGBoost exports the variance of its
fitted parametric Gaussian predictive distribution, whereas ExtraTrees
returns an empirical distribution of tree predictions. Neither should be
silently interpreted as the posterior of an exact Gaussian process. Test
presence in a file does not guarantee CI executed the optional NGBoost
cases: the NGBoost tests use dependency-dependent skip markers. The static
model test-reference scanner also does not track these runtime skips.

## Inspected regression noise and heterogeneous multitask implementations

Three regression model implementations and their corresponding test modules
were inspected on `main`. The modules contain **13 named test functions**
(seven in `test_heteroskedastic_gp.py`, four in
`test_joint_heteroskedastic_gp.py`, and two in
`test_mixed_heterogeneous_mtgp.py`). The first test module also exercises
the `HeteroskedasticKroneckerMultiTaskGP` prototype, so counts do not imply
seven independent tests of the single-task model.

| Model / source and tests | Numerical algorithm and delegation | Specific assertion evidence | Verification gaps |
| --- | --- | --- | --- |
| `HeteroskedasticSingleTaskGP` / `models/robust/robust.py`, `tests/models/robust/test_heteroskedastic_gp.py` | Local alternating response-GP and log-residual-variance-GP fitting via `botorch.fit.fit_gpytorch_mll`; exponentiates predicted log noise and applies a floor | Preserved raw inputs/labels, one-iteration fit returns self and finite floored noise, pre-fit noise-posterior guard, positive iteration guard | Recovery of a known heteroskedastic noise function, convergence across alternating iterations, calibration of observation variance |
| `HeteroskedasticKroneckerMultiTaskGP` prototype / same source and test module | Block-design multitask response covariance with separate log-noise GP; source labels public observation-posterior behavior as intentionally unsupported | Raw task-specific variance shape, distinct task-specific noise and flattened covariance diagonal, rejection of broadcast single-column variance | Full observation-posterior consistency and numerical Kronecker-noise propagation; the source explicitly describes this as a prototype |
| `JointHeteroskedasticSingleTaskGP` / `models/robust/joint_heteroskedastic.py`, `tests/models/robust/test_joint_heteroskedastic_gp.py` | Two `SingleTaskVariationalGP` latent processes; local Monte Carlo expected Gaussian log likelihood plus response/noise KL terms; `E[exp(log_noise)]` from Gaussian moments | Finite differentiable scalar loss with gradients reaching both latent processes, candidate-batch posterior/noise shapes, qUCB interoperability, state-dict prediction round trip | ELBO numerical reference, MC convergence, true noise recovery, trained posterior calibration |
| `MixedHeterogeneousMTGP` / `models/structured/heterogeneous.py`, `tests/models/structured/test_mixed_heterogeneous_mtgp.py` | Delegates heterogeneous task/feature-subset structure to BoTorch's `HeterogeneousMTGP`; locally replaces categorical-containing conditional covariance subsets with native mixed kernels | Presence of categorical and additive kernels, raw per-task inputs, exact MLL type, finite posterior mean/variance and shapes | Numerical task cross-covariances, feature-subset invariance, SAAS-enabled configuration and fitted predictive quality |

The iterative noise-GP and jointly optimized variational two-latent noise
model are different numerical algorithms. A finite one-iteration result does
not establish convergence of the alternating method. Likewise, the
heteroskedastic Kronecker *prototype* must not be presented as a complete
input-dependent observation-noise posterior. These distinctions are retained
for the Phase 7 mathematical checks.

## Source and assertion audit: multi-fidelity regression families

The shared implementation in `models/standard/multi_fidelity.py`, the
heteroskedastic extension in `models/robust/robust.py`, and four test modules
were inspected on `main`. These include **22 named test functions** (nine in
`test_single_task_multi_fidelity_gp.py`, four in
`test_multi_fidelity_sampling.py`, five in
`test_heteroskedastic_multifidelity.py`, and four in
`test_mixed_multi_fidelity.py`); several parametrized functions generate
more than one executed case.

| Model and tests | Delegated and local responsibilities | Assertions inspected | Unverified claims |
| --- | --- | --- | --- |
| `SingleTaskMultiFidelityGP` / `test_single_task_multi_fidelity_gp.py`, `test_multi_fidelity_sampling.py` | Direct inheritance from BoTorch's `SingleTaskMultiFidelityGP`; local raw-training-tensor retention and exact MLL mixin | Public constructor signature/default parity, raw-data and MLL contract, **posterior mean and variance agreement with upstream BoTorch after state transfer**, IID and Sobol sampler shapes and differentiability through design and fidelity inputs | Fitted multi-fidelity predictive accuracy, cost-efficiency and sampler distributional calibration |
| `MixedSingleTaskMultiFidelityGP` / `test_single_task_multi_fidelity_gp.py`, `test_mixed_multi_fidelity.py` | Local mixed design covariance excludes structural fidelity columns; BoTorch-native fidelity `DownsamplingKernel` retained; linear-truncated path deliberately rejected | Categorical/fidelity dimension separation, native kernel types, posterior and t-batch shapes, categorical-only design, continuous factory active dims, overlap/duplicate/invalid dimension checks | Numerical equivalence to independent kernel reference, cross-fidelity statistical recovery and large mixed search quality |
| `HeteroskedasticMultiFidelityGP` / `test_heteroskedastic_multifidelity.py` | Alternates response and log-noise GPs while retaining fidelity coordinates in both; delegates fitting to BoTorch `fit_gpytorch_mll` | Raw-data and posterior contract, finite qUCB, one-iteration noise-model fitting, positive predicted noise, integration with BoTorch cost-aware `qMultiFidelityKnowledgeGradient`, fidelity retained in noise model | Convergence of noise estimation across iterations, calibrated noise recovery, objective improvement versus standard multi-fidelity GP |

The direct upstream posterior equality test is stronger than structural API
checks but covers only one small fixed configuration and matching transferred
parameters. Multi-fidelity samplers are tested for shape and finite gradients,
not unbiasedness or convergence. `MixedSingleTaskMultiFidelityGP` keeps
categorical design columns distinct from fidelity identity; the locally
constructed mixed covariance is *not* a replacement for BoTorch's fidelity
kernel. Cost-aware knowledge-gradient interoperability establishes a usable
call path rather than optimization performance.

## Source and assertion audit: regression DeepGP models

The implementation in `models/expressive/deep_gp.py` and two targeted
test modules were inspected on `main`: **20 named test functions** (15 in
`tests/models/expressive/test_deep_gp.py`, five in
`tests/models/expressive/test_deep_gp_botorch.py`).

| Model family | Numerical method and delegation | Inspected assertion evidence | Verification limits |
| --- | --- | --- | --- |
| `SingleTaskDeepGP` | Hierarchy of GPyTorch `DeepGPLayer` modules with Cholesky variational distributions, `VariationalStrategy`, Gaussian likelihood and `DeepApproximateMLL(VariationalELBO)`; local `DeepGPPosterior` exposes Monte Carlo samples | Raw training data/layer structure, finite negative ELBO, hidden/output kernel gradients and actual Adam parameter update, minibatch total-data scaling contract, invalid hyperparameter guards | ELBO numerical reference, posterior approximation convergence, learned function recovery |
| `MultiTaskDeepGP` | Local task-id normalization and learned task embedding; inherited stochastic DeepGP on embedded features, **not** a BoTorch exact `MultiTaskGP` covariance | Task embedding/structure, posterior shape for raw long-format task inputs, finite nonzero embedding gradient, noncontiguous task-id rejection | Cross-task correlation accuracy and embedding identifiability |
| `MixedSingleTaskDeepGP` | Local categorical embedding followed by inherited DeepGP hierarchy; categorical labels are not treated as continuous GP coordinates | Raw mixed data and category metadata, posterior finite moments, nonzero categorical embedding gradient, unknown-category rejection | Embedding extrapolation, calibrated mixed-data uncertainty |
| BoTorch integration of `SingleTaskDeepGP` | Local sample-backed `DeepGPPosterior` supports upstream MC acquisitions and `botorch.optim.optimize_acqf` | Candidate gradient through posterior mean, `rsample` shape, finite qLogEI/qUCB, successful optimize_acqf call in original input coordinates, stable singleton-input standardization | Optimization regret, independent distributional Monte Carlo reference, multi-seed stability |

The `DeepGPPosterior` is a **sample-based approximation**, not an exact
Gaussian analytic posterior. A finite acquisition value and optimizer call
demonstrate interoperability rather than global optimality. The contract
test `tests/models/contracts/test_expressive_public_contract.py` also
explicitly forbids the legacy public alias `DeepKernelGP`; it must not be
mistaken for a registered model. Phase 7 should test Monte Carlo moment
accuracy and numerical stability separately from these contract checks.

## Source and assertion audit: expressive stationary and infinite-width kernels

The source and dedicated tests for `SpectralMixtureGP` variants
(`models/expressive/spectral_mixture.py`,
`tests/models/expressive/test_spectral_mixture.py`) and
`InfiniteWidthBNNGP` variants
(`models/expressive/infinite_width_bnn.py`,
`tests/models/expressive/test_infinite_width_bnn.py`) were inspected.
These checks cover standard, mixed, long-format multitask and block-design
Kronecker constructions, with different evidence strengths.

| Family | Mathematical object and implementation boundary | Directly inspected evidence | Remaining mathematical verification |
| --- | --- | --- | --- |
| Spectral mixture exact GPs | Local setup initializes GPyTorch `SpectralMixtureKernel` using either training data or empirical spectrum; parent BoTorch-compatible exact GP handles posterior and MLL | Finite spectral parameters, valid hyperparameter guards, preserved raw data, exact MLL, finite posterior/qLogEI and successful BoTorch optimization | Independent spectral-density reconstruction, recovery of known mixture frequencies and initialization sensitivity |
| Spectral mixed / multitask / Kronecker | Local class selects non-task/non-categorical dimensions for spectral data covariance; native categorical kernels or structured task covariance are composed with GPyTorch spectral kernel | Correct spectral active dimensions, block-design MLL and sampling shapes, category changes covariance, scalarized qLogEI, mixed `optimize_acqf_mixed` candidate feasibility | Numeric equality of composed covariance to explicit Kronecker or mixed-kernel reference, learned cross-task correlations |
| Infinite-width ReLU NNGP kernel | `InfiniteWidthReLUKernel` **locally** applies the arc-cosine ReLU covariance recursion at each depth, with ARD-scaled inputs; GPyTorch provides `Kernel` plumbing | Symmetric finite 12-by-12 Gram matrix with positive diagonal and eigenvalues >= -1e-8; invalid variance/depth guards | Independent depth-one closed-form reference, analytic diagonal recursion identity, deeper numerical stability |
| NNGP single/mixed/multitask/Kronecker | Local custom NNGP data kernel is passed to exact GP; task columns excluded from data covariance and categorical data handled through `make_mixed_covar_module` | Raw data/MLL, correct task active dimensions, finite posterior/samples/qLogEI, mixed-category changes covariance, both continuous and mixed BoTorch optimizer pathways | Numerical task/Kronecker covariance identity and objective performance on controlled functions |

These families must not be classified as wholly delegated. Spectral-mixture
kernel evaluation delegates to GPyTorch, while its input-role composition
and initialization are local; the infinite-width ReLU recursion itself is
implemented in robotorchan. In particular, Gram-matrix symmetry and a
tolerance-based PSD check are useful numerical properties, **not** a proof
that the recursion matches a published infinite-width kernel formula. That
independent algebraic reference belongs to Phase 7.

## Source and assertion audit: joint neural representation regression

The implementations in `models/high_dimensional/reduced/joint_neural.py` and
`joint_vae.py`, along with three dedicated test modules, were inspected.
There are **17 named test functions**: eleven for `JointEncoderGP`, four
for mixed joint models and two shared training-loss checks.

| Model family | Implemented mathematics and upstream boundary | Verified assertions in inspected tests | Numerical/empirical gaps |
| --- | --- | --- | --- |
| `JointEncoderGP` | Locally standardizes inputs and jointly trains a neural feature network through BoTorch `SingleTaskGP` exact marginal log likelihood; the public GP accepts original coordinates | Original training inputs retained, encoder dimensions, MLL gradient to encoder, candidate gradients through qLogEI, state-dict round trip, custom encoder contract, qLogEI/qLogNEI/qUCB, `optimize_acqf`, condition-on-observations and fantasize in original coordinates | Learned representation quality, comparison with frozen reduction, optimization stability across seeds |
| `HybridAutoEncoderGP` | Adds weighted standardized input reconstruction MSE to negative GP MLL; the decoder is local PyTorch | Shared finite scalar training loss and actual Adam parameter update; mixed variant verifies continuous-only reconstruction and decoder gradient | Exact weighted-loss decomposition and reconstruction–prediction tradeoff |
| `JointVAEGP` | Adds VAE KL divergence and stochastic reconstruction to negative GP MLL; default predictions use the posterior-mean latent code; optional `uncertainty_aware_posterior` performs Monte Carlo latent marginalization with Gaussian moment matching | Shared finite training loss; mixed variant checks mean/logvariance shapes and gradient flow into both variational heads and decoder | Independent KL and total-variance identities, stochastic latent MC convergence, calibration of uncertainty-aware posterior |
| `MixedJointEncoderGP`, `MixedHybridAutoEncoderGP`, `MixedJointVAEGP` | Reduces **only continuous coordinates** and appends unmodified categorical values; builds native mixed covariance rather than one-hot encoding categories | Categorical column survives encoding exactly, negative categorical index, original mixed posterior shape, encoder/decoder/VAE gradient propagation | Mixed-covariance numerical reference, sensitivity to rare categories and noisy categorical observations |

The joint encoder's training GP is constructed initially from detached
encoded training inputs, then reset to raw input coordinates so its
`forward` method can apply the trainable encoder. This must not be
confused with a frozen dimensionality reducer. The VAE default
`posterior(X)` uses deterministic latent means; its explicit
`uncertainty_aware_posterior` is a separate moment-matched approximation.
Existing tests of the joint VAE training loss do not by themselves establish
correctness of that optional uncertainty propagation.

## Classification documentation mapping at registry-ID granularity

Unlike regression `MODEL_REGISTRY`, the 29 entries in
`CLASSIFICATION_MODEL_REGISTRY` do **not** carry individual guide, theory,
or notebook metadata. The following is a documentation-ownership convention
derived from the implemented binary registry and existing shared references,
not an assertion that every entry has its own notebook.

| Binary registry ID subset | Shared usage guide | Shared theoretical reference | Notebook verification |
| --- | --- | --- | --- |
| `binary.standard`, `binary.mixed`, `binary.multitask`, `binary.kronecker_multitask` | `docs/models/classification.md` | `docs/theory/24_classification.md` | Per-ID execution **unverified** |
| `binary.map_saas`, `binary.saas`, `binary.reduced`, `binary.pca`, `binary.pls`, `binary.random_projection`, `binary.alebo`, `binary.joint_encoder` | `docs/models/classification.md` | `docs/theory/24_classification.md` | Per-ID execution **unverified** |
| `binary.deep_gp` | `docs/models/classification.md` | `docs/theory/24_classification.md` | Per-ID execution **unverified** |
| `binary.ensemble.gp`, `binary.ensemble.heterogeneous` | `docs/models/classification.md` | `docs/theory/24_classification.md` | Per-ID execution **unverified** |
| `binary.uncertain.categorical_input`, `binary.uncertain.continuous_input` | `docs/models/classification.md` | `docs/theory/24_classification.md` | Per-ID execution **unverified** |
| `binary.robust.label_noise`, `binary.robust.contaminated`, `binary.robust.replicate_labels`, `binary.robust.input_dependent_label_noise`, `binary.robust.nonstationary`, `binary.robust.mixed_nonstationary`, `binary.robust.multitask_nonstationary` | `docs/models/classification.md` | `docs/theory/24_classification.md` | Per-ID execution **unverified** |
| `binary.non_gp.bootstrap_gradient_boosting`, `binary.non_gp.random_forest`, `binary.non_gp.extra_trees`, `binary.non_gp.gradient_boosting`, `binary.non_gp.hist_gradient_boosting` | `docs/models/classification.md` | `docs/theory/24_classification.md` (probability/latent distinction; not a GP derivation for sklearn) | Per-ID execution **unverified** |

The shared classification guide explicitly documents family capabilities,
predictive probability versus latent posterior, robust observation-process
semantics, and non-GP limitations. The common GP classification theory
provides background, **not** a mathematical derivation for every listed
non-GP, ensemble, encoder, or robust variant. Those require family-specific
source-to-theory records in Phase 4. Neither missing registry metadata nor
shared documentation paths should be represented as broken files or as
individual executable demonstrations.

## Phase 2 inventory sign-off and scope

Phase 2 was accepted as a **registry-wide static provenance inventory**
after the model API CLI, regression documentation-path checks, the
classification shared-documentation mapping contract, and the repository
CI / Documentation links workflows passed. It does **not** certify that
every listed algorithm has been manually validated or that the mathematical
formulas have passed independent reference tests.

- 142 regression registry names and 29 classification IDs are enumerated.
- The API audit resolves concrete registered classes and implementation
  modules, static MRO, syntactic test references and direct imported calls.
  Absence of a syntactic match is not proof of absent test coverage.
- Representative source and assertion inspections and their explicit
  limitations are recorded in the Phase 2 sections above.
- Classification entries share a documented guide/theory owner without
  asserting individual notebooks or mathematically complete derivations.
- Numerical consistency and statistical behavior remain Phase 7 work,
  **not** an unclosed Phase 2 gate.

Phase 3 acquisition and optimizer static inventory and its CI-backed exit
contracts are complete on `main` at `e1ae6474dca742be2f882ddac7f61c9f77306614`.
Phase 4 develops the **method-level model theory-to-source records** below;
their verified status must not be inferred from Phase 2 inventory alone.

## Phase 4: theory-to-implementation records

The following records use the mapping contract from this directory's
`README.md`. Registry-wide theory, guide and notebook metadata continues
to live in `docs/model_coverage.json` rather than being copied here.
The provenance below is source and test inspection on `main` at
`e1ae6474dca742be2f882ddac7f61c9f77306614`.

### Exact single-task Gaussian-process regression

| Field | Verified mapping |
| --- | --- |
| Method / family | `SingleTaskGP` / exact Gaussian-process regression |
| Theory and mathematical object | `docs/theory/02_gaussian_process.md`; Gaussian likelihood, kernel-conditioned posterior mean/covariance and exact marginal log likelihood |
| Reference | BoTorch `botorch.models.SingleTaskGP` and Gaussian-process conditioning; this record does not claim an independent derivation |
| Public API | `robotorchan.models.SingleTaskGP`; `posterior`, `make_mll`, `fit`, and raw training-tensor accessors |
| Implementation | `src/robotorchan/models/standard/single_task.py`: `SingleTaskGP(FittableExactGPModelMixin, BoTorchSingleTaskGP)` |
| Ownership / upstream | **Delegated** posterior and GP behavior to `botorch.models.SingleTaskGP`; **Extended** raw training-data capture and common fit/MLL mixin in `models/base.py` |
| Numerical method | BoTorch/GPyTorch exact Gaussian process inference and `ExactMarginalLogLikelihood`; no locally implemented GP conditioning formula in this class |
| Verification | `tests/models/standard/test_single_task_gp.py`: direct upstream posterior mean/variance agreement with transferred state, duplicate-input finite posterior, exact MLL type, raw-data retention |
| Notebook | Registry-specified notebook in `docs/model_coverage.json`; executable coverage is **not** established by a path |
| Limitations | Equality holds on inspected configurations, not all kernels, batch shapes or noisy-data regimes; Phase 7 independent numerical references are pending |

### Native mixed-input Gaussian-process regression

| Field | Verified mapping |
| --- | --- |
| Method / family | `MixedSingleTaskGP` / mixed continuous and categorical exact GP |
| Theory and mathematical object | `docs/theory/02_gaussian_process.md` (GP conditioning); categorical covariance composition is owned by upstream BoTorch |
| Reference | `botorch.models.MixedSingleTaskGP`; do not equate integer-coded categories with an ordered continuous metric |
| Public API | `robotorchan.models.MixedSingleTaskGP(train_X, train_Y, cat_dims, ...)` |
| Implementation | `src/robotorchan/models/standard/single_task.py`: `MixedSingleTaskGP(FittableExactGPModelMixin, BoTorchMixedSingleTaskGP)` |
| Ownership / upstream | **Delegated** the native mixed kernel and posterior to `botorch.models.MixedSingleTaskGP`; **Extended** raw-data capture and common fit/MLL API |
| Numerical method | Exact GP inference with BoTorch mixed kernel; `cat_dims` accepts negative indices and stores the caller's categorical coordinate metadata |
| Verification | `tests/models/standard/test_mixed_single_task_gp.py`: constructor signature parity, native subclass identity, exact MLL, raw-data retention and posterior mean/variance agreement after state transfer |
| Notebook | Registry-specified notebook via generated coverage; no per-test notebook execution claim |
| Limitations | Static signature parity and a fixed numerical configuration do not establish every categorical kernel or optimizer combination; mathematical reference checks remain Phase 7 |

### Bernoulli variational Gaussian-process classification

| Field | Verified mapping |
| --- | --- |
| Method / family | `binary.standard` / `BinarySingleTaskGPClassifier` |
| Theory and mathematical object | `docs/theory/24_classification.md`; latent GP with a Bernoulli observation likelihood and variational evidence lower bound (ELBO) |
| Reference | GPyTorch `BernoulliLikelihood`, `VariationalELBO`; BoTorch `SingleTaskVariationalGP` infrastructure |
| Public API | `posterior` (latent), `predictive_distribution` (Bernoulli), `predict_proba` (class probabilities), `predict_class`, `make_mll` |
| Implementation | `src/robotorchan/models/classification/binary/standard/single_task.py` |
| Ownership / upstream | **Extended** BoTorch variational GP with local 0/1 label validation and class-probability adaptation; **Delegated** Bernoulli likelihood and ELBO to GPyTorch |
| Numerical method | Approximate variational latent posterior; likelihood integrates latent uncertainty for Bernoulli predictive probabilities; **not** exact Gaussian regression conditioning |
| Verification | `tests/acquisition/test_api_inventory.py` confirms a valid probability-based PredictiveEntropy score and `q=1` enforcement; classification model tests require further source-to-assertion mapping |
| Notebook | **Unverified** per-ID executable demo; common usage guide is `docs/models/classification.md` |
| Limitations | Probability calibration, ELBO fit convergence and variational approximation error unverified; Phase 7 numerical checks pending |

Further model families remain to be mapped at the same detail in Phase 4.
