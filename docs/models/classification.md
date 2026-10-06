# Classification models

robotorchan の classification model は、BoTorch-compatible なlatent posteriorと、分類固有の
class probability / label predictionを分離します。現在はbinary classificationを実装し、
共通契約は将来のmulticlassを想定しています。

## 最初に使うモデル

連続入力の通常の2値分類では `BinarySingleTaskGPClassifier` を第一候補にします。

```python
import torch
from robotorchan.models.classification import BinarySingleTaskGPClassifier

train_X = torch.rand(20, 3, dtype=torch.double)
train_Y = torch.randint(0, 2, (20,), dtype=torch.double)

model = BinarySingleTaskGPClassifier(train_X, train_Y)
mll = model.make_mll()

X = torch.rand(5, 3, dtype=torch.double)
latent = model.posterior(X)
probabilities = model.predict_proba(X)
labels = model.predict_class(X)
```

`posterior(X)` はclass probabilityではなくlatent GP posteriorです。確率が必要な場合は
`predict_proba(X)` を使ってください。

## モデル選択

| 問題 | モデル / family |
| --- | --- |
| 通常の連続入力binary classification | `BinarySingleTaskGPClassifier` |
| continuous + categorical | `MixedBinarySingleTaskGPClassifier` |
| task featureを持つlong-format data | `MultiTaskBinaryGPClassifier` |
| 全taskが同じXを共有するblock design | `KroneckerMultiTaskBinaryGPClassifier` |
| 独立classifierの合成 | `ClassificationModelList` |
| sparse high-dimensional relevance | MAP-SAAS / SAAS classifier |
| PCA / PLS等で入力を縮約 | reduced-space classifiers |
| random embedding | `ALEBOBinarySingleTaskGPClassifier` |
| learned representation | `JointEncoderBinaryGPClassifier` |
| stochastic deep hierarchy | `BinarySingleTaskDeepGPClassifier` |

## Capability Matrix

| model ID | input | task | high-dimensional strategy | posterior sampling |
| --- | --- | --- | --- | --- |
| `binary.standard` | continuous | single | none | Gaussian latent |
| `binary.mixed` | mixed | single | none | Gaussian latent |
| `binary.multitask` | continuous | multitask | none | Gaussian latent |
| `binary.kronecker_multitask` | continuous | multitask | none | Gaussian latent |
| `binary.map_saas` | continuous | single | MAP-SAAS | Gaussian latent |
| `binary.saas` | continuous | single | SAAS | Gaussian latent |
| `binary.reduced` / PCA / PLS / random projection | continuous | single | reduction | Gaussian |

| `binary.alebo` | continuous | single | random embedding | Gaussian latent |
| `binary.joint_encoder` | continuous | single | neural reduction | Gaussian latent |
| `binary.deep_gp` | continuous | single | deep | stochastic latent |

> `binary.saas` is a variational GP classifier with SAAS-style inverse-lengthscale shrinkage. It is not the classification counterpart of the regression `SaasFullyBayesian*` wrappers, which use NUTS/Pyro and `InferenceType.FULLY_BAYESIAN`.

### Regression-family coverage

Classification families are added according to classification semantics rather than by mechanically mirroring every regression model.

| regression/general family | binary classification status | rationale |
| --- | --- | --- |
| standard single-task / mixed / multitask | implemented | direct latent-classification counterparts exist |
| high-dimensional SAAS / MAP-SAAS / reduction / ALEBO | implemented | classification-specific inference is explicit |
| expressive DeepGP | implemented | stochastic latent hierarchy has a binary counterpart |
| expressive spectral mixture / infinite-width BNN | not implemented | no binary wrapper has been reviewed yet |
| structured output / HOGP | not implemented | structured regression outputs are not class-probability outputs |
| hierarchical / contextual | not implemented | requires a classification-specific likelihood and prediction contract |
| multi-fidelity | planned | Phase 31: classification-native variational MF model is meaningful; regression exact GP is not reusable directly |
| robust label noise / contamination / replicates | implemented | classification-native label semantics |
| input-dependent label noise | implemented | flip rates are functions of input X |
| Student-t likelihood | intentionally unsupported | no Bernoulli residual analogue |
| uncertain-input | not implemented | requires a classification-specific uncertain-input contract |
| non-GP surrogates | not implemented | probability/posterior contracts require separate classification adapters |
| preference | separate task semantics | pairwise preference observations are not binary class labels |

The absence of these binary counterparts is intentional coverage status, not a filesystem-family mismatch. New models should enter the family matching their model semantics and expose classification-specific capabilities before registry inclusion.

すべて `observation_type=classification`, variational inference,
`supports_posterior_samples=True` です。現在のclassification ALは `q=1` かつ
single-outputのみを明示的にサポートするため、multitask classifierは互換性判定で保守的に
rejectします。これはモデル自体がAL不能という意味ではなく、task selection / scalarizationの
契約が未定義なためです。

## Robust cross-family composition

Robust classification follows composition semantics instead of a Cartesian product of model
classes. Observation-process mechanisms such as label noise and contamination stay in the
likelihood layer, so they can be composed with compatible latent models without dedicated
family wrappers. `ClassificationModelList` already composes heterogeneous robust child models.

Nonstationarity changes the latent covariance itself, so reviewed mixed and multitask variants
are explicit:

- `binary.robust.nonstationary`: continuous single-task latent Gibbs covariance.
- `binary.robust.mixed_nonstationary`: Gibbs covariance on continuous dimensions plus native
  categorical covariance.
- `binary.robust.multitask_nonstationary`: Gibbs data covariance multiplied by task covariance.

High-dimensional robust combinations are not generated mechanically here. SAAS, reduced-space,
and neural-reduction combinations require separate interaction tests and remain part of the
high-dimensional cross-combination audit.

## Robust likelihood policy

回帰のrobust likelihoodをbinary classificationへ機械的に移植しません。特に
Student-t likelihoodは連続値の残差にheavy tailを与える観測モデルであり、0/1 Bernoulli
ラベルには対応する残差分布がありません。そのためbinary classifierは
`RobustnessType.STUDENT_T`をadvertiseしません。

外れラベル・誤ラベルへのrobustnessは、観測ラベル生成過程として次のように表現します。

| 問題 | classification-native mechanism |
| --- | --- |
| global class-conditional mislabeling | `binary.robust.label_noise` |
| separate contaminant label source | `binary.robust.contaminated` |
| repeated labels at identical X | `binary.robust.replicate_labels` |
| reliability varying with X | `binary.robust.input_dependent_label_noise` |

これは「robust classification likelihoodが存在しない」という意味ではありません。
Bernoulli probabilityへlabel-flip / contamination channelを合成すること自体が分類固有の
robust likelihoodです。clean latent posteriorとobserved-label probabilityは分離して扱います。

Student-t相当の名前だけを追加する薄いwrapperは作らず、新しいrobust mechanismを追加する場合は
Bernoulli/Categorical観測過程として意味が定義できることを要件とします。

## 入出力契約

binary labelは0/1だけを受け付けます。文字列classや任意整数を暗黙にencodeしません。
callerが渡したtraining dataはraw training-data contractに従って保持されます。

予測APIは次の責務を持ちます。

- `posterior` / `latent_posterior`: latent function uncertainty
- `predictive_distribution`: likelihoodを通したlabel distribution
- `predict_proba`: class probability
- `predict_class`: class decision
- `sample_latent`: latent posterior samples
- `sample_class_probabilities`: probability posterior samples
- `latent_variance`: latent uncertainty
- `predictive_variance`: predictive label variance
- `predictive_entropy`: predictive label entropy

## Active Learning

分類用には `PredictiveEntropy`, `MarginUncertainty`, `ProbabilityVariance`, `BALD`,
`LatentStraddle` を提供します。現在はいずれも `q=1` です。

回帰用acquisitionは、classifierのlatent posteriorがGaussianだからという理由だけでは互換と
みなしません。registry / capability compatibilityはclassification observation semanticsを
明示的に判定します。

## 学習

標準binary GP classifierはBernoulli likelihoodを持つvariational GPです。`make_mll()` は
`VariationalELBO`を構築します。exact Gaussian GPと同じfitting contractだと仮定しないでください。

## Multiclass-ready contract

multiclass実装はまだ公開していませんが、共通classification contractはbinary固有値を持ちません。

- `ClassificationMetadata.num_classes` は任意の2以上のクラス数を表現可能
- `class_labels` はprobability最終次元の順序を定義
- `CATEGORICAL` likelihoodと `PER_CLASS` latent structureを表現可能
- canonical multiclass tensor labelは `0..C-1` の整数class indexとして検証可能
- 0.5 threshold、Bernoulli likelihood、latent zero boundaryはbinary実装へ限定
- PredictiveEntropy / MarginUncertainty / ProbabilityVariance / BALDはclass probability
  vectorを扱うため、モデル側契約が実装されればmulticlassへ拡張可能
- LatentStraddleはlatent zero boundaryに依存するためbinary-onlyのまま

## 制約と今後の拡張

現在の主要な境界は次です。

- multiclass classifierは未実装
- classification ALはq=1
- regression familyを機械的に1対1移植しない
- Gaussian observation-noise modelをclassification noise modelとして再利用しない

multiclass追加時も `predict_proba` とprobability samplingの契約を維持し、Bernoulli固有処理を
binary implementationへ閉じ込めます。

詳細理論は [Gaussian Process Classification](../theory/24_classification.md) と
[Classification Active Learning](../theory/acquisition/classification-active-learning.md) を参照してください。


## Robust probability transforms

Scenario-wise class probabilities can be aggregated with
`RobustProbabilityTransform` using expectation, worst-case, lower-tail quantile/VaR,
or CVaR semantics. The scenario axis is explicit and the final class axis is never
reduced accidentally.

These outputs are robust class-probability utilities, not a new predictive probability
distribution. In particular, classwise worst-case, VaR, or CVaR values need not sum to
one because different classes may attain their lower tails in different scenarios.
Use `robust_class_probability` when a downstream feasibility or constraint calculation
needs one selected class probability. Phase 16 composes this utility with constrained BO.


## Classifier-backed feasibility for constrained BO

A probabilistic classifier can be composed with an objective acquisition without turning
the classifier into an outcome channel of the objective model.
`ClassificationProbabilityOfFeasibility` exposes the selected feasible-class probability,
and `RobustClassificationProbabilityOfFeasibility` first evaluates explicit input
scenarios and then applies the robust probability transform from Phase 15.

`FeasibilityWeightedAcquisition` multiplies an existing BoTorch-compatible objective
acquisition by classifier feasibility. For q-batches, the default `product` reduction is
an independence approximation for the event that all q candidates are feasible. The
optional `minimum` policy is an explicit conservative score; neither policy invents a
joint classifier posterior.

This is an outcome-feasibility decision layer. It is separate from candidate constraints
such as bounds, linear constraints, and nonlinear functions of X. When objective and
constraint outputs already share one BoTorch posterior, native BoTorch MC constraint
callables remain the preferred API.


## Non-GP binary classifiers

The binary classification family includes scikit-learn adapters for Random Forest,
Extra Trees, Gradient Boosting, and Histogram Gradient Boosting. These models implement
the classification prediction contract without pretending to own a GP latent posterior.

A single fitted estimator exposes `predict_proba`, class prediction, predictive entropy,
and Bernoulli predictive variance. It does not expose latent posterior samples, epistemic
probability variance, expected member entropy, or BALD mutual information. Returning zero
for those quantities would incorrectly imply certainty. Phase 18 adds bootstrap/member
sampling where ensemble disagreement has an explicit statistical meaning.


### Bootstrap ensemble classification

`BootstrapGradientBoostingBinaryClassifier` fits complete classifiers on independent
bootstrap resamples. The ensemble mean is the predictive class probability, while the
distribution across complete fitted members is an empirical epistemic probability posterior.

This distinction is intentional: boosting stages are not treated as posterior members.
`sample_class_probabilities`, `probability_variance`,
`expected_class_entropy`, and `mutual_information` operate on complete bootstrap members.

The registry advertises this family as `non_gp=True`,
`ensemble_posterior=True`, and `posterior_sampling_type="ensemble"`.
A latent Gaussian posterior is still not fabricated. Phase 19 generalizes the ensemble
posterior contract beyond this bootstrap implementation.


### Classification ensemble posterior

`ClassificationEnsemblePosterior` is the shared empirical posterior contract for class
probability vectors. It stores complete member predictions as
`members x ... x classes` and exposes the ensemble mean, epistemic probability variance,
predictive entropy, expected member entropy, BALD mutual information, and empirical member
sampling.

This is intentionally distinct from BoTorch's regression-oriented `EnsemblePosterior`.
Classification member probabilities are observation-space probability vectors, not latent
Gaussian function values. Bootstrap classifiers now route their uncertainty methods through
this common posterior. The contract is backend-neutral so GP and heterogeneous ensembles can
reuse it in later phases.


### GP classification ensembles

`GPBinaryClassificationEnsemble` composes independently fitted
`BinarySingleTaskGPClassifier` members that represent the same binary task and share the
same caller-supplied training data. Complete member predictive probabilities form a
`ClassificationEnsemblePosterior`.

Between-member variance and BALD-style mutual information therefore represent ensemble model
disagreement. They are deliberately distinct from the latent posterior uncertainty inside
each GP member. The ensemble exposes `member_latent_posteriors()` rather than inventing one
collapsed latent Gaussian posterior, and `make_mlls()` preserves independent variational
training objectives.

This ensemble is different from `ClassificationModelList`: a model list composes separate
classification outputs or tasks, while a GP classification ensemble combines multiple models
of the same classification target.


### Heterogeneous classification ensembles

`HeterogeneousBinaryClassificationEnsemble` combines complete binary classifiers from
different backend families, for example a variational GP classifier and a fitted tree
classifier. Members need only share binary class semantics and expose `predict_proba(X)`.

The integration boundary is the observation-space class probability. A heterogeneous ensemble
does not fabricate a common latent function, common likelihood, or common training objective.
Its `ClassificationEnsemblePosterior` represents between-model disagreement across complete
predictive probability vectors.

Phase 22 extends this composition with non-negative member weights. The weights are normalized
to one and are used consistently for predictive probabilities, between-model variance, expected
member entropy, BALD disagreement, and empirical member sampling. Equal weights remain the
default.

When the supplied weights represent posterior model probabilities, the same mechanism implements
discrete Bayesian model averaging in probability space. robotorchan does not infer those model
probabilities automatically: evidence-based, validation-based, stacking, or externally supplied
weight estimation remains a separate policy from the ensemble posterior contract.


## Post-hoc probability calibration

Classification probability calibration is separated from surrogate training. The initial calibration
API provides `TemperatureScalingCalibrator`, which applies a positive scalar temperature in
log-probability space and can be fitted on held-out validation probabilities and class labels.

`CalibratedBinaryClassifier` composes a classifier and calibrator without changing the underlying
latent posterior. Predictive probabilities and sampled class probabilities are both calibrated, so
probability variance, predictive entropy, expected class entropy, and BALD remain internally
consistent.

Temperature scaling is multiclass-ready at the probability-transform level. The current model
adapter is binary-first. Calibration data should be held out from model fitting; fitting a
calibrator on the same observations used to train the classifier does not provide an honest
out-of-sample calibration assessment.


### Calibration metrics

Calibration should be evaluated on held-out observations with more than one diagnostic. The
classification API provides:

- `classification_nll`: mean negative log likelihood, a proper scoring rule.
- `brier_score`: multiclass Brier score, also sensitive to probability quality.
- `expected_calibration_error`: observation-weighted confidence ECE over fixed bins.
- `maximum_calibration_error`: largest confidence/accuracy gap over non-empty bins.

ECE and MCE depend on the chosen bin count and use the maximum predicted class probability as
confidence. They should therefore be reported with `n_bins` and interpreted alongside NLL and
Brier score rather than as standalone proof of calibration. These metrics are evaluation
diagnostics; Phase 25 does not silently use them as training losses or acquisition functions.


### Imbalanced and cost-sensitive classification

Class imbalance and asymmetric decision cost are related but distinct concerns. Robotorchan keeps
them separate from the predictive probability model.

For a calibrated binary probability `p = P(y=1 | x, D)`, false-positive cost `C_FP`, and
false-negative cost `C_FN`, the minimum-expected-cost rule predicts class 1 when

`p >= C_FP / (C_FP + C_FN)`.

`binary_cost_sensitive_threshold`, `binary_expected_decision_cost`, and
`binary_cost_sensitive_prediction` implement this Bayes decision layer without modifying the
underlying posterior probabilities. This is important when the same probabilities are also used by
PoF, calibration diagnostics, or active-learning acquisitions.

`inverse_frequency_class_weights` provides a transparent binary imbalance diagnostic/weight
utility. It does not automatically alter GP training. In particular, robotorchan does not wrap
GPyTorch's standard variational ELBO with an ad-hoc class-weighted objective: doing so would change
the probabilistic model and its calibration semantics. Backends with native sample/class weighting
can consume explicit weights in backend-specific training APIs when that behavior is implemented
and documented.

For BO constraints, asymmetric operational costs should not be confused with candidate feasibility
probability. A calibrated `P(feasible)` remains a probability; cost-sensitive decisions belong to
the downstream decision policy.


### Reliability and out-of-distribution diagnostics

Reliability is not represented by one universal score. Robotorchan keeps three concepts separate:

- predictive entropy measures ambiguity in the predictive class distribution;
- mutual information measures epistemic disagreement when the model exposes that posterior quantity;
- input OOD score measures distance from a reference input distribution.

`ClassificationReliabilityEvaluator` provides these signals without changing the classifier posterior.
Its input-space OOD diagnostic uses a regularized Mahalanobis distance estimated from explicit
`reference_X`, or from `model.raw_train_X` when available. It is a diagnostic for covariate
departure, not a calibrated probability that a point is OOD.

`max_probability_reliability` is a simple multiclass-ready confidence diagnostic. Low maximum
probability can indicate ambiguity, but robotorchan deliberately does not label it as an OOD
detector. A classifier can be confidently wrong far outside its training distribution.

These diagnostics remain separate from calibration. Calibration evaluates whether predicted
probabilities match observed frequencies; OOD diagnostics evaluate whether a candidate is unlike
the reference domain or has high epistemic uncertainty.


### Conformal prediction sets

`SplitConformalClassifier` adds finite-sample prediction sets on top of any classifier exposing
`predict_proba`. It uses the inverse-probability nonconformity score
`1 - P(y_observed | x, D)` on a held-out calibration set and the split-conformal finite-sample
quantile `ceil((n + 1) * (1 - alpha))`.

The wrapper does not recalibrate or replace predictive probabilities. Instead, `prediction_set(X)`
returns a boolean mask over classes, and `prediction_set_size(X)` reports set size. The score and
set construction are multiclass-ready even though the current model program remains binary-first.

The usual marginal coverage statement requires exchangeability between the conformal calibration
examples and future observations. It does not imply conditional coverage for every input, subgroup,
or OOD region. Distribution shift can invalidate the guarantee, so conformal sets complement rather
than replace the reliability/OOD diagnostics from Phase 28.

Conformal calibration data must be held out from model fitting. If probability calibration such as
temperature scaling is also used, fit that transformation without leaking the conformal calibration
labels, then conformalize the final probability model on a separate held-out conformal split.


### Structured-family classification audit

The regression `models.structured` package mixes two fundamentally different ideas, so
classification does not mirror the directory mechanically.

| Regression family | Classification decision | Reason |
|---|---|---|
| OrthogonalAdditiveGP | DEFER | additive latent structure is meaningful, but requires a Bernoulli variational model rather than the exact Gaussian regression class |
| SACGP / LCEAGP | DEFER | contextual decomposition is meaningful on a latent classifier, but needs classification-native inference |
| LCEMGP / HeterogeneousMTGP | DEFER | task/context covariance can be reused only through the multitask classification posterior contract |
| HierarchicalConditionalKernelGP | DEFER | hierarchical input covariance is meaningful, but should be implemented as a variational classifier preserving structural parent dimensions |
| HigherOrderGP | UNSUPPORTED | tensor-valued Gaussian responses are not class-probability outputs |
| LatentKroneckerGP | UNSUPPORTED | a Gaussian response over an output-coordinate axis is not a classification-label posterior |

`classification_structured_audit()` exposes these decisions as runtime-readable metadata. Phase 30
does not add thin classifier wrappers around exact regression classes merely for family-name parity.

The deferred families are not rejected concepts. They are implementation candidates only when their
input/task covariance can be attached to the existing classification-native variational posterior,
sampling, calibration, active-learning, and optimization contracts. HigherOrderGP and
LatentKroneckerGP are different: their current output semantics themselves do not correspond to the
binary/multiclass classification contract.


### Multi-Fidelity classification audit

Multi-Fidelity has a meaningful classification counterpart because fidelity is an input-side
structural coordinate, not a Gaussian output type. The target model is therefore a
classification-native latent GP with Bernoulli likelihood and fidelity-aware covariance; it is not
an inheritance wrapper around regression `SingleTaskMultiFidelityGP`.

| Regression capability | Classification decision | Reason |
|---|---|---|
| SingleTaskMultiFidelityGP model role | IMPLEMENT | fidelity-aware latent classification is well-defined, but requires variational Bernoulli inference |
| MixedSingleTaskMultiFidelityGP | DEFER | compose categorical design covariance only after the base classification MF contract exists |
| MAP-SAAS MultiFidelity | DEFER | preserve fidelity dimensions and apply shrinkage only to design dimensions after the base MF classifier |
| PCA / PLS / RandomProjection MultiFidelity | DEFER | reduce only design dimensions; never absorb fidelity coordinates into the reducer |
| target-fidelity projection / fixed fidelity features | COMPOSE | public candidate-X semantics are model-independent and can be reused |
| fidelity-aware candidate constraints / TuRBO geometry | COMPOSE | optimizer-side fidelity coordinates remain structural and independent of outcome likelihood |
| AffineFidelityCostModel / cost utility | COMPOSE | evaluation cost is an input/fidelity property, not a regression-only posterior property |
| qMultiFidelityKnowledgeGradient workflow | DEFER | current qMFKG integration relies on Gaussian posterior, fantasy, and value semantics not yet established for classification |

The future base classifier must keep fidelity dimensions explicit in raw candidate space, exclude
them from ordinary design reduction and physical input perturbation, and expose target-fidelity
probabilities through the normal classification probability contract. Classification uncertainty,
calibration, reliability, conformal prediction, and active-learning utilities should consume those
probabilities without treating fidelity as a class/task dimension.

Cost-aware acquisition is a separate concern from the surrogate. Existing optimizer-side projection,
fixed-feature, constraint, and cost-model utilities can be composed where their contracts are
posterior-independent. This audit does not claim that regression qMFKG becomes valid merely because
the classifier accepts a fidelity coordinate.


### High-dimensional classification cross-family audit

High-dimensional classification already has dedicated MAP-SAAS / SAAS, reduced-space, ALEBO, and
joint-encoder model families. Phase 32 therefore treats cross-family support as a composition problem
by default rather than generating a named class for every Cartesian product.

| Cross-family combination | Decision | Contract |
|---|---|---|
| high-dimensional + calibration | COMPOSE | wrap final `predict_proba`; representation and latent GP remain unchanged |
| high-dimensional + conformal | COMPOSE | conformalize held-out probabilities; no high-dimensional-specific conformal class |
| high-dimensional + ensemble | COMPOSE | ensemble member probabilities through the common probability posterior |
| high-dimensional + reliability/OOD | COMPOSE | predictive diagnostics compose; input-distance OOD must use a meaningful representation/reference space |
| high-dimensional + candidate-time continuous input uncertainty | COMPOSE | marginalize candidate probabilities after raw-to-model input handling |
| high-dimensional + label contamination / flip likelihood | COMPOSE where likelihood-only | likelihood semantics can be attached without creating SAAS/ALEBO-specific public names |
| high-dimensional + nonstationary covariance | DEFER | both mechanisms alter covariance geometry and need an explicit combined kernel contract |
| high-dimensional + Multi-Fidelity | DEFER | fidelity dimensions must remain structural and excluded from reduction/embedding/shrinkage where appropriate |
| high-dimensional + mixed categorical structure | DEFER unless already explicit | categorical coordinates must not be silently reduced or embedded as continuous design dimensions |
| high-dimensional + DeepGP | DEFER | combines representation/hierarchy and inference changes; not a thin-wrapper composition |

The public API should not grow classes such as `CalibratedALEBO...`,
`ConformalMapSaas...`, or `EnsemblePCA...`. Those operations consume the existing classifier
probability contract and should remain wrappers/composition utilities.

For reduced-space and random-embedding models, candidate uncertainty must be defined in raw input
space first unless the user explicitly supplies a latent-space uncertainty model. Likewise, raw
Mahalanobis OOD scores are not automatically meaningful after aggressive reduction; a reference
representation must be chosen deliberately.

For future Multi-Fidelity combinations, fidelity coordinates are protected structural dimensions:
PCA/PLS/random projection/neural encoders must not absorb them, ALEBO embeddings must act on design
coordinates only, and SAAS shrinkage must be scoped to design covariance. This mirrors the
regression-side structural-dimension rule without claiming those classification combinations are
implemented today.


### Posterior and sampling final contract

Classification keeps three uncertainty surfaces distinct. `posterior` / `latent_posterior` is a
BoTorch latent-function posterior only for models that genuinely have one. `predict_proba` is the
posterior-predictive class probability after the likelihood, corruption channel, or calibration
layer. `sample_class_probabilities` represents epistemic probability samples and is the sampling
surface used by probability variance, expected class entropy, and BALD-style mutual information.

Deterministic non-GP classifiers do not fabricate a latent BoTorch posterior. Empirical ensembles
instead expose `ClassificationEnsemblePosterior`, which is explicitly a posterior over member
probability vectors and is not advertised as a BoTorch latent posterior. GP and heterogeneous
ensembles likewise do not collapse independent latent functions into a fictitious single latent GP.

Post-hoc calibration leaves the underlying latent posterior unchanged but transforms both predictive
probabilities and every epistemic probability sample. BALD-style mutual information must compute
predictive and expected conditional entropy from the same probability sample set so Monte Carlo
draw mismatch is not mistaken for epistemic disagreement.

Observation-space predictive variance remains distinct from epistemic probability variance:
Bernoulli/categorical variance describes label randomness conditional on the predictive
distribution, while `probability_variance` measures variation across posterior probability samples.


### Batch, asynchronous, and fantasization contract

Classification keeps candidate batch, pending-evaluation context, and fantasy-model state as
separate concepts.

| Capability | Current classification status | Contract |
|---|---|---|
| batched tensor evaluation | SUPPORTED | classifiers may evaluate multiple points in one tensor |
| joint q-batch AL utility | UNSUPPORTED | current PredictiveEntropy, MarginUncertainty, ProbabilityVariance, BALD, and LatentStraddle are pointwise and require q=1 |
| optimizer-side q > 1 from pointwise AL | UNSUPPORTED as joint utility | repeated/greedy selection needs an explicit diversity or conditioning policy |
| X_pending on current classification AL | UNSUPPORTED | pending points are not silently ignored or converted into a penalty |
| asynchronous classification AL | DEFER | requires a defined pending-point policy, not only storage of unresolved X |
| variational GP classification fantasize | DEFER | ExactGP fantasy semantics are not claimed for Bernoulli variational inference |
| deterministic non-GP fantasize | UNSUPPORTED | no latent posterior exists from which to construct fantasy observations |
| empirical ensemble fantasize | UNSUPPORTED | member disagreement is not a substitute for a conditioned fantasy model |

The q=1 restriction is semantic rather than a tensor-shape limitation. Pointwise entropy or BALD
evaluated on a tensor of q candidates does not become a joint batch acquisition merely by summing
or averaging scores. A future batch classifier acquisition must define redundancy/diversity or
joint information gain explicitly.

Likewise, unresolved `X_pending` belongs to the acquisition policy. Current classification
acquisitions do not expose `X_pending`, so they must not advertise asynchronous support. A future
async policy may penalize proximity, condition on pseudo/fantasy labels, or use a joint
information-theoretic objective, but that choice must be explicit.

Classification GP models use variational Bernoulli inference. They therefore do not inherit the
exact-Gaussian assumption that a posterior draw can always be appended through the standard
ExactGP fantasy lifecycle. Any future `fantasize` implementation must define how variational
parameters, inducing points, likelihood observations, and fantasy batch dimensions are updated.

Candidate q, posterior Monte Carlo sample dimensions, ensemble-member dimensions, uncertain-input
scenario dimensions, and any future fantasy dimensions remain independent axes.


### Multiclass readiness

The common classification contract is class-count agnostic: class probabilities use the final tensor
dimension, `class_labels` defines its ordering, and probability samples retain that class axis.
`ClassificationEnsemblePosterior`, probability calibration, predictive entropy, margin
uncertainty, probability variance, and BALD-style mutual information therefore do not require a
binary-only API redesign.

Binary-specific semantics remain isolated under `classification/binary`: Bernoulli likelihood,
the scalar latent decision boundary, threshold-based `predict_class`, label flip false-positive /
false-negative channels, and `LatentStraddle`. A future multiclass family must use categorical /
softmax likelihood semantics and a per-class or otherwise explicit latent-output structure rather
than inheriting `BinaryClassificationMixin`.

The public registry is assembled from task-specific fragments. Adding
`MULTICLASS_CLASSIFICATION_MODEL_SPECS` should therefore extend the registry without rewriting the
binary fragment or changing existing stable IDs. Multiclass model entries may use a dynamic
`num_classes=None` registry value when class count is constructor/data dependent; the instantiated
model remains responsible for concrete `num_classes` and `class_labels`.

Multiclass readiness does not imply that a multiclass surrogate is implemented. In particular,
multiclass calibration, class-conditional robustness, cost-sensitive decisions, latent-boundary
acquisitions, and structured/multitask latent shapes require their own semantic review.
