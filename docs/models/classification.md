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
| multi-fidelity | not implemented | fidelity-aware binary classification has not been reviewed |
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
