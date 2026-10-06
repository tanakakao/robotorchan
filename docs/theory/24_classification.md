# Gaussian Process Classification

## 1. 位置付け

GP classification は、連続値を直接回帰する Gaussian Process regression と異なり、
潜在関数 `f(x)` を分類確率へ変換して離散ラベルをモデル化します。robotorchan の現在の
classification API は binary classification を実装し、multiclass へ拡張できる予測契約を
分離しています。

```text
X
 -> latent GP f(X)
 -> likelihood / link
 -> class probability p(y | f)
 -> predicted class / classification uncertainty
```

`model.posterior(X)` は **latent function posterior** です。class probability posterior や
Bernoulli label distributionそのものではありません。この区別は acquisition function と
uncertainty の意味を判断する上で重要です。

## 2. Binary GP classification

binary labelを `y in {0, 1}` とし、潜在関数にGP priorを置きます。

```text
f ~ GP(m, k)
y | f ~ Bernoulli(pi(f))
```

現在の実装は GPyTorch の `BernoulliLikelihood` を使います。linkを通した確率を `p` とすると、

```text
p(y = 1 | f) = p
p(y = 0 | f) = 1 - p
```

です。Gaussian regressionと違い、Bernoulli likelihoodはGaussian likelihoodではないため、
標準的なexact GP regressionと同じclosed-form posteriorにはなりません。robotorchanの標準
binary classifierはBoTorch `SingleTaskVariationalGP`基盤のvariational inferenceを使います。

## 3. Variational inference

近似posteriorを `q(f)` とすると、学習ではvariational ELBOを最大化します。概念的には、

```text
ELBO = E_q(f)[log p(y | f)] - KL(q(u) || p(u))
```

です。`u` は inducing variables です。実装では `make_mll()` が `VariationalELBO` を返します。
したがってbinary classificationをexact Gaussian MLLとして学習してはいけません。

## 4. Posterior と prediction contract

robotorchanでは次を分離します。

| API | 意味 |
| --- | --- |
| `posterior(X)` | BoTorch-compatible latent posterior |
| `latent_posterior(X)` | latent posteriorであることを明示するsemantic alias |
| `predictive_distribution(X)` | likelihoodを通したpredictive label distribution |
| `predict_proba(X)` | class probability |
| `predict_class(X)` | probability thresholdによるclass label |
| `sample_latent(X)` | latent function samples |
| `sample_class_probabilities(X)` | latent samplesをlinkへ通したclass-probability samples |

この契約により、BoTorch側が期待するposterior interfaceを壊さず、分類固有のprobability
semanticsを別APIとして扱えます。

## 5. 分類での「ばらつき」

分類では少なくとも3種類を区別します。

### 5.1 Latent uncertainty

`f(x)` 自体のposterior varianceです。

```text
Var[f(x) | D]
```

これはGPが潜在decision functionをどの程度知らないかを表します。

### 5.2 Class-probability uncertainty

latent posteriorをlinkへ通すと、`p(y=c | f)` 自体もposterior samplesを持ちます。
そのsample varianceは、class probabilityについてのepistemic uncertaintyを表します。

```text
Var_q(f)[p(y=c | f)]
```

`ProbabilityVariance` はこの量を利用します。

### 5.3 Predictive label uncertainty

平均的なclass probabilityを `p` としたとき、実際のlabelはBernoulli random variableです。
binaryではlabel variance `p(1-p)`、より一般にはpredictive entropyで曖昧さを表せます。

```text
H[y | x, D] = - sum_c p_c log p_c
```

これはclass probability posteriorのvarianceとは同じではありません。

## 6. Active Learningとの接続

現在のclassification Active Learningはすべて `q=1` です。

| Acquisition | 主に使う不確実性 | Multiclass拡張 |
| --- | --- | --- |
| `PredictiveEntropy` | predictive label entropy | 自然 |
| `MarginUncertainty` | top-two class probability margin | 自然 |
| `ProbabilityVariance` | class-probability posterior variance | 自然 |
| `BALD` | predictive entropyとexpected conditional entropyの差 | 自然 |
| `LatentStraddle` | latent mean / varianceとdecision boundary | binary固有 |

BALDは概念的に、

```text
I[y, f | x, D] = H[y | x, D] - E_q(f)[H[y | f, x]]
```

を評価し、label ambiguityのうちlatent/model uncertaintyに由来する部分を選択します。

Gaussian latent posteriorを持つという理由だけで、回帰用 `Straddle` やqEIなどを分類へ
自動的に適用してはいけません。robotorchanのcapability metadataはobservation semanticsを
分離し、この誤接続を静的に拒否します。低レベルBoTorch APIを直接使う経路は維持します。

## 7. 実装済みモデルファミリー

現在のbinary classificationには次の系統があります。

| 構造 | 主なモデル |
| --- | --- |
| Standard | `BinarySingleTaskGPClassifier` |
| Mixed | `MixedBinarySingleTaskGPClassifier` |
| MultiTask | `MultiTaskBinaryGPClassifier` |
| Block-design MultiTask | `KroneckerMultiTaskBinaryGPClassifier` |
| Independent composition | `ClassificationModelList` |
| High-dimensional prior | MAP-SAAS / SAAS binary classifier |
| Reduced space | Reduced / PCA / PLS / Random Projection binary classifier |
| Random embedding | `ALEBOBinarySingleTaskGPClassifier` |
| Neural representation | `JointEncoderBinaryGPClassifier` |
| Deep GP | `BinarySingleTaskDeepGPClassifier` |

回帰モデルと名前を揃えること自体を目的にせず、Bernoulli likelihoodとclassification inferenceが
理論的に自然なfamilyだけを分類側へ移植します。Gaussian observation noiseを直接モデル化する
heteroskedastic GPやStudent-t observation modelなどは、そのまま分類器へ置換できません。

## 8. Multiclassへの拡張境界

binary実装は0/1 labelを厳密に検証します。将来のmulticlassではlabelを `0, ..., C-1` とし、
Categorical likelihood / multiclass latent representationを導入する必要があります。

共通層で固定してはいけないものは次です。

- Bernoulli likelihood
- class数2
- probability threshold 0.5
- latent decision boundary 0

一方、`predict_proba`、`predict_class`、class-probability sampling、predictive entropyという
高レベル契約はmulticlassでも維持できます。

## 9. 参考文献

- Hensman, J., Matthews, A. G. de G., & Ghahramani, Z. (2015). Scalable Variational
  Gaussian Process Classification. AISTATS.
- Gal, Y., Islam, R., & Ghahramani, Z. (2017). Deep Bayesian Active Learning with Image
  Data. ICML. BALDの実用的なclassification active-learning利用を扱います。
- Eriksson, D., Jankowiak, M. (2021). High-Dimensional Bayesian Optimization with Sparse
  Axis-Aligned Subspaces. UAI. SAAS priorの背景です。

## 10. 関連ドキュメント

- [Variational GP](09_variational_gp.md)
- [Mixed Variables](05_mixed_variables.md)
- [Multi-task / Multi-output](07_multitask_multioutput.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
- [Dimensionality Reduction GP](18_dimensionality_reduction_gp.md)
- [Expressive GP Models](22_expressive_gp.md)
- [Classification Active Learning](acquisition/classification-active-learning.md)
- [Classification model guide](../models/classification.md)


## High-dimensional cross-family composition

High-dimensional classification does not require a separate named model for every combination with
calibration, conformal prediction, ensembles, or reliability diagnostics. These layers operate on
the common probability contract and compose with MAP-SAAS, reduced-space, ALEBO, and joint-encoder
classifiers.

A new model family is justified when the combination changes latent covariance or representation
semantics. Nonstationary covariance and Multi-Fidelity are examples that require explicit design:
fidelity/task/category coordinates must remain structural, while reduction, random embedding, and
sparsity priors operate only on the intended design coordinates.

This distinction prevents family-parity work from producing thin wrappers while keeping low-level
BoTorch/GPyTorch model components available for genuinely new covariance constructions.
