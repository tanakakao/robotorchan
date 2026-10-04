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
