# Classification：latent GP posteriorとclass probabilityを分離する

## 1. 分類GPの確率モデル

二値ラベル`y∈{0,1}`を扱うとき、実数値潜在関数`f(x)`へGP priorを置き、Bernoulli likelihood経由でラベルを生成する。[理論第24章](../../../theory/24_classification.md)のとおり、Gaussian priorにBernoulli尤度は非共役なので、通常は近似推論（variationalなど）が必要となる。

**`posterior(X)`はlatent posteriorであって、成功確率ではない。** `predict_proba(X)`はclass probabilities、`predict_class(X)`は決定規則に従うラベルを返す。これらを混同すると、constrained BOやActive Learningの数式の対象が変わってしまう。

## 2. 理論と実装の対応

| 数学的対象 | 公開契約 | 対応する意味 |
| --- | --- | --- |
| 潜在関数`f(x)`の近似分布 | `model.posterior(X)` | BoTorch-compatibleなlatent値・不確実性 |
| `E[link(f(x)) \| D]` | `model.predict_proba(X)` | 予測クラス確率 |
| 確率からの分類判断 | `model.predict_class(X)` | 閾値・クラス選択に基づく判断 |
| 不確実性の学習 | `model.make_mll()`など | Bernoulli likelihoodに対するvariational目的（Exact Gaussian MLLではない） |
| 目的／制約へ接続 | `robotorchan.acquisition.composition` | 確率objective・classification feasibilityのbridge |

## 3. モデル群の選択

`BinarySingleTaskGPClassifier`は連続入力の基礎モデル。`MixedBinarySingleTaskGPClassifier`ではcategorical列と連続設計列を分ける。`MultiTaskBinaryGPClassifier`と`KroneckerMultiTaskBinaryGPClassifier`ではlong-format / block-designの入力構造が異なる。

高次元ではSAAS-style分類がvariationalであり、回帰のNUTSベース`SaasFullyBayesianSingleTaskGP`と同じ推論方式ではない。DeepGP分類は確率的な層を持つ。Non-GP分類器がclass probabilityを返す場合でもGaussian latent posteriorが存在するとは限らず、`posterior`要件のある獲得関数へ無条件に渡せない。

## 4. 利用例：予測量の違いを見る

```python
import torch
from robotorchan.models.classification import BinarySingleTaskGPClassifier

train_X = torch.rand(20, 3, dtype=torch.double)
train_Y = torch.randint(0, 2, (20,), dtype=torch.double)
model = BinarySingleTaskGPClassifier(train_X, train_Y)

X = torch.rand(5, 3, dtype=torch.double)
latent_posterior = model.posterior(X)
class_probabilities = model.predict_proba(X)
predicted_labels = model.predict_class(X)
print(latent_posterior.mean.shape, class_probabilities.shape, predicted_labels.shape)
```

この例はAPIの静的な説明であり、fitや校正を検証した結果ではない。

## 5. Active Learning・制約付きBOでの解釈

`PredictiveEntropy`は予測ラベルのentropy、`BALD`は情報量、`LatentStraddle`はlatent decision boundary近傍の探索である。これらは単一の「uncertainty」として代替できない。`ProbabilityVariance`は予測確率の変動を扱い、latent posterior varianceとは異なる。分類モデルを可行性`P(pass\|x)`として使う場合、objectiveが直接潜在GP値ではなく**実際の分類確率**を見るようにbridgeを設計する。

## 6. 校正・未対応・検証の範囲

Brier score、NLL、ECEなどはclass probabilityの予測校正を評価するための指標であり、latent varianceの値を直接比較する指標ではない。現行文書の分類モデルはbinary-firstであり、全multiclass対応を主張しない。出力の全familyやmulti-fidelity分類などの対応可否は[Classificationモデル一覧](../../../models/classification.md)で個別に確認する。

[Classification theory](../../../theory/24_classification.md)、[Classification model guide](../../../models/classification.md)、[旧Mapping](../models.md)、[Acquisition Mapping](../acquisition.md)、[Verification](../verification.md)を併読する。

## 7. Bernoulli likelihoodがlatent posteriorをクラス確率へ変える

BinarySingleTaskGPClassifierはBoTorchのVariational GPを基礎に、GPyTorch BernoulliLikelihoodとVariationalELBOを使う。近似潜在分布q(f(x))に対して

$$
P(y=1\mid x,\mathcal D)
=\mathbb E_{q(f(x))}[P(y=1\mid f(x))]
$$

とするため、一般にlatent posterior meanを単にsigmoidやprobit変換した値と同一視できない。src/robotorchan/models/classification/binary/standard/single_task.pyではpredictive_distribution()がlatent_posterior(X).distributionをself.likelihoodへ渡し、predict_proba()は結果のpositive probabilityから[P(y=0),P(y=1)]を構築する。sample_class_probabilities()は別経路としてlatent samplesから条件付き確率を生成する。

predict_class()はpredict_proba()[...,1]に閾値を適用するだけであり、クラス確率の校正を保証しない。閾値は0〜1の有限数である必要があり、範囲外を拒否する。Bernoulli likelihoodに対するELBOを利用する点はExact GPのGaussian MLLと異なる。

**注意：** APIが確率を返すことはBrier・ECE・NLLの十分な性能を意味せず、個別のデータ・学習条件で評価する必要がある。今回の照合はソース契約のみで、数値校正実験は行っていない。
