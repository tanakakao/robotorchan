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

## 8. HeterogeneousModelで回帰と分類の意味論を保つ

src/robotorchan/models/heterogeneous.pyのHeterogeneousModelはtorch.nn.Moduleのcontainerであり、BoTorchの単一Modelとして全outputを結合したposterior()を提供しない。entry_posterior(key,X)は子のposteriorをそのまま返すため、分類子ではlatent distributionになる。entry_predict_proba(key,X)はClassificationModelMixinの子にのみ使える。回帰子で呼ぶとTypeErrorになる。

分類子と回帰子を併用する制約付きBOでは、回帰acquisitionの価値とP(pass|x)を異なる数理量として扱う。分類出力をGaussian regressionの1列として連結しない。global output indexは各子のnum_outputsに応じて割り当て、output_ownerが子モデルのindexとlocal output indexを返す。output_namesやnamesは別名であり、位置indexが基準となる。

ModelListGPはBoTorch ModelListGPの独立GP合成とSumMarginalLogLikelihoodを用いるもので、分類確率と回帰posteriorの意味論を管理するHeterogeneousModelとは責務が異なる。ModelListGPを異種モデルの単一posteriorとして説明してはいけない。

**検証境界：** 上記はmodels/heterogeneous.pyおよびstandard/model_list.pyの静的実装契約。確率校正、複数目的＋分類制約のend-to-end最適化の数値検証を今回実施したわけではない。

## 9. 分類確率を目的と制約で使い分ける

同じpredict_proba(X)[..., c]でも、最大化する対象と可行性の判定に用いる対象では意味が異なる。ProbabilityObjective(output, class_index, direction)は対象クラスの予測確率をスカラー目的として評価する。例えば良品になる確率自体を最大化する場合である。一方、ClassificationProbabilityOfFeasibility(model, feasible_class)はクラス確率をP(feasible|X,D)として取得し、回帰目的の獲得関数に外部重みとして用いる。

| 意味論 | 公開契約 | 実装上の区別 |
| --- | --- | --- |
| 分類確率を最大化・最小化 | ProbabilityObjective.evaluate(model,X) | 指定クラス列を抽出しObjectiveDirectionを適用。RegressionObjectiveではない |
| 分類確率から潜在posteriorに基づくサンプルを生成 | ProbabilityObjective.sample(model,X,sample_shape=...) | classifier.sample_class_probabilitiesに依存。全分類器で利用可能とは限らない |
| 分類を制約として利用 | ClassificationProbabilityOfFeasibility(X) | predict_probaからfeasible_classの列を抽出。一般のjoint可行性ではない |
| サンプル単位の可行確率 | SampleProbabilityOfFeasibility | epistemic sampleの確率。平均PoFをそのままMC sampleへ置き換えない |
| 観測値に基づく連続制約 | SampleResidualFeasibility | posterior sampleから残差を計算し、残差 <= 0で可行 |
| 予測確率そのものの閾値制約 | ProbabilityResidualFeasibility | 決定論的な確率残差。P(別の制約が成立)とは異なる |

ProbabilityObjectiveのdirection=MINIMIZEは対象確率へ方向変換を適用するが、確率の物理的定義が負になるわけではない。分類制約では可行クラス番号を明示し、ラベルが0を可行とする問題で既定値feasible_class=1を誤用しない。class_indexはclassification metadataから解決され、未対応ラベル・回帰モデルへの指定は拒否される。

### 9.1 posterior predictiveとepistemic sampleは別の期待値

~~~text
latent posterior q(f|D)
    ├─ likelihoodによる周辺化 → predict_proba(X) → deterministic PoF
    └─ latent sample → sample_class_probabilities(X) → sample-wise PoF
~~~

予測確率の平均と、潜在関数posterior由来の確率sampleは異なる。src/robotorchan/acquisition/composition.py::make_sample_classification_feasibility_bridgeはclassification registryのsupports_probability_samplesをチェックし、非対応ならTypeErrorで拒否する。サンプル化できない分類器に平均PoFを複製して代替しない。

**照合根拠：** src/robotorchan/semantics/objectives.py、semantics/feasibility.py、semantics/probability.py、acquisition/composition.py。出力shapeや確率校正の独立数値実行は今回行っていない。
