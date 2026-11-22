# Classification：latent GP posteriorとclass probabilityを分離する

## 分類GPの確率モデル

二値ラベル`y∈{0,1}`を扱うとき、実数値潜在関数`f(x)`へGP priorを置き、Bernoulli likelihood経由でラベルを生成する。
[理論第24章](../theory/24_classification.md)のとおり、Gaussian priorにBernoulli尤度は非共役なので、通常は近似推論（variationalなど）が必要となる。

**`posterior(X)`はlatent posteriorであって、成功確率ではない。** `predict_proba(X)`はclass probabilities、
`predict_class(X)`は決定規則に従うラベルを返す。これらを混同すると、constrained BOやActive Learningの数式の対象が変わってしまう。

## 理論と実装の対応

| 数学的対象 | 公開契約 | 対応する意味 |
| --- | --- | --- |
| 潜在関数`f(x)`の近似分布 | `model.posterior(X)` | BoTorch-compatibleなlatent値・不確実性 |
| `E[link(f(x)) \| D]` | `model.predict_proba(X)` | 予測クラス確率 |
| 確率からの分類判断 | `model.predict_class(X)` | 閾値・クラス選択に基づく判断 |
| 不確実性の学習 | `model.make_mll()`など | Bernoulli likelihoodに対するvariational目的（Exact Gaussian MLLではない） |
| 目的／制約へ接続 | `robotorchan.acquisition.composition` | 確率objective・classification feasibilityのbridge |

## 分類確率のBoTorch Posterior（標準二値Probit GP）

`BinarySingleTaskGPClassifier` では、潜在関数に対する
`posterior(X)` と、分類確率に対する `probability_posterior(X)` を
区別して使用できます。後者はBoTorchの `Posterior` を継承しており、
`mean`・`variance`・`rsample`・`rsample_from_base_samples` を提供します。

```python
import torch

X = torch.linspace(0.0, 1.0, 20).unsqueeze(-1)
probability_posterior = model.probability_posterior(X)

mean = probability_posterior.mean  # [20, 2]
variance = probability_posterior.variance  # [20, 2]
samples = probability_posterior.rsample(torch.Size([1000]))  # [1000, 20, 2]
lower, upper = probability_posterior.credible_interval(0.95)
```

ここで `variance` は分類確率そのものの事後分散であり、
観測ラベルのBernoulli分散ではありません。
二値Probitでは潜在関数がGaussianで、観測確率が単調変換なので、
各クラスの等確率両側95%信用区間は**サンプリングせずに**求められます。
クラス0の下限は「クラス1の上限」を1から引いて計算します。
`credible_interval` はrobotorchanの拡張であり、
BoTorchの共通 `Posterior` インターフェースに含まれるものではありません。

BoTorchのPosteriorに合わせて、この新APIでは候補次元 `q` を維持します。
例えば入力が `[batch, 1, d]` なら
`probability_posterior(X).mean` は `[batch, 1, 2]` を返します。
従来の `predict_proba(X)` が `[batch, 2]` を返す場合でも、
既存APIの返却形状は変更しません。

BoTorchの `TransformedPosterior` は前後でサンプル形状が
同じであることを前提としています。今回は潜在出力1次元から
分類確率2次元へ変化するため、既存の潜在Posteriorを内部で使用し、
分類確率の出力形状のみを扱う軽量な `Posterior` 拡張を用います。

## マルチクラス・順序分類の確率Posterior

マルチクラスSoftmax GPのQMC推定は[第30章ガイド](30_multiclass_classification.md)、Ordinal GPの解析的予測確率と求積・区間推定は[第31章ガイド](31_ordinal_classification.md)へ移しました。

## EnsembleのBoTorch Posteriorとの整合

`ClassificationEnsemblePosterior` はBoTorch公開APIの
`EnsemblePosterior` を継承します。
メンバーの確率ベクトルをそのまま利用し、
`mean`、`variance`、`rsample`、
`rsample_from_base_samples` をBoTorchのPosterior契約で扱います。

```python
import torch
from botorch.sampling.get_sampler import get_sampler

# GPBinaryClassificationEnsemble、HeterogeneousBinaryClassificationEnsemble
# などの既存の分類Ensembleを使用する
posterior = ensemble.probability_posterior(test_X)

mean = posterior.mean
variance = posterior.variance
lower, upper = posterior.credible_interval(0.95)

sampler = get_sampler(posterior, sample_shape=torch.Size([128]), seed=7)
samples = sampler(posterior)
```

BoTorchの `IndexSampler` が選択され、
**1回のサンプルにつき各バッチのEnsembleメンバーを1つ選択**します。
同じバッチの `q` 個の候補には同一メンバーを用いるため、
各候補で独立にメンバーを引く場合と異なり、
メンバー間の予測パターンを保持します。

特に`GPBinaryClassificationEnsemble`では、
入力形状`[batch, 1, d]`のとき、各メンバーの従来の
`predict_proba`が`q=1`の候補軸を省略します。
`probability_posterior(X)`では候補軸を**内部で復元**し、
`posterior.mean`・`posterior.variance`を
`[batch, 1, classes]`として返します。
これによりBoTorchの`IndexSampler`はバッチを候補と取り違えず、
**バッチごとにメンバーを選び、各バッチの候補間では同一メンバーを保持**します。

一方、同種GP Ensembleの既存`predict_proba(X)`・
`probability_variance(X)`・`sample_class_probabilities(X)`などは
`q=1`で候補軸を省略する**従来の返却形状を維持**します。
新しいPosterior契約と旧予測APIでは形状が異なる点に注意してください。
入力`[n, d]`や`q>1`の既存動作は変更しません。

分散は従来のrobotorchanと同じ
`sum(w_i * (p_i - mean)^2)` であり、
BoTorchの `EnsemblePosterior.variance` が使う補正付き標本分散とは
意図的に異なります。
`credible_interval` はクラスごとに
メンバーの**加重経験分布の正確な周辺分位点**を計算します。
メンバー数が少ない場合は離散的な限界値になり、
信用区間の校正や同時被覆を保証するものではありません。
`num_samples` 引数は共通API互換のため保持されますが、
この経験分布の区間計算では使用しません。

| 確率Posterior | 不確実性の意味 | 信用区間の計算 |
| --- | --- | --- |
| 単一GP（二値・多クラス・Ordinal） | 潜在関数の事後分布に由来 | 解析的／求積／MC |
| 分類Ensemble | 完成済みメンバー予測の不一致 | 加重経験分位点 |

**注意：** GP Ensembleであっても、
今回の `probability_posterior` はメンバーの平均予測確率を集計します。
各メンバー内部のGP潜在不確実性をさらに周辺化した
階層的な混合Posteriorではありません。
各メンバーの潜在Posteriorを見る場合は
`member_latent_posteriors(X)` を使います。
既存の `predict_proba`、`probability_variance`、
Active Learning獲得関数の計算内容は変更しません。


## モデル群の選択

`BinarySingleTaskGPClassifier`は連続入力の基礎モデル。
`MixedBinarySingleTaskGPClassifier`ではcategorical列と連続設計列を分ける。
`MultiTaskBinaryGPClassifier`と`KroneckerMultiTaskBinaryGPClassifier`ではlong-format /
block-designの入力構造が異なる。

高次元ではSAAS-style分類がvariationalであり、回帰のNUTSベース`SaasFullyBayesianSingleTaskGP`と同じ推論方式ではない。
DeepGP分類は確率的な層を持つ。Non-GP分類器がclass probabilityを返す場合でもGaussian latent posteriorが存在するとは限らず、
`posterior`要件のある獲得関数へ無条件に渡せない。

## Active Learning・制約付きBOでの解釈

`PredictiveEntropy`は予測ラベルのentropy、`BALD`は情報量、`LatentStraddle`はlatent decision boundary近傍の探索である。
これらは単一の「uncertainty」として代替できない。`ProbabilityVariance`は予測確率の変動を扱い、latent posterior varianceとは異なる。
分類モデルを可行性`P(pass\|x)`として使う場合、objectiveが直接潜在GP値ではなく**実際の分類確率**を見るようにbridgeを設計する。

## 確率の解釈とモデルの制限

Brier score、NLL、ECEなどはclass probabilityの予測校正を評価するための指標であり、latent varianceの値を直接比較する指標ではない。
現行文書の分類モデルはbinary-firstであり、全multiclass対応を主張しない。出力の全familyやmulti-fidelity分類などの対応可否は
[Classificationモデル一覧](../models/classification.md)で個別に確認する。

## Bernoulli likelihoodがlatent posteriorをクラス確率へ変える

BinarySingleTaskGPClassifierはBoTorchのVariational GPを基礎に、
GPyTorch BernoulliLikelihoodとVariationalELBOを使う。近似潜在分布q(f(x))に対して

$$
P(y=1\mid x,\mathcal D)
=\mathbb E_{q(f(x))}[P(y=1\mid f(x))]
$$

とするため、一般にlatent posterior meanを単にsigmoidやprobit変換した値と同一視できない。
src/robotorchan/models/classification/binary/standard/single_task.pyではpredictive_distribution()が
latent_posterior(X).distributionをself.likelihoodへ渡し、
predict_proba()は結果のpositive probabilityから[P(y=0) ,P(y=1)]を構築する。
sample_class_probabilities()は別経路としてlatent samplesから条件付き確率を生成する。

predict_class()はpredict_proba()[...,1]に閾値を適用するだけであり、クラス確率の校正を保証しない。閾値は0〜1の有限数である必要があり、範囲外を拒否する。
Bernoulli likelihoodに対するELBOを利用する点はExact GPのGaussian MLLと異なる。

## HeterogeneousModelで回帰と分類の意味論を保つ

src/robotorchan/models/heterogeneous.pyのHeterogeneousModelはtorch.nn.Moduleのcontainerであり、
BoTorchの単一Modelとして全outputを結合したposterior()を提供しない。entry_posterior(key,X)は子のposteriorをそのまま返すため、
分類子ではlatent distributionになる。entry_predict_proba(key,X)はClassificationModelMixinの子にのみ使える。
回帰子で呼ぶとTypeErrorになる。

分類子と回帰子を併用する制約付きBOでは、回帰acquisitionの価値とP(pass|x)を異なる数理量として扱う。分類出力をGaussian regressionの1列として連結しない。
global output indexは各子のnum_outputsに応じて割り当て、output_ownerが子モデルのindexとlocal output indexを返す。
output_namesやnamesは別名であり、位置indexが基準となる。

ModelListGPはBoTorch ModelListGPの独立GP合成とSumMarginalLogLikelihoodを用いるもので、
分類確率と回帰posteriorの意味論を管理するHeterogeneousModelとは責務が異なる。ModelListGPを異種モデルの単一posteriorとして説明してはいけない。

## 分類確率を目的と制約で使い分ける

同じpredict_proba(X)[..., c]でも、最大化する対象と可行性の判定に用いる対象では意味が異なる。
ProbabilityObjective(output, class_index, direction)は対象クラスの予測確率をスカラー目的として評価する。
例えば良品になる確率自体を最大化する場合である。一方、ClassificationProbabilityOfFeasibility(model,
feasible_class)はクラス確率をP(feasible|X,D)として取得し、回帰目的の獲得関数に外部重みとして用いる。

- **分類確率を最大化・最小化**
  公開契約：ProbabilityObjective.evaluate(model,X)
  実装上の区別：指定クラス列を抽出しObjectiveDirectionを適用。RegressionObjectiveではない

- **分類確率から潜在posteriorに基づくサンプルを生成**
  公開契約：ProbabilityObjective.sample(model,X,sample_shape=...)
  実装上の区別：classifier.sample_class_probabilitiesに依存。全分類器で利用可能とは限らない

- **分類を制約として利用**
  公開契約：ClassificationProbabilityOfFeasibility(X)
  実装上の区別：predict_probaからfeasible_classの列を抽出。一般のjoint可行性ではない

- **サンプル単位の可行確率**
  公開契約：SampleProbabilityOfFeasibility
  実装上の区別：epistemic sampleの確率。平均PoFをそのままMC sampleへ置き換えない

- **観測値に基づく連続制約**
  公開契約：SampleResidualFeasibility
  実装上の区別：posterior sampleから残差を計算し、残差 <= 0で可行

- **予測確率そのものの閾値制約**
  公開契約：ProbabilityResidualFeasibility
  実装上の区別：決定論的な確率残差。P(別の制約が成立)とは異なる

ProbabilityObjectiveのdirection=MINIMIZEは対象確率へ方向変換を適用するが、確率の物理的定義が負になるわけではない。分類制約では可行クラス番号を明示し、
ラベルが0を可行とする問題で既定値feasible_class=1を誤用しない。class_indexはclassification metadataから解決され、
未対応ラベル・回帰モデルへの指定は拒否される。

### posterior predictiveとepistemic sampleは別の期待値

~~~text
latent posterior q(f|D)
    ├─ likelihoodによる周辺化 → predict_proba(X) → deterministic PoF
    └─ latent sample → sample_class_probabilities(X) → sample-wise PoF
~~~

予測確率の平均と、潜在関数posterior由来の確率sampleは異なる。
src/robotorchan/acquisition/composition.py::make_sample_classification_feasibility_bridgeは
classification registryのsupports_probability_samplesをチェックし、非対応ならTypeErrorで拒否する。
サンプル化できない分類器に平均PoFを複製して代替しない。

## 関連する実装資料

ライブラリ固有のモデル群、予測API、対応する組合せについては
[分類モデルガイド](../models/classification.md)を参照してください。

## 変分GPを分類器へ拡張する継承

[binary/standard/single_task.py][binary]の`BinarySingleTaskGPClassifier`は
`BinaryClassificationMixin`と`SingleTaskVariationalGP`を継承します。近似GPの計算を再利用し、
Bernoulli likelihoodと分類用ELBOへ接続します。`predictive_distribution()`がlatent分布をlikelihoodへ渡し、
`predict_proba()`がその正例確率から2クラス分のtensorを構成します。

Theoryのlink関数を介した期待値は、このlatent分布から予測ラベル分布への変換に対応します。
潜在sampleからの`sample_class_probabilities()`は別の経路であり、平均確率を複製してepistemic sampleにする処理ではありません。
クラス確率の構成・分類の判断はローカル拡張で、Bernoulli分布と変分GPの演算はGPyTorch・BoTorchの担当です。

[binary]: ../../src/robotorchan/models/classification/binary/standard/single_task.py

使用手順と結果の可視化は[26番Notebook](../../examples/notebooks/26_classification_active_learning.ipynb)を参照してください。

## 不確実性・確率校正・予測集合の責務分離

[分類理論の第6.2節](../theory/24_classification.md)では、
ラベルの誤記録、候補入力の揺らぎ、Ensembleによる確率のばらつき、
確率校正、OOD、Conformalの違いを説明しています。
これらはすべて「不確実性」と関連しますが、
操作する確率分布もモデルの契約も異なります。

| 問題 | 対応する実装 | 予測への影響 |
| --- | --- | --- |
| 観測ラベルの誤記録 | LabelNoiseBinarySingleTaskGPClassifier | 観測ラベルのBernoulli尤度を変更 |
| 候補入力の揺らぎ | ContinuousUncertainInputBinarySingleTaskGPClassifier | 摂動後の入力に対する予測確率を平均 |
| モデル間の不一致 | ClassificationEnsemblePosterior | メンバー確率の平均と分散を計算 |
| 予測確率の校正 | TemperatureScalingCalibrator | クラス確率を温度で変換 |
| 校正後の分類器 | CalibratedBinaryClassifier | 確率の予測と確率サンプルへ校正を適用 |
| 予測集合の構成 | SplitConformalClassifier | 保留データに基づいて予測集合を返す |

LabelNoiseBinarySingleTaskGPClassifierでは、
観測されたラベルの確率とラベル反転前の確率を区別します。
標準のpredict_probaは観測過程を含む予測です。
ラベル反転前の確率はpredict_clean_probaを参照します。

ContinuousUncertainInputBinarySingleTaskGPClassifierの
predict_proba_with_input_uncertaintyは、
候補時点の連続入力にGaussian摂動を加えて確率を平均します。
これは訓練入力自体の測定誤差を自動で修正する手法ではありません。

Temperature ScalingとSplit Conformalはどちらも
保留データを使いますが、前者は確率を変換し、
後者は指定された被覆率を目指す予測集合を構成します。
Conformalの被覆率は交換可能性などの仮定に依存し、
各入力における条件付き被覆率を保証するわけではありません。

対応する実行例は
[28番Notebook](../../examples/notebooks/28_classification_robust_uncertain.ipynb)
と[29番Notebook](../../examples/notebooks/29_classification_probability_layers.ipynb)
です。41番ではConformalの概念を説明しますが、
予測集合の実際の算出は行っていません。

## 単一点とq-batchの分類Active Learning

[分類獲得関数の理論](../theory/acquisition/13_classification_active_learning.md)
では、予測Entropy、Margin、Probability Variance、BALD、
潜在Straddleの数学的な違いを説明しています。
[獲得関数Guide](acquisition/13_classification_active_learning.md)
では、それぞれが要求するモデルの公開メソッドと
q-batchの対応方式を確認できます。

単一点のPredictiveEntropy等はq=1用です。
qPredictiveEntropy・qMarginUncertainty・qProbabilityVariance・qLatentStraddleは
周辺スコアを平均します。一方qBALDは同時ラベル情報利得を推定し、
二値ではq=1〜8、マルチクラスと順序分類ではクラス数Cについて
`C**q <= 256`の上限があります。
候補間の重複を直接考慮するかどうかが異なります。

## 拡張二値分類モデルの使い分け

評価忠実度が異なる場合は`MultiFidelityBinaryGPClassifier`、
連続とカテゴリの混在には`MixedMultiFidelityBinaryGPClassifier`を選びます。
周期的な境界にはSpectral Mixture、Contextごとの差にはContextual、
親子構造を持つデータにはHierarchicalを検討します。
ReLUネットワーク由来の非定常なカーネルを試す場合はInfinite-width BNNを選びます。

どのモデルでも、学習には`fit()`、クラス確率には`predict_proba()`、
潜在関数には`posterior()`を使用します。各モデルの構築引数と選択表は
[モデル一覧](../models/classification.md)、実行コードは
[分類拡張Notebook](../../examples/notebooks/27_binary_classification_extensions.ipynb)を参照してください。


## マルチクラス・順序分類の独立ガイド

- [マルチクラスGP分類の理論と実装](30_multiclass_classification.md)
- [順序GP分類の理論と実装](31_ordinal_classification.md)
- [分類制約付きBOと回帰モデルとの合成](32_heterogeneous_classification_constraints.md)
