# Classification models

robotorchan の classification model は、BoTorch-compatible なlatent posteriorと、分類固有の
class probability / label predictionを分離します。現在は二値分類・マルチクラス分類・
順序分類（Ordinal GP）を公開しており、モデルごとの出力構造は異なります。

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
probability_posterior = model.probability_posterior(X)
probability_mean = probability_posterior.mean
probability_variance = probability_posterior.variance
lower, upper = probability_posterior.credible_interval(0.95)
labels = model.predict_class(X)
```

`posterior(X)` はclass probabilityではなくlatent GP posteriorです。確率が必要な場合は
`predict_proba(X)` を使ってください。

## モデル選択

| 問題 | モデル / family |
| --- | --- |
| 通常の連続入力binary classification | `BinarySingleTaskGPClassifier` |
| continuous + categorical | `MixedBinarySingleTaskGPClassifier` |
| 順序なしの3クラス以上 | `MulticlassSingleTaskGPClassifier` / `MixedMulticlassSingleTaskGPClassifier` |
| 順序ありの3クラス以上 | `OrdinalSingleTaskGPClassifier` / `MixedOrdinalSingleTaskGPClassifier` |
| task featureを持つlong-format data | `MultiTaskBinaryGPClassifier` |
| 全taskが同じXを共有するblock design | `KroneckerMultiTaskBinaryGPClassifier` |
| 独立classifierの合成 | `ClassificationModelList` |
| sparse high-dimensional relevance | MAP-SAAS / SAAS classifier |
| PCA / PLS等で入力を縮約 | reduced-space classifiers |
| random embedding | `ALEBOBinarySingleTaskGPClassifier` |
| learned representation | `JointEncoderBinaryGPClassifier` |
| stochastic deep hierarchy | `BinarySingleTaskDeepGPClassifier` |

## 予測・学習の使い分け

`posterior(X)` は潜在関数の分布であり、class probability ではありません。確率には `predict_proba`、class の決定には `predict_class`、観測 label の分布には `predictive_distribution` を使います。`sample_latent` と `sample_class_probabilities` も別の量です。二値分類ではラベルは0/1、
マルチクラス・順序分類では0から`num_classes - 1`までの整数を指定します。
文字列や任意整数の暗黙変換は行いません。

標準 binary GP は Bernoulli 尤度の変分モデルです。`make_mll()` は `VariationalELBO` を構成します。Exact Gaussian GP と同じ学習契約と仮定しないでください。元データは raw training-data 契約に従い保持します。

### 分類確率Posteriorと不確実性の使い分け

`posterior(X)` は**潜在GPの事後分布**、`predict_proba(X)` は
**平均予測クラス確率**です。クラス確率そのものの不確実性は
`probability_posterior(X)` の `mean` / `variance` で確認します。
このPosteriorはBoTorchの `Posterior` としてサンプリングできます。
`credible_interval(0.95)` はクラス確率の**周辺95%信用区間**であり、
観測ラベルの予測区間でも、全クラスの同時95%領域でもありません。

| 対象モデル | `probability_posterior(X).mean` | `variance` | `credible_interval` |
| --- | --- | --- | --- |
| 標準Binary Probit GP | 解析的 | Gauss–Hermite求積 | 潜在Gaussian分位点の単調変換（MC不要） |
| Ordinal GP | 解析的 | Gauss–Hermite求積 | 潜在サンプルによるクラス別MC分位点 |
| Multiclass Softmax GP | 同一QMCサンプルから推定 | 同一QMCサンプルから推定 | クラス別MC分位点 |
| 分類Ensemble | メンバー確率の加重平均 | メンバー間の加重分散 | 加重経験分布の分位点 |

Standard/MixedのOrdinal・Multiclass GPで利用できますが、
全分類モデルに同じ確率Posterior APIが実装されていることを意味しません。
Ensembleの分散は**メンバー間の予測のばらつき**であり、
個々のGPの潜在事後分散ではありません。

`probability_variance(X)` と `ProbabilityVariance` は
**クラス確率の事後分散**を評価し、観測ラベルのBernoulli分散とは区別します。
標準Binary Probit GPの `probability_variance` および
`mutual_information`（BALD）はGauss–Hermite求積が既定です。
`method="mc"` を指定すると従来のMC方式も利用でき、
このときのみ `num_samples` がMCサンプル数を制御します。
Ordinal/Multiclassの `ProbabilityVariance` と `BALD` は既存のMC経路を使うため、
確率PosteriorのQMC/求積と有限サンプルで厳密一致するとは限りません。
一方、予測EntropyとMarginは平均予測クラス確率に基づく別の指標です。

実装と形状は[分類ガイド](../guide/24_classification.md)、
獲得関数との対応は[分類Active Learningガイド](../guide/acquisition/13_classification_active_learning.md)、
可視化は[26番Notebook](../../examples/notebooks/26_classification_active_learning.ipynb)、
[29番Notebook](../../examples/notebooks/29_classification_probability_layers.ipynb)、
[30番Notebook](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)、
[31番Notebook](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)
を参照してください。

## Robust モデル

誤 label・汚染・反復 label・入力依存の信頼性は観測過程として表します。連続残差用の Student-t を binary label に機械的に移植しません。非定常性は潜在共分散を変える別の機構です。likelihood と互換潜在モデルの合成を優先し、全 family の直積 class を作りません。具体的な registry ID と組合せ条件は[詳細契約](../../docs-internal/contracts/classification-models.md)にまとめています。

## Active Learning と BO 制約

`PredictiveEntropy`・`MarginUncertainty` は予測クラス確率を評価します。`ProbabilityVariance` はクラス確率そのものの事後分散、`BALD` は潜在関数とラベルの相互情報量（Ensembleではメンバー間の不一致）を評価します。標準Binary Probit GPのProbability VarianceとBALDは既定でGauss–Hermite求積を使い、MCサンプリングは必須ではありません。`LatentStraddle` は二値の潜在境界を使います。単一点版は `q=1` 専用で、`qPredictiveEntropy`・`qMarginUncertainty`・`qProbabilityVariance` は周辺スコア平均、`qBALD` は候補間の依存を含む同時ラベル情報利得です。潜在 posterior がGaussianというだけで回帰用獲得関数に対応するとはみなしません。

`ClassificationProbabilityOfFeasibility` は指定 class の実行可能確率を取得し、`FeasibilityWeightedAcquisition` は目的獲得値に重みを掛けます。q-batch の既定 `product` は点間独立の近似、`minimum` は保守的なスコアで、joint 分類 posterior を作るものではありません。既知の入力に対する `CandidateConstraints` とは別の層です。

`RobustProbabilityTransform` は scenario ごとの class 確率を expectation・worst-case・VaR・CVaR で集約します。class ごとの下側 tail は異なる scenario に由来し得るため、集約値の和が 1 になるとは限りません。新たな確率分布として解釈せず、特定 class が必要なら `robust_class_probability` を使います。

## Non-GP と ensemble

Random Forest・Extra Trees・Gradient Boosting・Histogram Gradient Boosting の adapter は予測確率を提供しますが、GP の潜在 posterior を持つとはみなしません。単一推定器の確率だけから epistemic 分散・BALD を作らず、未対応の量を 0 として返して確実性を装うこともありません。

bootstrap ensemble はデータを再抽出して完成した分類器を複数学習します。メンバー間の確率のばらつきと、個々の予測 label の不確実性は区別します。GP ensemble・異種 ensemble は確率空間で合成し、重みと class 順を保持します。共通の潜在関数を持たない異種 ensemble に `LatentStraddle` を適用しません。

## 較正・信頼性・意思決定

`TemperatureScalingCalibrator` は正の温度で log 確率を変換します。`CalibratedBinaryClassifier` は潜在 posterior を変えず、予測確率と確率 sample を較正します。そのため entropy・確率分散・BALD の入力は変わりますが、潜在境界を使う `LatentStraddle` は変わりません。

較正にはモデル学習から保留したデータを使います。NLL・Brier・ECE・MCE は診断であり獲得関数ではありません。不均衡・誤分類費用に応じた決定閾値、OOD 診断、conformal prediction set は、それぞれの意味と検証条件を[詳細契約](../../docs-internal/contracts/classification-models.md)で確認してください。高信頼の予測や空間外検知のスコアだけで正しさを保証しません。

## 対応範囲を確認する

multiclassおよびordinal classifierは実装済みです。通常のmulticlassはクラスごとの
潜在GPとSoftmax、ordinalは単一潜在GPと単調なOrdered-probit閾値を利用します。
Bernoulli、0.5 閾値、潜在 0 境界は binary 実装に閉じ込めます。Multi-Fidelity、高次元・robust の組合せ、batch・非同期・fantasization の可否は class 名から推測せず capability とテストを確認してください。

数理は[分類理論](../theory/24_classification.md)、実装との対応は[分類ガイド](../guide/24_classification.md)、選択基準は[分類 Active Learning](../theory/acquisition/13_classification_active_learning.md)から辿れます。API 形状・対応行列・未対応条件の正本は[内部契約](../../docs-internal/contracts/classification-models.md)です。

## 拡張二値分類モデル（Phase 2〜8）

| 入力・仮定 | モデル | 主な指定 |
| --- | --- | --- |
| 連続入力・異なる評価忠実度 | `MultiFidelityBinaryGPClassifier` | `data_fidelities` / `iteration_fidelity` |
| Mixed・異なる評価忠実度 | `MixedMultiFidelityBinaryGPClassifier` | `cat_dims`, fidelity列 |
| 周期性・複数周波数成分 | `SpectralMixtureBinaryGPClassifier` | `num_mixtures` |
| Mixed・周期性 | `MixedSpectralMixtureBinaryGPClassifier` | `cat_dims`, `num_mixtures` |
| Context別の偏差 | `ContextualBinaryGPClassifier` | `context_feature` |
| Group / Subgroup階層 | `HierarchicalBinaryGPClassifier` | `group_feature`, `subgroup_feature` |
| 無限幅ReLU NNGPカーネル | `InfiniteWidthBNNBinaryGPClassifier` | `depth`, `weight_variance` |
| Mixed・無限幅ReLU NNGP | `MixedInfiniteWidthBNNBinaryGPClassifier` | `cat_dims`, `depth` |

すべてBernoulli likelihoodを持つ変分GP分類器です。Fidelity列は設計変数とは別に扱い、
Context / Group / Subgroupには非負整数IDを指定します。Subgroup IDは異なるGroup間で再利用できません。
NNGPは**無限幅ニューラルネットワークに対応する解析的カーネル**であり、有限幅BNNの重みを学習しません。

### q-batch Active Learning

`qPredictiveEntropy`、`qMarginUncertainty`、`qProbabilityVariance`、
`qLatentStraddle`は候補ごとの**周辺スコアの平均**です。
`qBALD`は潜在関数の同時サンプルからラベルベクトルを列挙し、
候補間の冗長性を考慮します。binaryは`1 <= q <= 8`、
多クラス・順序分類では`C**q <= 256`の列挙上限があります。
単一点用の`PredictiveEntropy`等とは区別してください。

学習から候補提案までの例は
[分類拡張Notebook](../../examples/notebooks/27_binary_classification_extensions.ipynb)を参照してください。
比較コードは[ベンチマーク](../../benchmarks/binary_classification_extensions.py)にあります。

モデル別の入力・q-batch対応・検証範囲は[二値分類拡張の対応表](binary_classification_extension_matrix.md)を参照してください。


# Ordinal GP：Active Learningと候補生成

数理の導出は[分類GP理論](../theory/24_classification.md)、
実装との対応は[分類Guide](../guide/24_classification.md)、
学習から2ラウンドの候補提案・Mixed探索までの実行例は
[52番 Ordinal GP Notebook](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)を参照してください。

Ordinal labels (0, 1, ..., C-1) have a **known order**. Unlike ordinary
multiclass classification, the ordered-probit model uses a single latent
Gaussian process and increasing thresholds. `OrdinalSingleTaskGPClassifier`
and `MixedOrdinalSingleTaskGPClassifier` expose the existing classification
prediction contract, so existing acquisitions can be reused without new
ordinal-specific acquisition implementations.

| Acquisition | Standard | Mixed | Batch behavior |
| --- | --- | --- | --- |
| PredictiveEntropy | Yes | Yes | q=1 only |
| MarginUncertainty | Yes | Yes | q=1 only |
| ProbabilityVariance | Yes | Yes | q=1 only |
| BALD | Yes | Yes | q=1 only |
| qPredictiveEntropy | Yes | Yes | Mean marginal entropy |
| qMarginUncertainty | Yes | Yes | Mean marginal top-two uncertainty |
| qProbabilityVariance | Yes | Yes | Mean marginal epistemic variance |
| qBALD | Yes | Yes | Joint information via label enumeration; C**q <= 256 |
| LatentStraddle / qLatentStraddle | No | No | Binary zero-boundary only |

The three marginal q-acquisitions **do not** model redundancy between
candidates. qBALD evaluates joint label information using correlated latent
posterior samples, but its cost grows exponentially with the number of
classes and candidates. For three classes, q=5 is allowed (243 label
vectors), whereas q=6 is rejected (729 vectors).

## Minimal continuous-input example

```python
import torch

from robotorchan.acquisition.active_learning.classification import qPredictiveEntropy
from robotorchan.models.classification import OrdinalSingleTaskGPClassifier
from robotorchan.optim.strategies.sobol import SobolSearchStrategy

X = torch.linspace(0, 1, 15, dtype=torch.double).unsqueeze(-1)
y = torch.bucketize(X[:, 0], torch.tensor([0.35, 0.7], dtype=torch.double))
model = OrdinalSingleTaskGPClassifier(X, y, num_classes=3)
model.fit(epochs=20, lr=0.01)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
result = SobolSearchStrategy(bounds, num_samples=128, seed=11).optimize(
    qPredictiveEntropy(model), q=2
)
print(result.candidates)
```

For Mixed inputs, specify `cat_dims` when constructing the model **and**
provide a corresponding `MixedVariableSpace` to the sampling optimizer.
The model's categorical dimensions do not automatically configure candidate
generation. `CandidateConstraints` constrain **inputs**, not ordinal
class probabilities.

Phase 3 tests cover continuous Sobol candidate generation, Mixed Sobol
sampling with a linear input constraint, q=1 and q-batch acquisitions,
and the qBALD enumeration limit. They do not establish compatibility with
every optimization backend or guarantee better selection performance.

## Ordinal GP：Active Learningのラベル効率比較

順序回帰の基本動作（標準／Mixed、確率予測、q-batch候補提案）は
[31番Notebook](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)
で確認できます。次に、**同じラベル予算で追加ラベルをどう選ぶか**を
比較したい場合は独立したベンチマークを使います。

[ordinal_active_learning.py](../../examples/benchmarks/ordinal_active_learning.py)
は3段階の合成順序ラベルを使い、次の手法を同条件で比較します。

| 比較する選択手法 | 実際に使う方法 | 何を優先するか |
| --- | --- | --- |
| `random` | 未観測プールから一様ランダムに選ぶ | 比較用の基準 |
| `entropy` | 既存`PredictiveEntropy`を最大化 | 予測ラベルの曖昧さ |
| `bald` | 既存`BALD`を最大化 | 潜在モデルに関する情報量 |

それぞれ同じseed、初期ラベル付き15点、未観測候補75点、独立保留30点、
追加ラベル予算・学習条件から開始します。各ラウンドでは未観測候補から1点を選び、
**選択した後**にそのラベルを学習データへ追加してモデルを学習し直します。
連続入力と、カテゴリ列を持つMixed入力の双方を評価します。

```bash
python examples/benchmarks/ordinal_active_learning.py --seeds 3 --budget 5 --epochs 8
```

出力は`ordinal_al_results.csv`です。seed・Mixedフラグ・方法・追加ラベル数ごとに、
Accuracy（高いほどよい）、NLL（低いほどよい）、
Brier score（低いほどよい）、
ordinal MAE（クラス番号の絶対誤差、低いほどよい）を記録します。
MAEは**クラス番号の間隔を同じ尺度とみなせる場合のみ**物理的に解釈してください。

このベンチマークは**合成データの実行・比較可能性を確認するもの**で、
Ordinal GPや特定の獲得関数が一般に優れている証拠ではありません。
単一点`q=1`の逐次ラベル取得を比較し、`qBALD`のjoint batch性能を
比較するものではありません。実務での性能評価には複数生成過程、クラス不均衡、
ラベル誤差、評価のばらつき、計算時間・ラベルコストの検討が必要です。
