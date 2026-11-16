# 分類Active Learning：理論と実装の対応

[分類Active Learning理論](../../theory/acquisition/13_classification_active_learning.md)
の不確実性の定義に対し、ここでは実装で参照する予測API、
獲得関数、q-batchの評価方式、確率校正との関係を整理します。
数式の導出はTheory、実行コードと可視化は
[39番Notebook](../../../examples/notebooks/39_classification_active_learning.ipynb)
と[47番Notebook](../../../examples/notebooks/47_binary_classification_extensions.ipynb)
を参照してください。

## 単一点の分類獲得関数

| 公開獲得関数 | モデル側の入力契約 | 評価する量 |
| --- | --- | --- |
| PredictiveEntropy | predict_proba | 予測クラス確率のEntropy |
| MarginUncertainty | predict_proba | 上位2クラスの確率差の小ささ |
| ProbabilityVariance | probability_variance | クラス確率の事後分散のクラス平均 |
| BALD | mutual_information | ラベルと潜在変数の相互情報量 |
| LatentStraddle | latent_posterior | 潜在平均と分散に基づくゼロ境界探索 |

これらの単一点版は候補のq次元が1の入力を要求します。
候補のTensor形状はBoTorchに合わせて通常 batch_shape × q × d です。
必要なモデル側メソッドが存在しない場合は、対応済みとみなさず
公開APIの契約を確認してください。

二値Probit GPの `BinarySingleTaskGPClassifier`（Mixed等の標準派生モデルを含む）では、
`predict_proba` は潜在ガウス分布を解析的に積分します。
さらに `probability_variance` と `mutual_information` は、
デフォルトでGPyTorchの128点Gauss–Hermite求積法を利用します。
確率分散は数値安定性を考慮した `E[(p_f - E[p_f])^2]`、BALDは
`H(E[p_f]) - E[H(p_f)]` で計算します。これは観測ラベルの
Bernoulli分散 `E[p_f] * (1 - E[p_f])` とは異なる量です。
確率分散・BALDの計算は決定論的ですが、数値求積には近似誤差があります。

```python
variance = model.probability_variance(test_X)  # [..., 2] (q=1で末尾の1次元を省略)
bald = model.mutual_information(test_X)  # [...], q>1では[..., q]

# 比較検証などで従来のMC推定を明示的に利用する場合
variance_mc = model.probability_variance(test_X, method="mc", num_samples=2048)
bald_mc = model.mutual_information(test_X, method="mc", num_samples=2048)
```

既存の `num_samples` 引数は互換性のために保持しますが、
二値Probit GPのデフォルト（`method="quadrature"`）では使用しません。
`method="mc"` の場合にのみサンプル数を指定します。
この改善は二値の標準GP尤度系を対象とし、マルチクラス、
校正後モデル、Ensembleや非GP分類器の推定方法を変更するものではありません。

LatentStraddleは二値分類専用です。
同じ確率でも潜在分布とラベル分布は異なるため、
確率だけを返すNon-GPモデルには潜在境界の基準を
無条件に適用できません。

## 分類確率Posteriorと獲得関数の数値整合

分類GPの `posterior(X)` は潜在関数の事後分布です。
`probability_posterior(X)` は確率ベクトルの事後分布であり、
`mean` と `variance` を提供します。
`credible_interval(0.95)` は**クラス確率そのものの周辺信用区間**です。
観測ラベルの予測集合でも、全クラスの同時信用領域でもありません。

| モデル | 確率の平均 | 確率の事後分散 | 区間の計算 |
| --- | --- | --- | --- |
| 二値Probit GP | 解析的 | Gauss–Hermite求積 | 潜在Gaussian分位点を変換 |
| Ordinal GP | 解析的 | Gauss–Hermite求積 | MCによる周辺分位点 |
| マルチクラスGP | 同じQMCサンプルから平均 | 同じQMCサンプルから分散 | MCによる周辺分位点 |
| Ensemble | メンバーの加重平均 | メンバー間の加重分散 | 加重経験分位点 |

**不確実性の対象を混同しないことが重要です。**
`ProbabilityVariance` はクラスごとの
`probability_posterior(X).variance` を平均した量に対応します。
これは観測ラベルのBernoulli/Categorical分散ではありません。
`PredictiveEntropy` は平均予測確率のエントロピー、
`BALD` は予測エントロピーから平均条件付きエントロピーを引いた量です。
`credible_interval` の上下限を獲得関数スコアとしてそのまま代用しません。

二値Probitでは、確率分散とBALDは決定論的なガウス求積を使います。
Ordinalとマルチクラスでは、既存の獲得関数は
潜在確率サンプルに基づくMC推定を利用します。
マルチクラスの確率Posterior平均・分散は別途seed指定可能なQMC推定なので、
**独立な有限サンプルでは数値が完全一致する必要はありません。**
一致すべき数式を、参照サンプルや数値積分と比較してください。
Ensembleはメンバー間の離散分布に基づくため、
GP個体内部の不確実性が含まれるとは限りません。

```python
# X の形状は [batch, q, d]。q=1の場合を想定。
posterior = model.probability_posterior(X)
probability_mean = posterior.mean
probability_variance = posterior.variance
lower, upper = posterior.credible_interval(0.95)
# 2クラスProbitならProbabilityVarianceは分散のクラス平均。
```

実行例は
[39番（二値Probit）](../../../examples/notebooks/39_classification_active_learning.ipynb)、
[41番（Ensemble）](../../../examples/notebooks/41_classification_probability_layers.ipynb)、
[51番（マルチクラス）](../../../examples/notebooks/51_multiclass_classification_active_learning.ipynb)、
[52番（Ordinal）](../../../examples/notebooks/52_ordinal_classification_active_learning.ipynb)
にあります。

## q-batchの実装方式

| 公開獲得関数 | バッチの評価方式 | 情報重複を直接考慮するか |
| --- | --- | --- |
| qPredictiveEntropy | 各候補の周辺Entropyの平均 | いいえ |
| qMarginUncertainty | 各候補の周辺Marginスコアの平均 | いいえ |
| qProbabilityVariance | 各候補の確率分散スコアの平均 | いいえ |
| qLatentStraddle | 各候補の潜在Straddleスコアの平均 | いいえ |
| qBALD | 同時ラベル分布と潜在値の相互情報量 | はい、同時分布を通じて |

周辺集約型のq版は「q個の獲得値の平均」であり、
jointなラベルEntropyや情報利得ではありません。
同じような候補が並んでも、それだけで減点しません。

qBALDは、潜在事後から**候補間で整合した同時サンプル**を取得し、
二値ラベルの全組合せを使ってjoint相互情報量を評価します。
現行実装は二値分類、q=1〜8を対象とします。
候補の多様性を考慮できる一方、距離の下限は保証しません。
実際の候補最適化には獲得関数と最適化器の
q-batch対応を両方確認する必要があります。

## Ensembleと不確実性

Ensembleの予測確率はメンバー確率の重み付き平均です。
ClassificationEnsemblePosteriorでは、クラス確率の平均、
メンバー間の分散、予測Entropy、
メンバーごとのEntropy平均、BALD型の不一致を区別します。

異種モデルで共通の潜在関数が存在しない場合、
確率のEnsembleと潜在Straddleを同じ意味で扱いません。
メンバー分散が大きいことは
必ずしも校正されたBayesian事後分散を意味しません。

## Probability Calibrationとの合成

TemperatureScalingCalibratorは、クラス確率の対数を
正の温度でスケーリングし、正規化します。
CalibratedBinaryClassifierは予測確率と確率サンプルに
校正を適用しますが、潜在GPの事後分布は変更しません。

そのためEntropy、Margin、ProbabilityVariance、BALDは
校正の影響を受ける可能性があります。
一方、潜在posteriorへ委譲するLatentStraddleの
スコアは確率校正だけでは変更されません。

校正した確率を可行性に利用するときも、
ClassificationProbabilityOfFeasibilityが扱うのは
指定クラスの予測確率です。
CalibrationとConformal predictionは異なる目的であり、
NLL、Brier Score、ECEなどの診断を
獲得関数そのものと混同しないでください。

## 関連資料

- [分類モデルGuide](../24_classification.md)：潜在事後・クラス確率・構造モデル
- [モデル選択資料](../../models/classification.md)：分類モデルと対応範囲
- [二値分類拡張対応表](../../models/binary_classification_extension_matrix.md)：q-batchとモデル構成
- [39番Notebook](../../../examples/notebooks/39_classification_active_learning.ipynb)：EntropyとBALDの可視化
- [41番Notebook](../../../examples/notebooks/41_classification_probability_layers.ipynb)：Ensembleと確率校正
- [47番Notebook](../../../examples/notebooks/47_binary_classification_extensions.ipynb)：Mixedとq=2
- [分類契約](../../../docs-internal/contracts/classification-models.md)：出力形状と非対応条件

Theoryは数理、Guideは実装対応、Notebookは実行例という
役割を維持します。APIの完全な組合せ互換性は
各モデルのcapabilityとテストで確認してください。
