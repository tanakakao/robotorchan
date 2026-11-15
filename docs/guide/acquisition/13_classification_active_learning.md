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
`H(E[p_f]) - E[H(p_f)]` で計算し、観測ラベルのBernoulli分散
予測平均確率から計算するBernoulli分散とは区別します。
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
