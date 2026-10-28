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

## 予測・学習の使い分け

`posterior(X)` は潜在関数の分布であり、class probability ではありません。確率には `predict_proba`、class の決定には `predict_class`、観測 label の分布には `predictive_distribution` を使います。`sample_latent` と `sample_class_probabilities` も別の量です。label は 0/1 を渡し、文字列や任意整数の暗黙変換は行いません。

標準 binary GP は Bernoulli 尤度の変分モデルです。`make_mll()` は `VariationalELBO` を構成します。Exact Gaussian GP と同じ学習契約と仮定しないでください。元データは raw training-data 契約に従い保持します。

## Robust モデル

誤 label・汚染・反復 label・入力依存の信頼性は観測過程として表します。連続残差用の Student-t を binary label に機械的に移植しません。非定常性は潜在共分散を変える別の機構です。likelihood と互換潜在モデルの合成を優先し、全 family の直積 class を作りません。具体的な registry ID と組合せ条件は[詳細契約](../../docs-internal/contracts/classification-models.md)にまとめています。

## Active Learning と BO 制約

`PredictiveEntropy`・`MarginUncertainty` は予測 class 確率、`ProbabilityVariance`・`BALD` は確率 sample や ensemble の情報、`LatentStraddle` は二値の潜在境界を使います。現在はいずれも q=1 です。潜在 posterior が Gaussian というだけで回帰用獲得関数に対応するとはみなしません。

`ClassificationProbabilityOfFeasibility` は指定 class の実行可能確率を取得し、`FeasibilityWeightedAcquisition` は目的獲得値に重みを掛けます。q-batch の既定 `product` は点間独立の近似、`minimum` は保守的なスコアで、joint 分類 posterior を作るものではありません。既知の入力に対する `CandidateConstraints` とは別の層です。

`RobustProbabilityTransform` は scenario ごとの class 確率を expectation・worst-case・VaR・CVaR で集約します。class ごとの下側 tail は異なる scenario に由来し得るため、集約値の和が 1 になるとは限りません。新たな確率分布として解釈せず、特定 class が必要なら `robust_class_probability` を使います。

## Non-GP と ensemble

Random Forest・Extra Trees・Gradient Boosting・Histogram Gradient Boosting の adapter は予測確率を提供しますが、GP の潜在 posterior を持つとはみなしません。単一推定器の確率だけから epistemic 分散・BALD を作らず、未対応の量を 0 として返して確実性を装うこともありません。

bootstrap ensemble はデータを再抽出して完成した分類器を複数学習します。メンバー間の確率のばらつきと、個々の予測 label の不確実性は区別します。GP ensemble・異種 ensemble は確率空間で合成し、重みと class 順を保持します。共通の潜在関数を持たない異種 ensemble に `LatentStraddle` を適用しません。

## 較正・信頼性・意思決定

`TemperatureScalingCalibrator` は正の温度で log 確率を変換します。`CalibratedBinaryClassifier` は潜在 posterior を変えず、予測確率と確率 sample を較正します。そのため entropy・確率分散・BALD の入力は変わりますが、潜在境界を使う `LatentStraddle` は変わりません。

較正にはモデル学習から保留したデータを使います。NLL・Brier・ECE・MCE は診断であり獲得関数ではありません。不均衡・誤分類費用に応じた決定閾値、OOD 診断、conformal prediction set は、それぞれの意味と検証条件を[詳細契約](../../docs-internal/contracts/classification-models.md)で確認してください。高信頼の予測や空間外検知のスコアだけで正しさを保証しません。

## 対応範囲を確認する

multiclass 共通の metadata・確率ベクトル契約は準備されていますが、公開 multiclass classifier は未実装です。Bernoulli、0.5 閾値、潜在 0 境界は binary 実装に閉じ込めます。Multi-Fidelity、高次元・robust の組合せ、batch・非同期・fantasization の可否は class 名から推測せず capability とテストを確認してください。

数理は[分類理論](../theory/24_classification.md)、実装との対応は[分類ガイド](../guide/24_classification.md)、選択基準は[分類 Active Learning](../theory/acquisition/classification-active-learning.md)から辿れます。API 形状・対応行列・未対応条件の正本は[内部契約](../../docs-internal/contracts/classification-models.md)です。
