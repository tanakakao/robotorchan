# Active Learning：数式とローカル実装の接続

## 1. 何を最大化するか

Active Learning (AL) は目的関数の最大値を探すBOとは区別し、未観測の応答・分類境界について情報を得るために候補を選ぶ。[理論9章](../../theory/acquisition/09_active_learning.md)を数理的正本とする。

回帰ではposteriorの平均 `mu(x)` と分散 `v(x)` を取り、`PosteriorVariance(x)=v(x)`、`PosteriorStd(x)=sqrt(max(v(x),0))` を評価する。この二つは非負分散上で順位は単調に対応するが、utilityの**尺度**は異なる。これらはq=1の周辺情報量であり、q個を同時に評価するjoint acquisitionではない。

分類では観測ラベル確率 `p(y|x,D)` と潜在GPの確率変数を区別する。ラベルのpredictive entropy `H(Y|x,D)=-sum_c p_c log p_c` と、モデルの`mutual_information`で評価するBALDは同じ計算ではない。

## 2. 実装への対応

| 公開Acquisition | 数学的score / 参照 | 実装所有権と使用するモデルAPI |
| --- | --- | --- |
| `PosteriorVariance` | `v(x)` | `active_learning/variance.py::PosteriorVariance.forward`でImplemented。momentsは`model.posterior(X)`へDelegated |
| `PosteriorStd` | `sqrt(max(v(x),0))` | `variance.py::PosteriorStd.forward`がvarianceの返値にclampとsqrtを適用（Extended） |
| `PredictiveEntropy` | `-sum p_c log p_c` | `classification.py::PredictiveEntropy.forward`が`predict_proba`を呼びentropyを計算 |
| `MarginUncertainty` | `1-(p_(1)-p_(2))` | `classification.py::MarginUncertainty.forward`が上位2クラスの確率差を使用 |
| `ProbabilityVariance` | class-probability varianceのclass平均 | `classification.py`からモデルの`probability_variance(X,num_samples)`へDelegated |
| `BALD` | predictive entropyと期待conditional entropyの差に対応するmutual information | `classification.py`から`mutual_information(X,num_samples)`へDelegated。posterior latent varianceではない |
| `LatentStraddle` | `beta * sigma_latent - abs(mu_latent)` | `classification.py`で`latent_posterior(X)`に基づき境界を評価 |

`Straddle`、`RandomizedStraddle`、`BoundaryVariance`は[Level Set章](10_level_set.md)へ、`ExpectedPredictiveInformationGain`は[情報理論章](05_information_theoretic.md)へ分ける。上表のALクラスはBoTorch-native EI等とは異なるローカル実装である。実装の全監査記録は[Acquisition Mapping](../../../.github/project-records/theory-guide/mapping/acquisition.md)のPhase 5 regression/classification AL区間に残す。

## 3. 入出力shapeと拒否条件

`X`の形は通常`batch_shape × 1 × d`であり、acquisitionの返値はcandidate batchに対するscalar scoreとなる。対象のregression/classification AL registryはいずれも`max_q=1`。よって`q=2`のjoint最適化を保証しない。

`PosteriorVariance`はmulti-output posteriorで`output_index`が無ければ明示的に拒否する。指定時はその出力次元だけを抽出する。またstructured-output posteriorやensemble posteriorも明示的に拒否する（decoratorがあってもguardは優先される）。

分類の`PredictiveEntropy`は`predict_proba`を要求する。確率varianceとBALDは`num_samples>=2`を要求し、対応するモデルAPIが無ければconstruct時にTypeErrorとなる。単にGPのlatent `posterior`があれば分類ALを計算できる、という契約ではない。

## 4. 具体的な使い分け

回帰の関数値をできるだけ広く学習する単一点候補には`PosteriorVariance(model)`が基準となる。未知境界`f(x)=t`の周辺へ集中したい場合は[Level Set章](10_level_set.md)のStraddle系を選ぶ。分類ラベルの不確実性を狙う場合は`PredictiveEntropy(classifier)`、モデルパラメータ不確実性によるラベル情報を狙う場合は`BALD(classifier)`とし、予測確率の意味論を確認する。

この使い分けは推薦基準であり、特定のモデル・optimizer組合せの実行可能性を保証しない。AL optimizerはcandidate constraints、continuous/discrete、batch shapeに別途対応している必要がある。

## 5. 数学的検証と未検証の境界

[検証索引](../../../docs-internal/verification/theory-guide/acquisition.md)で参照する`tests/acquisition/test_phase8_regression_al_formula_reference.py`は固定float64のposterior moment scoreを独立数式と比較する。分類entropyとlatent boundaryには`test_phase8_classification_al_reference.py`、BALDの固定64サンプルmomentには`test_phase8_bald_probability_variance_reference.py`がある。

検証されたのは**指定したscore計算**であり、posteriorの較正、MC収束、joint q-batchのAL最適性は証明されない。旧Mappingのassertion・数値tol・未対応条件は移行台帳が検収されるまで保全する。

## 6. 回帰Active Learningの最小実行例

`PosteriorVariance`と`PosteriorStd`は、モデルが返す同じ潜在posteriorから
それぞれ分散と標準偏差を取り出す。以下は`q=1`の3候補を評価する例であり、
BOの目的値の改善量やbatch間のjoint情報量ではない。

```python
import torch
from robotorchan.acquisition import PosteriorStd, PosteriorVariance
from robotorchan.models import SingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
candidates = torch.tensor([[[0.15]], [[0.45]], [[0.85]]], dtype=torch.double)
variance_score = PosteriorVariance(model)(candidates)
std_score = PosteriorStd(model)(candidates)
assert variance_score.shape == (3,)
assert torch.isfinite(variance_score).all()
assert torch.isfinite(std_score).all()
```

標準偏差は分散の非負部分の平方根であり、両者は同じ順序で候補を評価するが
スコアの尺度は異なる。
分類のPredictiveEntropy・BALDは別の予測確率APIを必要とするため、
この回帰GPにそのまま適用しない。

## 実装補足：理論本文から分離した対応情報

### 9.20 BoTorch / robotorchanとの対応

BoTorch は qNegativeIntegratedPosteriorVariance など experimental-design 向け acquisition を提供しています。

robotorchan は regression AL 向けに PosteriorVariance、PosteriorStd、ExpectedPredictiveInformationGain
を提供し、BoTorch native で十分な手法は再実装しません。

現在の robotorchan-specific acquisition には q、structured output、ensemble posterior
などに明示的な適用範囲があります。Theory 上の一般形と現在の実装範囲を混同せず、詳細は次を参照してください。

- [Regression Active Learning guide](../../optimization/regression_active_learning.md)
- [EPIG guide](../../optimization/epig.md)
- [Acquisition integration status](../../../docs-internal/contracts/optimization/acquisition-integration.md)

## 理論章から分離した実装上の条件

#### 9.4 Posterior Variance Sampling：実装上の補足

posterior variance が大きい点は、現在の surrogate がその latent function value を十分に知らない領域です。

#### 9.10 Negative Integrated Posterior Variance：実装上の補足

BoTorch の qNegativeIntegratedPosteriorVariance（qNIPV）は、candidate batch を観測した後の integrated
posterior variance を基準に candidate set を評価する考え方です。

#### 9.16 RegressionとClassification：実装上の補足

本章は robotorchan の現在の regression acquisition theory を中心に扱い、classification-specific acquisition
を同じ数式へ無理に統合しません。

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

PosteriorVariance はその候補自身の uncertainty を評価します。

- BOとALはposteriorを共有できてもdecision purposeが違う
- PosteriorVariance / Stdはcandidate自身のmarginal uncertaintyを見るq=1 criterion
- candidate setとtarget / integration setを分ける
- qNIPVはpointwise variance上位q点ではなくtarget-region uncertainty reductionを評価する
- predictive varianceが高いこととreducible epistemic uncertaintyが大きいことは同義ではない
- EPIGはcandidate自身のentropyではなくtarget predictionsとのinformation gainを見る
- MESとEPIGはmutual informationを使ってもinformation targetが違う
- current EPIGはq=1 single-output joint-Gaussian finite-target-set contract
- current PosteriorVariance / Std runtimeはensemble posteriorをrejectする
- classification-specific ALはregression uncertainty acquisitionへ無理に統合しない

```text
PosteriorVariance / PosteriorStd
    candidate 自身の uncertainty

Integrated variance / qNIPV
    target region 全体の uncertainty reduction

EPIG
    target prediction についての information gain
```
