# Active Learning：数式とローカル実装の接続

## 何を最大化するか

Active Learning (AL) は目的関数の最大値を探すBOとは区別し、未観測の応答・分類境界について情報を得るために候補を選ぶ。
[理論9章](../../theory/acquisition/09_active_learning.md)を数理的正本とする。

回帰ではposteriorの平均 `mu(x)` と分散 `v(x)` を取り、`qPosteriorVariance(x)=v(x)`、
`qPosteriorStd(x)=sqrt(max(v(x),0))` を評価する。この二つは非負分散上で順位は単調に対応するが、utilityの**尺度**は異なる。
q>1では各候補の周辺分散を平均して単一のbatch scoreを返す。候補間の相関や重複は考慮しない。

分類では観測ラベル確率 `p(y|x,D)` と潜在GPの確率変数を区別する。ラベルのpredictive entropy `H(Y|x,D)=-sum_c p_c log p_c` と、
モデルの`mutual_information`で評価するBALDは同じ計算ではない。

## 実装への対応

- **`qPosteriorVariance`**
  数学的score / 参照：`v(x)`
  実装所有権と使用するモデルAPI：`active_learning/variance.py::qPosteriorVariance.forward`でImplemented。
  momentsは`model.posterior(X)`へDelegated

- **`qPosteriorStd`**
  数学的score / 参照：`sqrt(max(v(x),0))`
  実装所有権と使用するモデルAPI：`variance.py::qPosteriorStd.forward`がvarianceの返値にclampとsqrtを適用（Extended）

- **`PredictiveEntropy`**
  数学的score / 参照：`-sum p_c log p_c`
  実装所有権と使用するモデルAPI：`classification.py::PredictiveEntropy.forward`が`predict_proba`を呼びentropyを計算

- **`MarginUncertainty`**
  数学的score / 参照：`1-(p_(1)-p_(2))`
  実装所有権と使用するモデルAPI：`classification.py::MarginUncertainty.forward`が上位2クラスの確率差を使用

- **`ProbabilityVariance`**
  数学的score / 参照：class-probability varianceのclass平均
  実装所有権と使用するモデルAPI：`classification.py`からモデルの`probability_variance(X,num_samples)`へDelegated

- **`BALD`**
  数学的score / 参照：predictive entropyと期待conditional entropyの差に対応するmutual information
  実装所有権と使用するモデルAPI：`classification.py`から`mutual_information(X,num_samples)`へDelegated。
  posterior latent varianceではない

- **`LatentStraddle`**
  数学的score / 参照：`beta * sigma_latent - abs(mu_latent)`
  実装所有権と使用するモデルAPI：`classification.py`で`latent_posterior(X)`に基づき境界を評価

`qStraddle`、`qRandomizedStraddle`、`qBoundaryVariance`は[Level Set章](10_level_set.md)へ、
`ExpectedPredictiveInformationGain`は[情報理論章](05_information_theoretic.md)へ分ける。
上表のALクラスはBoTorch-native EI等とは異なるローカル実装である。

## 入出力shapeと拒否条件

`X`の形は通常`batch_shape × 1 × d`であり、acquisitionの返値はcandidate batchに対するscalar scoreとなる。
`qPosteriorVariance`と`qPosteriorStd`はq>1に対応し、q軸の平均分散を使う。分類ALは引き続き`max_q=1`。

`qPosteriorVariance`はmulti-output posteriorで`output_index`が無ければ明示的に拒否する。指定時はその出力次元だけを抽出する。
またstructured-output posteriorやensemble posteriorも明示的に拒否する（decoratorがあってもguardは優先される）。

分類の`PredictiveEntropy`は`predict_proba`を要求する。確率varianceとBALDは`num_samples>=2`を要求し、
対応するモデルAPIが無ければconstruct時にTypeErrorとなる。単にGPのlatent `posterior`があれば分類ALを計算できる、という契約ではない。

## 具体的な使い分け

回帰の関数値をできるだけ広く学習する単一点候補には`qPosteriorVariance(model)`が基準となる。未知境界`f(x)=t`の周辺へ集中したい場合は
[Level Set章](10_level_set.md)のqStraddle系を選ぶ。分類ラベルの不確実性を狙う場合は`PredictiveEntropy(classifier)`、モデルパラメータ不
確実性によるラベル情報を狙う場合は`BALD(classifier)`とし、予測確率の意味論を確認する。

この使い分けは推薦基準であり、特定のモデル・optimizer組合せの実行可能性を保証しない。AL optimizerはcandidate constraints、
continuous/discrete、batch shapeに別途対応している必要がある。

### BoTorch / robotorchanとの対応

BoTorch は qNegIntegratedPosteriorVariance など experimental-design 向け acquisition を提供しています。

robotorchan は regression AL 向けに qPosteriorVariance、qPosteriorStd、
ExpectedPredictiveInformationGainを提供し、BoTorch native で十分な手法は再実装しません。

現在の robotorchan-specific acquisition には q、structured output、ensemble posteriorなどに明示的な適用範囲があります。
Theory 上の一般形と現在の実装範囲を混同せず、詳細は次を参照してください。

- [Regression Active Learning guide](../../optimization/regression_active_learning.md)
- [EPIG guide](../../optimization/epig.md)
- [統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)

## 補足となる実装上の設計

#### Posterior Variance Sampling：実装上の補足

posterior variance が大きい点は、現在の surrogate がその latent function value を十分に知らない領域です。

#### Negative Integrated Posterior Variance：実装上の補足

BoTorch の qNegIntegratedPosteriorVariance（qNIPV）は、
candidate batch を観測した後の integrated posterior variance を基準に candidate set を評価する考え方です。

#### RegressionとClassification：実装上の補足

本章は robotorchan の現在の regression acquisition theory を中心に扱い、
classification-specific acquisitionを同じ数式へ無理に統合しません。

## 数理モデルとクラス構造

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

qPosteriorVariance はその候補自身の uncertainty を評価します。

- BOとALはposteriorを共有できてもdecision purposeが違う
- qPosteriorVariance / Stdはq>1でもcandidate自身のmarginal uncertaintyを平均する
- candidate setとtarget / integration setを分ける
- qNIPVはpointwise variance上位q点ではなくtarget-region uncertainty reductionを評価する
- predictive varianceが高いこととreducible epistemic uncertaintyが大きいことは同義ではない
- EPIGはcandidate自身のentropyではなくtarget predictionsとのinformation gainを見る
- MESとEPIGはmutual informationを使ってもinformation targetが違う
- current EPIGはq=1 single-output joint-Gaussian finite-target-set contract
- current qPosteriorVariance / Std runtimeはensemble posteriorをrejectする
- classification-specific ALはregression uncertainty acquisitionへ無理に統合しない

```text
qPosteriorVariance / qPosteriorStd
    candidate 自身の uncertainty

Integrated variance / qNIPV
    target region 全体の uncertainty reduction

EPIG
    target prediction についての information gain
```

## marginal uncertaintyを抽出するローカル実装

[variance.py](../../../src/robotorchan/acquisition/active_learning/variance.py)の
`qPosteriorVariance`はBoTorchの`AcquisitionFunction`を継承します。`forward()`はposteriorの分散を取得し、
出力軸を選択し、q軸の平均を返します。`qPosteriorStd`はそれを継承し、非負へclampしてから平方根を取ります。posteriorの構成はモデルへ委譲しますが、
uncertaintyをscoreへ変換する処理は独自実装です。

structured outputやensembleには一意なreductionを定義していないため、型・shapeの条件で拒否します。
これは将来の応答をfantasizeしてtarget全体を評価するBoTorchのqNIPVとも、候補とtargetの情報量を計算するEPIGとも異なる計算です。
EPIGの内部は[情報理論章](05_information_theoretic.md)へ分けています。

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/28_regression_active_learning.ipynb)を参照してください
。


## q-batchの評価方式

`qPosteriorVariance`は候補集合の周辺分散の平均を最大化する。
`qPosteriorStd`はその平均分散の平方根を返す。両者ともq=1で従来の値と一致する。
候補間の相関を直接評価しないため、候補の重複を避ける保証はない。

`qJointPosteriorEntropy`は単一出力のGaussian posteriorに対し、
候補集合の共分散行列を用いてjoint differential entropyを評価する。
これは候補自身の不確実性に基づく基準であり、外部のtarget分布に対するEPIGとは異なる。
