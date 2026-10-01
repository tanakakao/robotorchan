# 4. Batch and Noisy Acquisition

## 4.1 最初に4つの軸を分ける

Batch / noisy BOでは、次を同じ意味で使わないことが重要です。

| 概念 | 問い |
| --- | --- |
| batch / q | 今回いくつの新しいcandidateを選ぶか |
| noisy | 観測値とlatent function valueを区別する必要があるか |
| pending / async | 結果待ちの実評価を考慮するか |
| fantasization | 未観測結果を仮想的に条件付けしたmodelを作るか |

これらは組み合わせられますが別の軸です。特にpending pointとfantasy pointを同一視しません。

## 4.2 q > 1 は pointwise ranking ではない

1回の実験サイクルで q 点を同時に評価する場合、候補集合

$$
X=(x_1,\ldots,x_q)
$$

全体に対して acquisition value を定義します。

重要なのは、

$$
\alpha_q(X)
\neq
\sum_{j=1}^{q}\alpha_1(x_j)
$$

が一般的だという点です。

候補点間にposterior dependenceがある場合、似た候補を複数選ぶ価値は重複し得ます。
q-acquisitionは候補集合のjoint posterior semanticsを保ってutilityを評価します。

Gaussian GPではcovariance matrixがこのdependenceを表しますが、一般のsampleable posteriorでは
必ずしも明示的なGaussian covariance matrixを持つとは限りません。

## 4.3 Joint posteriorはGaussian covarianceだけではない

候補集合に対する Gaussian posterior は概念的に

$$
f(X)\mid\mathcal D_n
\sim
\mathcal N(\mu(X),\Sigma(X))
$$

です。

# 4. Batch and Noisy Acquisition

## 4.4 qEI

最大化問題で q 点のうち最も良い結果による improvement を

$$
I(X)
=
\max\left(
\max_{j=1,\ldots,q} f(x_j)-f_{\mathrm{best}},
0
\right)
$$

とすれば、

$$
q\operatorname{EI}(X)
=
\mathbb E[I(X)]
$$

です。

一般に q > 1 の期待値は MC sampling によって評価されます。

この定義から、qEI は「EI が高い q 点」を独立に選ぶ手法ではないことが分かります。

## 4.5 Observation noise が変えるもの

観測モデルを

$$
y(x)=f(x)+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,\sigma_\epsilon^2)
$$

とします。

noise があると、観測値

$$
\max_i y_i
$$

が潜在関数 $`f`$ の最良値とは限りません。偶然大きな noise を含む観測を incumbent として固定すると、improvement criterion が歪む可能性があります。

したがって noisy BO では、

```text
observed best
```

と

```text
latent function の best
```

を区別する必要があります。

## 4.6 Noisy Expected Improvement

Noisy Expected Improvement（NEI）は、baseline points における未知の latent function values についても posterior
uncertainty を考慮します。

概念的には、baseline latent values を posterior から積分しながら、それぞれの plausible world で improvement を評価します。

qNEIはさらにcandidate batchのjoint posteriorを扱うため、

- baseline uncertainty
- candidate uncertainty
- candidate correlation
- observation noise

を同時に扱うことになります。

## 4.7 Baseline points

NEI 系では baseline set が重要です。baseline は「何と比較して improvement を測るか」を posterior 上で定義するために使われます。

baseline の選び方は単なる API detail ではなく、acquisition の計算量や意思決定対象に影響します。大規模 baseline では pruning
などの計算上の工夫が重要になる場合があります。

## 4.8 Pending points と非同期BO

並列実験では、すでに評価を開始したが結果が返っていない点

$$
X_{\mathrm{pending}}
$$

が存在します。

pending points を無視すると、同じ領域へ重複して新しい候補を投入する可能性があります。

非同期BOでは、completed observationsでmodelをfitし、結果待ちの実評価を `X_pending` として
対応acquisitionへ渡す構成を取れます。robotorchanは独自のasync scheduler stateを持ちません。

`X_pending` は実際に評価中のcandidateというcontextです。acquisition内部のlookahead fantasy
variablesとは別物です。pending対応の有無もmodel固有flagではなく、主としてacquisition contractです。

## 4.9 Sequential greedy batch と joint optimization

batch candidates の生成には、q 点を一度に joint optimization する方法だけでなく、1点ずつ条件付きで追加する sequential greedy
strategy もあります。

```text
candidate 1 を選ぶ
    ↓
candidate 1 を pending / conditioned として扱う
    ↓
candidate 2 を選ぶ
    ↓
...
```

joint optimizationとgreedy constructionは同じアルゴリズムではありません。
sequential optimizationで先に選んだcandidateをpending-like contextとして利用する場合も、
それは外部で既に実行中の `X_pending` と同一の状態管理ではありません。

## 4.10 Fantasizationをpendingと分ける

fantasizationは、未観測点のoutcomeをsampleし、その仮想観測でconditionしたmodelを作る操作です。
これはreal pending evaluationを表す `X_pending` そのものではありません。

Knowledge Gradientやmulti-step lookaheadではacquisition内部の将来意思決定を表すために
fantasy modelが本質的です。一方、すべてのq-acquisitionやasync workflowがmodel-level
`fantasize()` を必要とするわけではありません。

したがって次を分けます。

~~~text
q > 1              candidate setのdecision dimension
X_pending          unresolved real evaluations
fantasy batch      hypothetical outcomesによるmodel batch dimension
one-shot rows      acquisition内部のauxiliary decision variables
~~~

これらのshapeを相互に推測しません。

## 4.11 Monte Carlo と sample consistency

q-acquisition の多くは MC expectation を最適化します。

acquisition optimization の途中で sampling noise が過度に変化すると、objective surface 自体が揺れて gradient
optimization が難しくなります。QMC sampling や共通 base samples は、MC estimator の variance と optimization 
stability に関係します。

これは acquisition の理論的 utility と、その数値最適化をつなぐ重要な実装論点です。

## 4.12 Log formulation

qEI / qNEI の値が非常に小さい領域では、数値 underflow や gradient degradation が問題になります。

qLogEI / qLogNEI は improvement-based criterion の数値安定性を改善する formulation です。log variant
を「異なる探索目的」と考えるのではなく、同じ意思決定基準を安定して最適化するための重要な数値的設計として理解します。

## 4.13 Batch size と計算量

q を増やすと候補集合の joint dimension が増え、MC sampling と acquisition optimization の計算負荷も増えます。

したがって大きな q では、

- joint optimization の難易度
- posterior sampling cost
- restart / raw sample 数
- sequential generation の利用
- pending points の数

などを含めて設計する必要があります。

## 4.14 BoTorch / robotorchan との対応

BoTorch は qLogEI、qLogNEI などの batch / noisy acquisition と pending-point semantics
を提供します。robotorchan は標準手法を再実装せず、BoTorch native path を利用します。

具体的な API は [Standard acquisition](../../optimization/standard_acquisition.md)、現在の対応状況は
[Acquisition integration status](../../optimization/acquisition-integration.md) を参照してください。

### 4.12.1 qLogEI / qLogNEI と posterior requirement

現在の registry では `qLogExpectedImprovement` と
`qLogNoisyExpectedImprovement` の posterior requirement を
`POSTERIOR_SAMPLES` として表現しています。

したがって「Gaussian posteriorだから利用可能」とだけ判断せず、
model registry の `supports_posterior_samples` と組み合わせて compatibility を判定します。

またpending / asynchronous semanticsとcandidate optimizerのfeasibilityは別責務です。
model-level fantasization supportも別契約であり、`X_pending` 対応から自動的には推論しません。

実行可能な互換性matrixは
[Batch / async / fantasization compatibility](../../optimization/batch-async-fantasization.md)
を正とします。

## 4.15 この章で覚えておくこと

- q、noise、pending、fantasizationは別の軸
- q-acquisitionはpointwise score上位q点ではなくjoint utilityを評価する
- joint semanticsはGaussian covariance matrixに限定されない
- NEIではobserved bestとlatent baseline uncertaintyを区別する
- `X_pending` は結果待ちの実評価でありlookahead fantasy variableではない
- async BOのためにrobotorchan独自scheduler stateは導入しない
- model `fantasize()` はすべてのbatch / async acquisitionに必須ではない
- candidate q、fantasy batch、one-shot auxiliary rowsを同じshape axisだと考えない
- qLogEI / qLogNEIはBoTorch-native pathを利用する

## 4.16 まとめ

Batch / noisy BO では、単一点の acquisition intuition だけでは不十分です。

```text
q > 1
    joint posterior と候補間 correlation

noise
    observed value と latent function value の区別

pending
    実行中評価を考慮した非同期意思決定
```

を分けて理解する必要があります。

次の発展として、情報量そのものを価値とする [Information-theoretic acquisition](05_information_theoretic.md)
と、将来の意思決定価値を見る [Lookahead acquisition](06_lookahead.md) があります。

## References

1. Ginsbourger, D., Le Riche, R., and Carraro, L. (2010). Kriging Is Well-Suited to Parallelize
   Optimization. In *Computational Intelligence in Expensive Optimization Problems*.
2. Letham, B. et al. (2019). Constrained Bayesian Optimization with Noisy Experiments.
   *Bayesian Analysis*, 14(2), 495-519.
3. Balandat, M. et al. (2020). BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.
   *Advances in Neural Information Processing Systems 33*.
