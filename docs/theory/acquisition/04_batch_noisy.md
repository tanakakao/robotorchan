# 4. Batch and Noisy Acquisition

## 4.1 q > 1 は pointwise ranking ではない

1回の実験サイクルで q 点を同時に評価する場合、候補集合

\[
X=(x_1,\ldots,x_q)
\]

全体に対して acquisition value を定義します。

重要なのは、

\[
\alpha_q(X)
\neq
\sum_{j=1}^{q}\alpha_1(x_j)
\]

が一般的だという点です。

GP posterior では候補点間に covariance があるため、似た候補を複数選ぶ情報価値は重複し得ます。q-acquisition は joint posterior を使って候補集合としての価値を評価します。

## 4.2 Joint posterior

候補集合に対する Gaussian posterior は概念的に

\[
f(X)\mid\mathcal D_n
\sim
\mathcal N(\mu(X),\Sigma(X))
\]

です。

\(\Sigma(X)\) の off-diagonal elements が候補間の posterior correlation を表します。

pointwise score だけを見ると、この相関構造を無視して近接した候補ばかり選ぶ可能性があります。batch acquisition はこの joint distribution を直接扱います。

## 4.3 qEI

最大化問題で q 点のうち最も良い結果による improvement を

\[
I(X)
=
\max\left(
\max_{j=1,\ldots,q} f(x_j)-f_{\mathrm{best}},
0
\right)
\]

とすれば、

\[
q\operatorname{EI}(X)
=
\mathbb E[I(X)]
\]

です。

一般に q > 1 の期待値は MC sampling によって評価されます。

この定義から、qEI は「EI が高い q 点」を独立に選ぶ手法ではないことが分かります。

## 4.4 Observation noise が変えるもの

観測モデルを

\[
y(x)=f(x)+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,\sigma_\epsilon^2)
\]

とします。

noise があると、観測値

\[
\max_i y_i
\]

が潜在関数 \(f\) の最良値とは限りません。偶然大きな noise を含む観測を incumbent として固定すると、improvement criterion が歪む可能性があります。

したがって noisy BO では、

```text
observed best
```

と

```text
latent function の best
```

を区別する必要があります。

## 4.5 Noisy Expected Improvement

Noisy Expected Improvement（NEI）は、baseline points における未知の latent function values についても posterior uncertainty を考慮します。

概念的には、baseline latent values を posterior から積分しながら、それぞれの plausible world で improvement を評価します。

qNEI はさらに candidate batch の joint posterior を扱うため、

- baseline uncertainty
- candidate uncertainty
- candidate correlation
- observation noise

を同時に扱うことになります。

## 4.6 Baseline points

NEI 系では baseline set が重要です。baseline は「何と比較して improvement を測るか」を posterior 上で定義するために使われます。

baseline の選び方は単なる API detail ではなく、acquisition の計算量や意思決定対象に影響します。大規模 baseline では pruning などの計算上の工夫が重要になる場合があります。

## 4.7 Pending points と非同期 BO

並列実験では、すでに評価を開始したが結果が返っていない点

\[
X_{\mathrm{pending}}
\]

が存在します。

pending points を無視すると、同じ領域へ重複して新しい候補を投入する可能性があります。

非同期 BO では、pending evaluations が将来情報を与えることを考慮して次の候補を選びます。具体的な処理は acquisition family によって異なりますが、BoTorch では多くの MC acquisition が `X_pending` を扱える設計になっています。

## 4.8 Sequential greedy batch と joint optimization

batch candidates の生成には、q 点を一度に joint optimization する方法だけでなく、1点ずつ条件付きで追加する sequential greedy strategy もあります。

```text
candidate 1 を選ぶ
    ↓
candidate 1 を pending / conditioned として扱う
    ↓
candidate 2 を選ぶ
    ↓
...
```

joint optimization と greedy construction は同じアルゴリズムではありません。計算量と候補集合品質の trade-off があります。

## 4.9 Monte Carlo と sample consistency

q-acquisition の多くは MC expectation を最適化します。

acquisition optimization の途中で sampling noise が過度に変化すると、objective surface 自体が揺れて gradient optimization が難しくなります。QMC sampling や共通 base samples は、MC estimator の variance と optimization stability に関係します。

これは acquisition の理論的 utility と、その数値最適化をつなぐ重要な実装論点です。

## 4.10 Log formulation

qEI / qNEI の値が非常に小さい領域では、数値 underflow や gradient degradation が問題になります。

qLogEI / qLogNEI は improvement-based criterion の数値安定性を改善する formulation です。log variant を「異なる探索目的」と考えるのではなく、同じ意思決定基準を安定して最適化するための重要な数値的設計として理解します。

## 4.11 Batch size と計算量

q を増やすと候補集合の joint dimension が増え、MC sampling と acquisition optimization の計算負荷も増えます。

したがって大きな q では、

- joint optimization の難易度
- posterior sampling cost
- restart / raw sample 数
- sequential generation の利用
- pending points の数

などを含めて設計する必要があります。

## 4.12 BoTorch / robotorchan との対応

BoTorch は qLogEI、qLogNEI などの batch / noisy acquisition と pending-point semantics を提供します。robotorchan は標準手法を再実装せず、BoTorch native path を利用します。

具体的な API は [Standard acquisition](../../optimization/standard_acquisition.md)、現在の対応状況は [Acquisition integration status](../../optimization/acquisition-status.md) を参照してください。

## 4.13 まとめ

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

次の発展として、情報量そのものを価値とする [Information-theoretic acquisition](05_information_theoretic.md) と、将来の意思決定価値を見る [Lookahead acquisition](06_lookahead.md) があります。
