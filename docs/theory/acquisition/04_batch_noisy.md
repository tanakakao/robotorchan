# 4. Batch and Noisy Acquisition

[改善量章](02_improvement.md)のEIと、[GP章](../02_gaussian_process.md)の同時事後分布を前提とします。
本章では候補数を $q$ とし、集合の効用、比較基準の不確実性、結果待ちの観測を区別します。

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

候補間の共分散は joint acquisition の値に影響します。
一方、Gaussian でない sampleable posterior でも、同時サンプル
f(X) によって依存構造を保った batch utility を定義できます。
各点の marginal posterior だけでは、一般に joint acquisition を再現できません。

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

例えば比較基準が0で、二候補の周辺分布がともに標準正規分布だとします。
完全に同じ値を取る $F_2=F_1$ ならqEIは $\mathbb E[(F_1)_+]=1/\sqrt{2\pi}\simeq0.399$ です。
一方 $F_2=-F_1$ なら $\max(0,F_1,F_2)=|F_1|$ なので約0.798になります。
周辺のEIは同じでも、候補を組み合わせる価値は相関によって変わります。

固定した基準に対する集合の改善は、各標本の下で候補を追加しても減りません。
また、既に良い候補が多い集合へ一つの候補を足す増分は小さくなるため、期待改善にも限界効用の逓減があります。
ただし、貪欲に一つずつ選ぶ方法と、集合を同時に最適化する方法の解が必ず一致するわけではありません。

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

最大化問題で baseline B={b_1,...,b_m} と候補集合 X={x_1,...,x_q} を考えると、
代表的な joint noisy improvement は

$$
q\operatorname{NEI}(X)
=
\mathbb E\left[
\left(
\max_{x\in X}f(x)
-
\max_{b\in B}f(b)
\right)_+
\middle|\mathcal D_n
\right]
$$

と表せます。ここで $(u)_+=\max(u,0)$ です。
候補値と baseline 値は同じ joint posterior から評価し、
baseline の最大値も固定した観測値ではなく確率変数として扱います。
この式は基本的な unconstrained noisy-improvement の概念式であり、
制約、目的変換、pending points、近似・pruning を含む場合は
対応する utility と計算手順を別途定義する必要があります。

qNEIはさらにcandidate batchのjoint posteriorを扱うため、

- baseline uncertainty
- candidate uncertainty
- candidate correlation
- observation noise

を同時に扱うことになります。

同じ点を候補と基準の両方へ置くと、正しい同時標本ではその潜在値は同一です。
基準がその一点だけなら改善は各標本で零となります。
ところが候補値と基準値を独立な $\mathcal N(0,1)$ として生成すると、差の分散が2になり、
誤って $\mathbb E[(F-F')_+]=1/\sqrt\pi\simeq0.564$ の改善を与えてしまいます。
ノイズ付き改善では候補同士だけでなく、候補と基準の依存も保つ必要があります。

## 4.7 Baseline points

NEI 系では baseline set が重要です。baseline は「何と比較して improvement を測るか」を posterior 上で定義するために使われます。

## 4.8 Pending points と非同期BO

並列実験では、すでに評価を開始したが結果が返っていない点

$$
X_{\mathrm{pending}}
$$

が存在します。

pending points を無視すると、同じ領域へ重複して新しい候補を投入する可能性があります。

非同期BOでは、完了した観測で事後分布を更新し、結果待ちの実評価を
候補選択の条件に含めます。結果待ちの評価点と、将来の結果を仮定する
lookahead の仮想観測は別の概念です。

## 4.9 Sequential greedy batch と joint optimization

batch candidates の生成には、q 点を一度に joint optimization する方法だけでなく、1点ずつ条件付きで追加する sequential greedy
strategy もあります。

```text
candidate 1 を選ぶ
    ↓
candidate 1 の未知の結果を考慮して次の追加価値を評価する
    ↓
candidate 2 を選ぶ
    ↓
...
```

joint optimizationとgreedy constructionは同じアルゴリズムではありません。
sequential optimizationで先に選んだcandidateをpending-like contextとして利用する場合も、
それは外部で既に実行中の評価点と同一ではありません。

## 4.10 Fantasizationをpendingと分ける

fantasizationは、未観測点のoutcomeをsampleし、その仮想観測でconditionしたmodelを作る操作です。
これは実際に結果待ちとなっている評価点そのものではありません。

Knowledge Gradientやmulti-step lookaheadではacquisition内部の将来意思決定を表すために
fantasy modelが本質的です。一方、すべてのバッチ獲得関数や非同期の手順が、
仮想観測ごとの明示的なモデル構築を必要とするわけではありません。

したがって次を分けます。

~~~text
q > 1              同時に選ぶ候補集合
pending            結果待ちの実評価
fantasy            仮想的な観測結果による条件付け
auxiliary decisions  lookahead の内部意思決定変数
~~~

これらの役割は異なります。

## 4.11 Monte Carlo と sample consistency

q-acquisition の多くは MC expectation を最適化します。

acquisition optimization の途中で sampling noise が過度に変化すると、objective surface 自体が揺れて gradient
optimization が難しくなります。QMC sampling や共通 base samples は、MC estimator の variance と optimization 
stability に関係します。

これは acquisition の理論的 utility と、その数値最適化をつなぐ重要な実装論点です。

## 4.12 Log formulation

qEI / qNEI の値が非常に小さい領域では、数値 underflow や gradient degradation が問題になります。

qLogEI / qLogNEI は improvement-based criterion の数値安定性を改善する formulation です。log variant
の意図は改善基準の安定した数値評価です。ただし厳密な対数変換と、正部分・最大値などの平滑化を含む近似は区別し、有限の平滑化係数による差も考慮します。

## 4.13 Batch size と計算量

q を増やすと候補集合の joint dimension が増え、MC sampling と acquisition optimization の計算負荷も増えます。

したがって大きな q では、

- joint optimization の難易度
- posterior sampling cost
- restart / raw sample 数
- sequential generation の利用
- pending points の数

などを含めて設計する必要があります。

## 4.15 この章で覚えておくこと

候補数、観測ノイズ、結果待ち、仮想観測による条件付けは別の軸です。
候補集合の効用は、周辺スコアの順位だけではなく同時分布で評価します。
ノイズ付き改善では、基準の潜在値と候補の値を同じ標本で扱うことが重要です。

結果待ちの実評価と、先読みのために生成する仮想結果は区別します。
補助的な将来判断の点も実評価候補とは異なり、すべてのバッチ手法が同じ条件付けを必要とするわけではありません。
対数・平滑化による数値的な改善は、効用の定義と近似誤差を分けて理解します。

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
