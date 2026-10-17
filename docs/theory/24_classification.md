# Gaussian Process Classification

## 1. Problem formulation

Gaussian process (GP) classification models a discrete label through a latent
real-valued function, rather than assuming Gaussian observation noise on the
label itself. For binary observations \(y_i\in\{0,1\}\), let

$$
f\sim\mathcal{GP}(m,k),\qquad
y_i\mid f_i\sim\operatorname{Bernoulli}(\pi(f_i)),
\quad f_i=f(x_i).
$$

The link \(\pi:\mathbb R\to(0,1)\) may be logistic,
\(\pi(f)=1/(1+\exp(-f))\), or probit, \(\pi(f)=\Phi(f)\).
The prior on latent values is multivariate Gaussian; the conditional
observation distribution is Bernoulli. Neither the posterior over latent
values nor the predictive label distribution is generally Gaussian.

## 2. Posterior inference

For \(f=(f_1,\ldots,f_n)\) and data
\(\mathcal D=\{(x_i,y_i)\}_{i=1}^n\), Bayes' rule gives

$$
p(f\mid\mathcal D)
=\frac{p(f)\prod_{i=1}^n p(y_i\mid f_i)}
{\int p(f')\prod_{i=1}^n p(y_i\mid f_i')\,df'}.
$$

The Bernoulli likelihood is not conjugate to the Gaussian prior, so the
normalizing integral is generally intractable. Common approximations include
Laplace approximation, expectation propagation, and variational inference.
Their approximation assumptions and numerical properties differ.

### 2.1 Sparse variational inference

Let \(u=f(Z)\) denote inducing variables at inducing inputs \(Z\), and
introduce a variational distribution \(q(u)\). The implied latent
approximation is

$$
q(f)=\int p(f\mid u)q(u)\,du.
$$

A standard evidence lower bound is

$$
\log p(y\mid X)
\geq
\mathcal L
=
\sum_{i=1}^n
\mathbb E_{q(f_i)}[\log p(y_i\mid f_i)]
-
\operatorname{KL}\!\left(q(u)\,\|\,p(u)\right).
$$

The bound follows by Jensen's inequality. The likelihood expectations may
require quadrature or Monte Carlo estimation. For minibatch training,
the data-fit term must be scaled consistently with the total data size.
An ELBO is not the exact Gaussian marginal log likelihood used in GP
regression.

## 3. Latent uncertainty and predictive probability

Given an approximate latent posterior \(q(f_*\mid\mathcal D)\), the
posterior-predictive probability is

$$
\bar p_*(x)
=
p(y_*=1\mid x,\mathcal D)
\approx
\int \pi(f_*)q(f_*\mid\mathcal D)\,df_*.
$$

**In general, \(\bar p_*(x)\neq\pi(\mathbb E_q[f_*])\).**
Applying the link to the posterior mean discards latent uncertainty.

Three distinct quantities must not be conflated:

1. **Latent uncertainty:** \(\operatorname{Var}_q[f_*]\), measured on the
   latent-function scale.
2. **Probability uncertainty:**
   \(\operatorname{Var}_q[\pi(f_*)]\), describing posterior variation of
   the Bernoulli success parameter.
3. **Predictive label uncertainty:**
   \(\operatorname{Var}[y_*\mid x,\mathcal D]
   =\bar p_*(1-\bar p_*)\), including irreducible label randomness
   conditional on a known success probability.

The law of total variance makes the distinction explicit:

$$
\underbrace{\bar p_*(1-\bar p_*)}_{\text{predictive label variance}}
=
\underbrace{\mathbb E_q[\pi(f_*)(1-\pi(f_*))]}_{\text{conditional label variance}}
+
\underbrace{\operatorname{Var}_q[\pi(f_*)]}_{\text{probability uncertainty}}.
$$

For a probit link and Gaussian marginal
\(q(f_*)=\mathcal N(\mu_*,s_*^2)\), the predictive probability has the
analytic form

$$
\bar p_*=\Phi\!\left(\frac{\mu_*}{\sqrt{1+s_*^2}}\right).
$$

This identity does not hold for the logistic link without approximation.

## 4. Classification uncertainty and active learning

For \(C\) classes, predictive entropy is

$$
H[y_*\mid x,\mathcal D]
=
-\sum_{c=1}^{C}\bar p_c(x)\log\bar p_c(x).
$$

It measures uncertainty in the future label, not solely uncertainty about
the latent model. Bayesian active learning by disagreement (BALD) uses the
mutual information between the label and latent function:

$$
I(y_*;f_*\mid x,\mathcal D)
=
H[y_*\mid x,\mathcal D]
-
\mathbb E_{q(f_*)}\!\left[H[y_*\mid f_*]\right].
$$

Thus, high predictive entropy need not imply high information gain: a label
can remain intrinsically noisy even when the latent function is well known.
Probability-variance criteria instead use
\(\operatorname{Var}_q[\pi(f_*)]\). Boundary-based criteria focus on
uncertainty near a specified decision threshold; their statistical target
differs from both entropy and variance sampling.

These are *selection objectives*, not properties of a GP posterior alone.
See [Active Learning](acquisition/09_active_learning.md) and
[Classification Active Learning](acquisition/classification-active-learning.md)
for acquisition theory.

## 5. Multiclass classification

For \(C>2\), one common construction uses a vector of latent functions
\(f_1(x),\ldots,f_C(x)\) and a categorical likelihood with softmax link:

$$
p(y=c\mid f(x))
=
\frac{\exp(f_c(x))}{\sum_{j=1}^{C}\exp(f_j(x))}.
$$

The latent functions may be independent a priori or correlated through a
multi-output covariance. The predictive class probability integrates the
softmax over their **joint** latent posterior:

$$
p(y=c\mid x,\mathcal D)
=
\mathbb E_{p(f(x)\mid\mathcal D)}
\left[
\frac{\exp(f_c(x))}{\sum_j\exp(f_j(x))}
\right].
$$

The softmax likelihood is invariant to adding the same constant to all
latent logits, so identifiability requires care. Multiclass inference and
decision rules cannot generally be reduced to a single binary threshold.

## 6. Modeling assumptions and limitations

- The link function and latent covariance jointly determine the class
  probability model; a flexible kernel does not guarantee calibration.
- Label imbalance, asymmetric misclassification costs, and dataset shift
  affect decision thresholds and evaluation, not merely GP fitting.
- A binary Gaussian-likelihood regression model is not equivalent to
  Bernoulli GP classification.
- Calibration measures such as Brier score and log loss assess predictive
  probabilities. Accuracy alone does not assess calibration.
- The latent variance, probability variance, and predictive entropy have
  different units and interpretations; they are not interchangeable.

## 6.1 分類閾値と最適化制約の意味

二値分類で $\bar p(x)=P(y=1\mid x,\mathcal D)$ とすると、
予測クラスは閾値 $\tau$ を使って
$\hat y(x)=\mathbf 1[\bar p(x)\geq\tau]$ と定義できます。
$\tau=0.5$ は対称な誤分類コストの下で自然な選択ですが、
コストやクラス事前確率、意思決定の目的が変われば最適な閾値も変わります。

制約付きベイズ最適化で「合格確率が少なくとも $\eta$」という要件を
課す場合、条件は $\bar p(x)\geq\eta$ です。
これは潜在平均 $\mathbb E[f(x)]\geq0$ と一般に等価ではありません。
また、予測クラスの閾値と最適化上の要求確率は異なる概念です。

## 6.2 不均衡・ラベルノイズ・校正

クラス不均衡がある場合、全体の正解率だけでは少数クラスの性能を
判断できません。確率予測の評価には対数損失、Brier score、
校正曲線などを併用し、必要に応じてクラス別の指標も確認します。

観測ラベルが誤って記録される場合、潜在関数の不確実性だけでなく、
ラベル生成過程そのものをモデル化する必要があります。
たとえば真のクラス $z\in\{0,1\}$ に対して対称な反転確率
$\epsilon\in[0,1/2)$ を仮定すると、

$
P(y=1\mid f)=\epsilon+(1-2\epsilon)\pi(f).
$

この式は通常のBernoulliリンクとは異なる観測過程を表します。
観測ラベルのノイズと予測クラスの不確実性を混同しないことが重要です。

## 6.3 多クラス潜在表現の識別性

softmaxは全クラスの潜在値に同じ定数を加えても変化しません。
そのため、クラス確率は識別可能でも潜在ロジットの絶対的な
基準は識別できない場合があります。基準クラスを固定する方法や
適切な事前分布を用いる方法が考えられます。

多クラスの予測確率は潜在関数の**同時分布**を通じて計算されるため、
各クラスの潜在平均にsoftmaxを適用するだけでは、一般に
事後不確実性を正しく反映できません。

## 7. References

- Rasmussen, C. E., & Williams, C. K. I. (2006).
  *Gaussian Processes for Machine Learning*, Chapter 3.
- Hensman, J., Matthews, A. G. de G., & Ghahramani, Z. (2015).
  *Scalable Variational Gaussian Process Classification*. AISTATS.
- Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011).
  *Bayesian Active Learning for Classification and Preference Learning*.

## 8. Related theory

- [Variational GP](09_variational_gp.md)
- [Mixed Variables](05_mixed_variables.md)
- [Multi-task / Multi-output](07_multitask_multioutput.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
- [Dimensionality Reduction GP](18_dimensionality_reduction_gp.md)
- [Expressive GP Models](22_expressive_gp.md)

For library-specific model families, prediction APIs, and supported
compositions, see the [Classification model guide](../models/classification.md).
