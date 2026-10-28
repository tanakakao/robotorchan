# Robust Gaussian Process

[GP章](02_gaussian_process.md)の観測尤度と、[変分GP章](09_variational_gp.md)のELBOを前提とします。
本章では潜在値 $f_i=f(x_i)$ と残差 $r_i=y_i-f_i$ を用い、外れ値への影響を尤度の形から考えます。

## 直感

標準的な Gaussian process (GP) 回帰は、観測誤差を Gaussian と置くことで
解析的に扱いやすい posterior を得ます。一方、製造・実験データでは、センサ異常、
転記誤り、一時的なプロセス逸脱などにより、少数の大きな残差が混入することがあります。

- heavy-tailed likelihood: Student-t
- nominal / contaminated observation mixture
- sparse observation correction: relevance pursuit

これは heteroskedastic noise や robust objective とは別の問題です。

### 最初に「robust」の対象を分ける

「robust GP」という名前だけでは、何に頑健なのかは決まりません。

~~~text
Yの一部に極端な誤差
  → outlier / contamination / heavy tail

Var(Y | X=x) が場所で変化
  → heteroskedastic noise

観測したX自体が不確か
  → uncertain-input model

f(x)のsmoothnessやlengthscaleが場所で変化
  → nonstationary model

実行時にXが揺れる
  → robust decision / input perturbation
~~~

この章の中心は最初の **observation-side robustness** です。
同じ「予測が外れる」という症状でも、原因が違えばモデルも変わります。

## 1. Gaussian likelihood と外れ値感度

標準的な観測モデルを

```text
f ~ GP(m, k)
y_i = f(x_i) + epsilon_i
epsilon_i ~ Normal(0, sigma^2)
```

とします。Gaussian log likelihood の負の対数は、定数項を除けば

```text
(y_i - f_i)^2 / (2 sigma^2)
```

を含みます。そのため残差の寄与は二乗で増加します。通常のGaussian modelでは、極端な残差を
latent function・noise level・kernel hyperparameterの組合せで説明しようとするため、
少数の観測がposteriorやhyperparameter推定へ大きく影響することがあります。

robust GP の各方式は、この影響を異なる仮定で緩和します。

## 2. Student-t likelihood

Student-t observation model は

```text
y_i | f_i ~ StudentT(nu, f_i, sigma)
```

と書けます。自由度 `nu` が小さいほどtailが重くなり、大きな残差へGaussianより高い確率を
与えます。これは「どの点がoutlierか」を明示的に選ぶモデルではなく、
**観測誤差全体へheavy-tailed likelihoodを置く**考え方です。

非共役なStudent-t尤度を誘導点変分近似で扱う場合、観測数を $n$ として負のELBOを次のように平均化できます。

```text
loss = (- E_q(f)[log p(y | f)] + KL[q(u) || p(u)]) / n
```
ここで $q(u)$ は誘導変数の変分分布、$q(f)$ はそこから構成する潜在値の分布です。
Student-tの尺度を $s>0$（上の `sigma` に対応）、自由度を $\nu>0$ とすると、定数を除く負対数尤度と残差方向の勾配は

$$
\ell(r)=\frac{\nu+1}{2}\log\left(1+\frac{r^2}{\nu s^2}\right),\qquad
\frac{d\ell}{dr}=\frac{(\nu+1)r}{\nu s^2+r^2}.
$$

Gaussianの勾配 $r/s^2$ が残差とともに増え続けるのに対し、上の勾配は大きな残差で零へ近づきます。
例えば $s=1,\nu=3,r=10$ では勾配は $40/103\simeq0.388$ で、Gaussianの10より小さくなります。
これは潜在値を固定した尤度の局所的な感度であり、学習済みGP全体の影響が必ず小さいという保証ではありません。

また、$s$ は標準偏差ではありません。$\nu>2$ の観測誤差分散は $\nu s^2/(\nu-2)$ です。
$\nu\le2$ では有限な分散がなく、$\nu\le1$ では平均も存在しないため、位置パラメータを期待値と解釈できません。
入力・タスクの構造を追加しても、この尤度の意味は変わりません。

## 3. Contamination mixture

少数点だけが異なる生成機構から来ると考える場合、contamination model が自然です。

```text
p(y_i | f_i)
  = (1 - pi) Normal(y_i; f_i, sigma_in^2)
  + pi       Normal(y_i; f_i, sigma_out^2)

0 < pi < 1
sigma_out > sigma_in > 0
```

`pi` は contamination probability です。Student-t が全観測へ同じ heavy-tailed family を
適用するのに対し、こちらは nominal component と gross-error component を明示的に分けます。

```text
loss
  = (- E_q(f)[log p_mix(y | f)] + beta * KL[q(u) || p(u)]) / n
```
上の目的関数で $\beta=1$ なら通常のELBOに対応します。
KLの重みを変更した目的を、無条件に同じ周辺尤度のELBOと呼ぶことはできません。

潜在成分 $z_i$ を通常なら0、異常なら1とすると、残差 $r_i$ を与えた下での異常成分の確率は

$$
P(z_i=1\mid y_i,f_i)
=\frac{\pi\,\mathcal N(r_i;0,\sigma_{out}^2)}
{(1-\pi)\mathcal N(r_i;0,\sigma_{in}^2)+\pi\mathcal N(r_i;0,\sigma_{out}^2)}.
$$

これはBayesの定理で混合成分を推論したもので、単に残差へ固定の閾値を当てる規則ではありません。
潜在値 $f_i$ も未知なので、最終的な異常確率にはその事後不確実性も反映する必要があります。

## 4. Sparse correction と relevance pursuit

別の考え方は、すべての観測分布を heavy-tailed に変更せず、少数の観測だけに sparse な
補正を割り当てることです。概念的には

```text
y = f(X) + a + epsilon
```

とし、補正ベクトル `a` の大部分がゼロである構造を利用します。

この方式では「少数の観測だけに追加のnoise correctionが必要」というsparsityを
明示します。Student-tのglobal heavy-tail assumptionや、固定mixture parameterを使う
contamination modelとは異なる仮定です。

ここで補正 $a$ を推定する平均のずれとするか、確率的な追加誤差とするかを区別します。
例えば独立な $a_i\sim\mathcal N(0,v_i)$、$v_i\ge0$ として、非零の $v_i$ が少数なら、

$$
K_y=K_f+\sigma^2I+\operatorname{diag}(v)
$$

となります。これは少数の観測の分散を膨らませる構成です。
決定論的な $a_i$ による平均補正と同じ推論ではありません。
Relevance Pursuitでは、このような追加の観測分散の支持集合を選ぶ考え方を用います。
補正の自由度を無制限にすると真の局所変化まで誤差として説明し得るため、疎性の仮定が重要です。

### Relevance Pursuitは「外れ値を削除する前処理」ではない

Relevance Pursuit系では、候補となる異常観測をdataframeから消して通常GPを再学習する、
という操作を目的にしていません。sparseな追加noise構造をmodel fittingの中で扱います。

したがって、

~~~text
outlier detection → 行を削除 → ordinary GP
~~~

と

~~~text
robust GP → 異常観測を含めたままobservation modelを拡張
~~~

は別の設計です。

## 5. Mixed variables

混合入力でも、観測誤差の頑健性と入力共分散は別々に定義します。
連続変数とカテゴリの意味を分け、カテゴリコードの数値差を連続距離とみなしません。

特に MultiTask では、

```text
K((x, t), (x', t')) = K_data(x, x') * K_task(t, t')
```

のように分離可能な共分散を組み合わせられます。これはタスク間で入力側の相関構造を共有する仮定です。
## 6. Robustness の種類を混同しない

| 問題 | 主な対象 | 代表的な考え方 |
| --- | --- | --- |
| 外れ値・heavy tail | observation model | Student-t |
| gross-error mixture | observation model | contamination |
| sparse outlier correction | observation model | relevance pursuit |
| 入力依存ノイズ | observation variance | heteroskedastic GP |
| 入力座標の不確かさ | input covariance | uncertain-input GP |
| 実行時入力摂動 | candidate / scenario | perturbation sampling |
| CVaR / worst case | decision objective | risk aggregation |

したがって robust surrogate を使うだけで CVaR 最適化になるわけではなく、逆に CVaR
objective を使うだけで training observation の外れ値に頑健になるわけでもありません。

### latent posteriorと観測分布を区別する

robust likelihoodを使うときは、

~~~text
latent posterior
p(f(x) | D)

conditional observation likelihood
p(y | f(x))
~~~

を区別します。次回の観測の予測分布は、尤度を潜在事後分布で積分した
$p(y_*\mid D)=\int p(y_*\mid f_*)p(f_*\mid D)\,df_*$ です。

この違いは「将来の真の応答を最適化したい」のか、「次の noisy observation自体の分布を
扱いたい」のかを考えるときに重要です。

## 7. Bayesian optimization との関係

robust observation model は posterior mean と posterior uncertainty の両方を変えるため、
EI、UCB、NEI などの acquisition value に間接的に影響します。

一方で acquisition が期待値、quantile、CVaR、worst-case のどれを最適化するかは別の層です。
surrogate robustness と decision robustness は直交する設計軸として扱う方が安全です。

## 8. 仮定の比較

| 仮定 | 対応する考え方 |
| --- | --- |
| 残差分布全体がheavy-tailed | Student-t likelihood |
| nominal dataにgross-error componentが混ざる | contamination mixture |
| 少数観測へsparse correctionを置く | relevance pursuit |
| noise variance自体が入力で変わる | heteroskedastic GP |
| 入力値そのものが不確か | uncertain-input GP |
| covariance structureが場所で変わる | nonstationary GP |

## 10. この章で覚えておくこと

頑健なGPは、外れ値を削除する前処理ではなく、観測誤差の仮定を変える方法です。
Student-tの重い裾、混合成分、疎な追加分散は異なる機構を表します。
観測ノイズの入力依存性、入力位置の不確かさ、潜在関数の非定常性とも区別します。

また、外れ値に頑健な予測モデルを使うことと、実行時の最悪値やCVaRを最適化することは別です。
混合入力やマルチタスクでも、入力の構造と観測誤差の機構を分け、
潜在応答の事後分布と次回観測の予測分布を使い分けます。

## 参考文献

1. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.
2. Vanhatalo, J., Jylänki, P., and Vehtari, A. (2009).
   Gaussian process regression with Student-t likelihood.
   *Advances in Neural Information Processing Systems 22*.
3. Jylänki, P., Vanhatalo, J., and Vehtari, A. (2011).
   Robust Gaussian process regression with a Student-t likelihood.
   *Journal of Machine Learning Research*, 12, 3227-3257.
4. Knoblauch, J., Jewson, J., and Damoulas, T. (2022).
   An optimization-centric view on Bayes' rule: Reviewing and generalizing
   variational inference. *Journal of Machine Learning Research*, 23.
5. Ament, S., Santorella, E., Eriksson, D., Ludocic, M., and Poloczek, M. (2024).
   Robust Gaussian processes via relevance pursuit.
   *Proceedings of the 27th International Conference on Artificial Intelligence
   and Statistics (AISTATS)*.

## 関連章

- [Gaussian Process](02_gaussian_process.md)
- [Heteroskedastic / Replicate Noise](15_heteroskedastic_noise.md)
- [Uncertain-input Gaussian Process](16_uncertain_input_gp.md)
- [Acquisition Function](04_acquisition_function.md)
