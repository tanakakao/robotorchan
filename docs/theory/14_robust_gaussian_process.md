# Robust Gaussian Process

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

実装上は


```text
loss = (- E_q(f)[log p(y | f)] + KL[q(u) || p(u)]) / n
```
- Student-t 単一タスク GP
- 混合入力 Student-t 単一タスク GP
- Student-t マルチタスク GP
- 混合入力 Student-t マルチタスク GP

MultiTask 版では data covariance と task covariance を積で構成する ICM 型の構造を使い、
Mixed MultiTask 版では task feature と categorical feature を分離します。

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
- Contaminated 単一タスク GP
- 混合入力 Contaminated 単一タスク GP
- Contaminated マルチタスク GP
- 混合入力 Contaminated マルチタスク GP

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

- Robust Relevance Pursuit 単一タスク GP
- 混合入力 Robust Relevance Pursuit 単一タスク GP
- Robust Relevance Pursuit マルチタスク GP
- 混合入力 Robust Relevance Pursuit マルチタスク GP

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

Mixed 版で robust observation mechanism 自体は変わりません。変わるのは input covariance
です。continuous feature と categorical feature を分離し、カテゴリコードを Euclidean
distance の連続値として解釈しません。

特に MultiTask では、


```text
K((x, t), (x', t')) = K_data(x, x') * K_task(t, t')
```
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

observed response distribution
p(y | f(x))
~~~

を区別します。

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

- robust GPは「怪しい行を自動削除するGP」の総称ではない
- Student-t、contamination、relevance pursuitは異なるobservation assumptionsを持つ
- heteroskedasticity、input uncertainty、nonstationarityは別のdata-generating mechanismである
- surrogate robustnessとCVaR / worst-caseなどのdecision robustnessは別の設計軸である
- BOへ渡すlatent posteriorと、観測noiseを含むresponse distributionを区別する
- Mixed / MultiTask化しても、robust mechanismとstructural featureの責務は分離する

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
