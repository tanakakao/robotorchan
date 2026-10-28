# Uncertain-input Gaussian Process

## 直感

通常の GP は入力 `x` が正確に既知であると仮定します。しかし材料配合量、温度、位置、
加工条件、センサ座標などでは、設定値・観測値と真の実現値が一致しないことがあります。

uncertain-input GP は output noise ではなく、**入力座標そのものの不確かさ**を covariance
計算へ組み込みます。

### 最初に「どのXが不確かか」を分ける

入力の不確かさには少なくとも2つの異なる問題があります。


~~~text
training-input uncertainty
  過去の観測を「どのXで得たか」が不確か

candidate-time perturbation
  提案したnominal Xを実行すると実現Xが揺らぐ
~~~

前者は**surrogate inference**の問題です。後者は主に**decision utility**の問題です。
両方が同時に存在することもありますが、一方を扱えば他方も自動的に解決するわけではありません。

## 1. 入力誤差の確率モデル

観測された入力を `x_obs`、真の入力を `x_true` として、


```text
x_true ~ Normal(mu_x, Sigma_x)
```

または誤差表現として


```text
x_obs = x_true + delta
delta ~ Normal(0, Sigma_x)
```

を考えます。

通常の GP kernel `k(x, x')` は確定した2点間の covariance です。入力が確率変数なら、
必要なのは入力分布について周辺化した covariance です。


```text
k_bar(i, j)
  = E_{x_i, x_j}[k(x_i, x_j)]
```

このexpected kernelにより、入力位置のuncertaintyをGP covarianceへ反映できます。

ここで行っているのは「観測された `観測入力` を一度だけrandom jitterして通常GPをfitする」
処理ではありません。各training pointを入力分布として扱い、その分布間でkernelを周辺化します。

## 2. Gaussian uncertainty と RBF kernel

2つの入力分布を


```text
x_i ~ Normal(mu_i, Sigma_i)
x_j ~ Normal(mu_j, Sigma_j)
```

とし、ARD lengthscale matrix を `Lambda` とすると、expected RBF covariance は概念的に


```text
k_bar(i, j)
  = |Lambda|^(1/2)
    |Lambda + Sigma_i + Sigma_j|^(-1/2)
    exp(
      -1/2 (mu_i - mu_j)^T
      (Lambda + Sigma_i + Sigma_j)^(-1)
      (mu_i - mu_j)
    )
```

という形になります。

入力uncertaintyがゼロなら `Sigma_i = Sigma_j = 0` となり、
このexpected covarianceは通常のRBF covarianceへ戻ります。

なお、この解析式は**Gaussian input uncertainty × RBF kernel**という組合せに依存します。
「任意のkernelへ同じ式を適用できる」という一般則ではありません。

実装では Cholesky decomposition を使って線形系と log determinant を計算し、明示的な
matrix inverse を避けています。

## 3. Diagonal uncertainty と full covariance

Uncertain Input 単一タスク GP は2種類の training-input uncertainty 表現を受け取ります。

### 3.1 標準偏差


```text
観測入力_std.shape == 観測入力.shape
```

の場合、


```text
Sigma_x = diag(観測入力_std^2)
```

として各入力次元が独立な Gaussian uncertainty を表します。

### 3.2 Full covariance


```text
観測入力_covar.shape == 観測入力.shape[:-1] + (d, d)
```

を渡すと、入力次元間の相関を保持できます。

例えば2つのプロセス条件が同じ校正誤差を共有している場合、diagonal variance だけでは
表現できない uncertainty structure を full covariance で扱えます。

実装では covariance matrix に対して

- finite
- symmetric
- positive semidefinite

を検証します。

`観測入力_std` と `観測入力_covar` は同時には指定せず、必ずどちらか一方を指定します。

## 4. Training representation

実装内部では各 training point を


```text
[mu_x, vec(Sigma_x)]
```

という augmented representation に変換し、専用 kernel が mean と covariance を分離して
読み取ります。

## 5. Training uncertaintyを学習しているわけではない

つまりモデルが `観測入力` と `観測応答` だけから各点の入力誤差分散を自動推定するわけではありません。


~~~text
caller
  → observed 観測入力
  → 観測入力_std または 観測入力_covar
  → uncertain-input kernel

model
  → 与えられた入力分布を使ってcovarianceを計算
~~~

入力uncertainty自体が未知なら、そのuncertaintyをどう推定するかは別のstatistical modelが必要です。

## 6. Prediction candidate は deterministic

したがって、このモデルが直接表すのは

**uncertain training inputs + deterministic posterior query**

です。

「提案した candidate 自体も実行時に揺らぐ」という問題を同じ posterior call が自動的に
解決するわけではありません。candidate perturbation は別の scenario / objective /
acquisition layer で扱います。

## 7. Mixed uncertain input

mixed search space では continuous coordinate の uncertainty と categorical identity を
分離する必要があります。

混合入力 Uncertain Input 単一タスク GP は

- continuous dimensions: Gaussian uncertainty を周辺化
- categorical dimensions: deterministic category として categorical kernel で処理

します。

covariance は概念的に


```text
K_mixed
  = K_uncertain_continuous
    * K_categorical
```

です。

カテゴリコードを `0.0, 1.0, 2.0` のような数値として Gaussian perturbation することは
意味的に不適切なので行いません。

## 8. Categorical uncertainty

カテゴリそのものが不確かな場合は別のモデル化が必要です。

例えば真の category `c` が


```text
P(c = c_1) = p_1
P(c = c_2) = p_2
...
P(c = c_K) = p_K
```

という categorical distribution に従うなら、category pair に対する covariance を確率で
周辺化します。

概念的には


```text
E[k(c, c')]
  = sum_a sum_b
    P(c = a) P(c' = b) k(a, b)
```

です。

### Uncertain categoricalはdeterministic categoryのMixed GPとも違う

混合入力 Uncertain Input 単一タスク GP のcategoryは確定値です。一方、
Uncertain Categorical 単一タスク GP は1つのcategorical featureについて
category probability vectorを明示的に受け取ります。

例えば


~~~text
P(material=A) = 0.7
P(material=B) = 0.2
P(material=C) = 0.1
~~~
したがって、


~~~text
categorical code + Gaussian jitter
~~~
## 9. Input noise と output noise の違い

output noise は


```text
y = f(x) + epsilon_y
```

であり、同じ `x` に対する observation のばらつきです。

input uncertainty は


```text
x_true = x_obs + epsilon_x
y = f(x_true) + epsilon_y
```

の `epsilon_x` に対応します。

非線形な `f` では、input uncertainty を単純に output noise へ置き換えることは一般には
できません。入力 uncertainty は kernel geometry 自体を変えるためです。

output noise については
[Heteroskedastic / Replicate Noise](15_heteroskedastic_noise.md) を参照してください。

## 10. Training-input uncertainty と candidate perturbation

| 問題 | 不確かな対象 | 主な処理層 |
| --- | --- | --- |
| Training-input uncertainty | 過去の観測 X | surrogate kernel |
| Candidate perturbation | 提案 X の実現値 | scenario / objective |
| Environmental variable | 制御不能な w | scenario / risk |
| Output noise | 観測 y | likelihood / noise model |

candidate perturbationを


~~~text
x_realized = x_candidate + delta
~~~
したがってscenario数 `n_w` をsurrogateのpublic input dimensionへ追加するわけではありません。

実際に最適化したいquantityが


```text
E_delta[f(x + delta)]
```

なのか、quantile、CVaR、worst-case なのかを定義する必要があります。

これは training-input uncertainty の kernel marginalization だけでは決まりません。

現時点のpublic uncertain-input familyは主に

- Uncertain Input 単一タスク GP
- 混合入力 Uncertain Input 単一タスク GP
- Uncertain Categorical 単一タスク GP

です。

MultiTask / MultiFidelity / dimensionality-reductionとのcross-productを一般にサポートすると
本文から推測してはいけません。複数構造を組み合わせる場合はregistryと実装contractを確認します。

## 11. Bayesian optimization との関係

uncertain training input を無視すると、観測位置を過度に正確だとみなし、lengthscale や
posterior uncertainty を誤って推定する可能性があります。

一方、candidate robustness を求める場合は surrogate posterior に加えて perturbation scenario
を acquisition objective に伝播させる必要があります。

したがって、


```text
uncertain-input surrogate
!= robust candidate objective
```

です。両方が必要な問題では、それぞれを独立した層として合成します。

## 13. この章で覚えておくこと

- output noiseとinput uncertaintyは異なる
- training-input uncertaintyとcandidate-time perturbationも異なる
- 現行continuous modelはcaller-supplied Gaussian input uncertaintyをexpected RBF kernelへ入れる
- analytic expected-kernel式はGaussian uncertainty × RBFという仮定に依存する
- Mixed版ではGaussian uncertaintyはcontinuous dimensionsだけに適用する
- uncertain categorical inputはcategory probability distributionとして別modelで扱う
- candidate robustnessはscenario / risk-objective layerで定義する

## 参考文献

1. Girard, A., Rasmussen, C. E., Quiñonero-Candela, J., and Murray-Smith, R. (2003).
   Gaussian process priors with uncertain inputs: Application to multiple-step
   ahead time series forecasting.
   *Advances in Neural Information Processing Systems 15*.
2. McHutchon, A. and Rasmussen, C. E. (2011).
   Gaussian process training with input noise.
   *Advances in Neural Information Processing Systems 24*.
3. Dallaire, P., Besse, C., and Chaib-draa, B. (2009).
   Learning Gaussian process models from uncertain data.
   *Neural Information Processing*, Lecture Notes in Computer Science.
4. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [Gaussian Process](02_gaussian_process.md)
- [Robust Gaussian Process](14_robust_gaussian_process.md)
- [Heteroskedastic / Replicate Noise](15_heteroskedastic_noise.md)
- [Acquisition Function](04_acquisition_function.md)
