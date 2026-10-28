# Uncertain-input Gaussian Process

[GP章](02_gaussian_process.md)の共分散と、[カーネル章](03_kernel.md)のRBFを前提とします。
本章では入力次元を $d$ とし、入力点を確率変数とするモデルと、入力分布に対する平均応答のモデルを区別します。

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

ただし、この二つの誤差表現は追加の仮定なしに同じ条件付き分布を与えるものではありません。
$x_{obs}=x_{true}+\delta$ から $p(x_{true}\mid x_{obs})$ を求めるには、
真の入力の事前分布も必要です。以下では入力の条件付き分布が別途与えられたものとして考えます。

通常のカーネルは確定した二点の関数値の共分散です。入力分布 $P_i,P_j$ に対して、
**独立な二つの積分変数**を使う分布間カーネルを次のように定義できます。

```text
k_bar(i, j)
  = E_{x_i, x_j}[k(x_i, x_j)]
```

このカーネルは、入力分布に対する平均関数値 $G_i=\int f(x)\,dP_i(x)$ の共分散を正確に与えます。
固定した入力分布とGPの下で、線形汎関数 $G_i$ の集合は同時正規分布になります。
したがって、一度だけ入力を乱数で動かして学習する処理とは異なります。

一方、真の観測が $f(X_i)$ という一回のランダムな入力点で得られた場合、
入力を積分した分布は一般に正規分布ではありません。特に分散の対角項では、同じ $X_i$ を両側に用いる
$\mathbb E[k(X_i,X_i)]$ と、独立な二標本の $\mathbb E[k(X_i,X_i')]$ は異なります。
分布平均のカーネルをランダム入力点のモデルへ使う際は、この差を近似として扱う必要があります。

## 2. Gaussian uncertainty と RBF kernel

2つの入力分布を

```text
x_i ~ Normal(mu_i, Sigma_i)
x_j ~ Normal(mu_j, Sigma_j)
```

とし、二つの入力が独立で、`Lambda = diag(ell_1^2, ..., ell_d^2)` が正定値なら、分散1のRBFに対する分布間共分散は

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

この式は差 $X_i-X_j$ が平均 $\mu_i-\mu_j$、共分散 $\Sigma_i+\Sigma_j$ の正規分布となることを使い、
二次形式の指数関数を積分して得られます。信号分散が $\sigma_f^2$ なら右辺全体をその値で倍します。
数値計算では逆行列そのものより、Cholesky分解による線形求解と対数行列式を使えます。

例えば1次元で $\ell=1$、同じ平均を持つ分散0.25の入力分布同士なら、分布平均の分散は
$1/\sqrt{1+0.25+0.25}\simeq0.816$ です。一方、分散1の定常RBFの $f(X)$ の分散は、
零平均の事前分布の下で1のままです。平均応答と一回の実現値を区別する必要がある理由がここにあります。

分布間カーネルの半正定値性は、特徴写像 $\phi$ の期待値を使って
$\bar k(P,Q)=\langle\mathbb E_P[\phi(X)],\mathbb E_Q[\phi(X)]\rangle$ と表せることからも分かります。
存在に必要な積分可能性を仮定すれば、係数 $a_i$ に対する二次形式は
$\|\sum_i a_i\mathbb E_{P_i}[\phi(X)]\|^2\ge0$ です。

## 3. Diagonal uncertainty と full covariance

正規入力分布は、対角共分散か一般の共分散で記述できます。

### 3.1 標準偏差

各座標の標準偏差ベクトルを与える場合、

```text
Sigma_x = diag(s_x^2)
```

として各入力次元が独立な正規分布を表します。$s_x$ の各成分は対応する座標の標準偏差です。

### 3.2 Full covariance

共分散行列 $\Sigma_x\in\mathbb R^{d\times d}$ を与えると、同じ入力点の座標間の相関を保持できます。

例えば2つのプロセス条件が同じ校正誤差を共有している場合、diagonal variance だけでは
表現できない uncertainty structure を full covariance で扱えます。

共分散行列には次の性質が必要です。

- finite
- symmetric
- positive semidefinite

を検証します。

標準偏差と共分散行列は同じ不確かさの別表現です。二つの独立したノイズ源として重ねて加算しません。
また、入力点内の座標間共分散と、複数の入力点が共通の校正誤差を持つ点間依存性は別です。
後者では一般に差の共分散が $\Sigma_i+\Sigma_j-\operatorname{Cov}(X_i,X_j)-\operatorname{Cov}(X_j,X_i)$ となり、独立入力の式をそのまま使えません。

## 4. Training representation

各入力分布は、平均と共分散を並べて

```text
[mu_x, vec(Sigma_x)]
```

という形で表せます。ただし、並べたベクトルへ通常のユークリッド距離を適用するのではなく、
平均・共分散として意味を分けてカーネルを評価します。

## 5. Training uncertaintyを学習しているわけではない

入力分布を既知として与えることと、入力・応答から入力誤差分散自体を推定することは異なる問題です。

~~~text
外部の測定・校正
  → 入力の平均と標準偏差、または平均と共分散
  → 入力分布を定義
  → 分布間の共分散を計算
~~~

入力uncertainty自体が未知なら、そのuncertaintyをどう推定するかは別のstatistical modelが必要です。

## 6. Prediction candidate は deterministic

予測先の入力を確定値とする構成では、

**uncertain training inputs + deterministic posterior query**

です。

という問題を扱います。一方、提案した入力自体が実行時に揺らぐ場合は、
その摂動分布について期待応答やリスクを別に定義します。
例えば $X\sim\mathcal N(0,0.25)$ と $f(x)=x^2$ なら、$f(\mathbb E[X])=0$ ですが、
$\mathbb E[f(X)]=0.25$ です。平均入力での予測だけでは実行時の平均応答にも一致しません。

## 7. Mixed uncertain input

mixed search space では continuous coordinate の uncertainty と categorical identity を
分離する必要があります。

一つの構成では、連続部分だけに正規入力分布を置き、カテゴリは確定値としてカテゴリ共分散で扱います。

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

確定カテゴリを持つ混合入力とは異なり、カテゴリ自体が未知なら、各入力についてカテゴリ確率ベクトルを定義します。

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

という操作とは異なります。カテゴリ共分散行列を $B$、二つの確率ベクトルを $p,q$ とすれば、
独立なカテゴリ標本についての期待共分散は $p^\top Bq$ です。
同じ分布での $p^\top Bp$ も平均応答の分散であり、同じランダムカテゴリを両側に使う場合の
$\sum_c p_c B_{cc}$ とは一般に異なります。

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

実行時の入力摂動を

~~~text
x_realized = x_candidate + delta
~~~
と定義しても、摂動標本の個数が設計変数の次元になるわけではありません。

実際に最適化したいquantityが

```text
E_delta[f(x + delta)]
```

なのか、quantile、CVaR、worst-case なのかを定義する必要があります。

これは training-input uncertainty の kernel marginalization だけでは決まりません。

入力不確実性にタスク・忠実度・次元削減を組み合わせる場合も、それぞれの確率モデルと座標の意味を別途定義します。
具体的な対応範囲は[実装ガイド](../guide/16_uncertain_input_gp.md)を参照してください。

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

入力位置の不確かさと観測応答のノイズ、学習入力の不確かさと実行時の摂動は、それぞれ別の問題です。
入力分布を既知として与えることと、その分布自体を推定することも区別します。
正規入力とRBFの解析式は、その仮定に依存した分布間共分散です。

分布平均の応答と、一回のランダム入力点での応答は同じ確率変数ではありません。
混合入力では連続部分とカテゴリの確率を分け、実行時の頑健性は期待値やリスクの目的として定義します。

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
