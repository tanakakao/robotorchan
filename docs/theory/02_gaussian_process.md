# 2. Gaussian Process

## 2.1 Gaussian Process の前に：確率分布から考える

Gaussian Process（GP）を理解するには、まず「予測値を1つ返す」のではなく、
**予測値に確率分布を持たせる**という考え方が重要です。

例えば材料強度について「120 MPa」とだけ予測するモデルでは、その予測に
どれほど自信があるかは直接分かりません。確率分布として予測すれば、
「120 MPa付近がもっともありそうだが、その周辺にも可能性がある」と表現できます。

### 2.1.1 正規分布

1つの実数値 $`z`$ が正規分布に従うことを

$$
z \sim \mathcal{N}(\mu, \sigma^2)
$$

と書きます。$`\mu`$ は平均、$`\sigma^2`$ は分散です。
平均は「中心がどこか」、分散は「どの程度ばらつくか」を表します。

### 2.1.2 多変量正規分布

複数の値をまとめたベクトルについては、

$$
\mathbf{z} = [z_1, \ldots, z_n]^T
$$

$$
\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)
$$

と書けます。$`\boldsymbol{\mu}`$ は平均ベクトル、$`\Sigma`$ は共分散行列です。
共分散行列の対角成分は各値の分散、非対角成分は異なる値どうしの関係を表します。

GPでは、1点ずつ独立に予測するのではなく、この**複数点間の相関を含む分布**
を扱うことが重要です。

## 2.2 Gaussian Process とは

Gaussian Process（GP）は、**未知の関数に対する確率分布を表すベイズ的モデル**です。
より厳密には、任意の有限個の入力点を選んだとき、その点における関数値が
多変量正規分布に従うような確率過程を Gaussian Process と呼びます。

通常の回帰では、例えば

$$
y = w^T x + b
$$

のように有限個のパラメータ $`w`$, $`b`$ を学習します。

一方、GPでは未知関数 $`f(x)`$ に対して

$$
f(x) \sim \mathcal{GP}(m(x), k(x,x'))
$$

という事前分布を仮定します。

ここで

- $`m(x)`$ は平均関数
- $`k(x,x')`$ は共分散関数（kernel）

です。

GPの重要な特徴は、任意の有限個の入力点

$$
X = \{x_1,\ldots,x_n\}
$$

における関数値ベクトル

$$
\mathbf{f} = [f(x_1),\ldots,f(x_n)]^T
$$

が多変量正規分布に従うことです。

$$
\mathbf{f} \sim \mathcal{N}(\mathbf{m}, K)
$$

ここで

$$
\mathbf{m}_i = m(x_i)
$$

$$
K_{ij} = k(x_i,x_j)
$$

です。

## 2.3 GPを直感的に理解する

GPでは「近い入力点では似た関数値を取りやすい」といった仮定を kernel によって表現します。

例えば、温度が 500 ℃ と 501 ℃ の条件では、目的特性も似る可能性が高いと考えます。

このとき、入力点同士の類似度を

$$
k(x,x')
$$

で表します。

類似度が大きい点では、関数値も強く相関すると仮定します。

つまり GP は、

```text
入力空間の近さ・類似性
        ↓
Kernel
        ↓
関数値同士の共分散
        ↓
未観測点の予測分布
```

という構造になっています。

## 2.4 観測ノイズ

実験値や測定値にはノイズが含まれることがあります。

一般に

$$
y_i = f(x_i) + \epsilon_i
$$

とし、

$$
\epsilon_i \sim \mathcal{N}(0,\sigma_n^2)
$$

と仮定します。

このとき観測値 $`y`$ の共分散行列は

$$
K_y = K + \sigma_n^2 I
$$

となります。

ここで $`I`$ は単位行列です。

ノイズを入れることで、すべての観測点を完全に補間するのではなく、観測誤差を許容した回帰ができます。

## 2.5 事前分布と事後分布

GPでは、データを見る前には未知関数に対して事前分布を持ちます。

$$
f \sim \mathcal{GP}(m,k)
$$

観測データ

$$
\mathcal{D}=\{X,y\}
$$

を得ると、ベイズ則により関数の分布を更新します。

更新後の分布を posterior（事後分布）と呼びます。概念的には Bayes 則

$$
p(f \mid \mathcal{D}) \propto p(\mathcal{D} \mid f) p(f)
$$

によって、prior $`p(f)`$ と likelihood $`p(\mathcal{D}\mid f)`$ を組み合わせ、
posterior $`p(f\mid\mathcal{D})`$ を得ます。

BOでは、この posterior の予測平均だけでなく予測不確実性も使って、
次の候補点を選択します。

## 2.6 予測分布の導出

学習入力を $`X`$、予測したい点を $`x_*`$ とします。

学習点における観測値と、予測点における未知関数値の同時分布は

$$
\begin{bmatrix}
y\\f_*
\end{bmatrix}
\sim
\mathcal{N}
\left(
\begin{bmatrix}
m(X)\\m(x_*)
\end{bmatrix},
\begin{bmatrix}
K + \sigma_n^2 I & k_*\\
k_*^T & k_{**}
\end{bmatrix}
\right)
$$

と書けます。

ここで

$$
k_* =
\begin{bmatrix}
k(x_1,x_*)\\
\vdots\\
k(x_n,x_*)
\end{bmatrix}
$$

$$
k_{**}=k(x_*,x_*)
$$

です。

学習点と予測点は別々の分布ではなく、kernel による相関を含む
**1つの同時正規分布**として表されています。観測値 $`y`$ が得られたという条件を
付けて予測点の分布を求める操作を conditioning（条件付け）と呼びます。

多変量正規分布の条件付き分布から、posterior mean は

$$
\mu(x_*)
=
m(x_*)
+
k_*^T(K+\sigma_n^2I)^{-1}(y-m(X))
$$

となります。

posterior variance は

$$
\sigma^2(x_*)
=
k_{**}
-
k_*^T(K+\sigma_n^2I)^{-1}k_*
$$

です。

平均の式では、観測値と prior mean の差 $`y-m(X)`$ が、学習点と予測点の
共分散を通して予測へ反映されます。分散の式では、prior variance $`k_{**}`$ から
観測によって減少した不確実性を差し引いています。

数式には $`(K+\sigma_n^2I)^{-1}`$ が現れますが、実装では通常、
逆行列を明示的に作らず Cholesky 分解などを用いて線形方程式を解きます。

平均関数を 0 とすれば

$$
\mu(x_*) = k_*^T(K+\sigma_n^2I)^{-1}y
$$

と簡略化できます。

## 2.7 予測平均の意味

posterior mean

$$
\mu(x_*)
$$

は、既知の観測値を kernel による類似度で重み付けした予測と解釈できます。

予測点 $`x_*`$ と強く相関する観測点ほど、その予測に大きく影響します。

そのため、kernel の設計は予測形状に直接影響します。

## 2.8 予測分散の意味

posterior variance

$$
\sigma^2(x_*)
$$

は「その点についてモデルがどれだけ不確かか」を表します。

観測点の近くでは一般に不確実性が小さくなり、観測点から離れると大きくなります。

ここで $`\sigma^2(x_*)`$ は、潜在的な関数値 $`f(x_*)`$ の posterior variance です。
将来のノイズ付き観測を

$$
y_* = f(x_*) + \epsilon_*
$$

とすると、独立な Gaussian noise のもとでは

$$
\operatorname{Var}(y_* \mid \mathcal{D})
=
\operatorname{Var}(f_* \mid \mathcal{D}) + \sigma_n^2
$$

となります。**関数そのものの不確実性と測定ノイズは別のもの**です。

また、Gaussian likelihood と固定された hyperparameter のもとでは、
posterior covariance は観測入力、kernel、noise などから決まり、
観測値 $`y`$ そのものには依存しません。posterior mean は観測値に依存します。

この予測分散が BO にとって非常に重要です。

通常の点予測モデルでは

```text
x → y_hat
```

ですが、GPでは

```text
x → Normal(mu(x), sigma^2(x))
```

となります。

このため、

- 予測平均が高い場所
- 不確実性が高い場所

の両方を評価できます。

## 2.9 GPと補間

ノイズがほぼ 0 の Exact GP では、学習点付近で posterior variance が非常に小さくなり、観測点をほぼ補間します。

ただし実験データに測定誤差がある場合、ノイズを 0 と仮定しすぎると過学習につながる可能性があります。

そのため実務では、

- ノイズをモデルに推定させる
- 既知の測定分散 `train_Yvar` を与える

といった扱いが重要です。

## 2.10 Hyperparameter

Kernel には lengthscale や outputscale などの hyperparameter があります。

例えば RBF kernel では

$$
k(x,x')
=
\sigma_f^2
\exp\left(
-\frac{\|x-x'\|^2}{2\ell^2}
\right)
$$

と書けます。

ここで

- $`\ell`$ : lengthscale
- $`\sigma_f^2`$ : outputscale

です。

これらは通常、データから学習します。

## 2.11 Marginal Log Likelihood

Exact GP では hyperparameter を Marginal Log Likelihood（MLL）最大化によって推定することが一般的です。

観測値 $`y`$ の周辺尤度は

$$
p(y\mid X,\theta)
=
\mathcal{N}(y; m(X), K_\theta + \sigma_n^2I)
$$

です。

その対数は

$$
\log p(y\mid X,\theta)
=
-\frac{1}{2}(y-m)^TK_y^{-1}(y-m)
-\frac{1}{2}\log|K_y|
-\frac{n}{2}\log(2\pi)
$$

となります。

第1項は観測データへの適合を表します。第2項の $`\log|K_y|`$ は、
モデルが許す関数の広がりに関係する complexity term です。
第3項は正規分布の正規化定数です。

したがって周辺尤度は、単純な訓練誤差だけでなく、
**データへの適合とモデルの複雑さの両方**を考慮します。

そのため、GPは単純な誤差最小化とは異なる形で hyperparameter を学習します。

## 2.12 Cholesky 分解と計算量

実装上は、

$$
(K+\sigma_n^2I)^{-1}
$$

を明示的に逆行列として計算するのではなく、通常 Cholesky 分解を用いて線形方程式を解きます。

密な $`n \times n`$ 共分散行列を直接扱う標準的な Exact GP では、
Cholesky 分解が主要な計算になります。その代表的な計算量は概ね

$$
\mathcal{O}(n^3)
$$

メモリ量は

$$
\mathcal{O}(n^2)
$$

です。

このためデータ数 $`n`$ が非常に大きくなると Exact GP は重くなります。

その場合は Variational GP などの近似手法を検討します。

## 2.13 ARD

入力次元ごとに異なる lengthscale を持たせることを Automatic Relevance Determination（ARD）と呼びます。

$$
k(x,x')
=
\sigma_f^2
\exp\left(
-\frac{1}{2}
\sum_{j=1}^d
\frac{(x_j-x'_j)^2}{\ell_j^2}
\right)
$$

各次元に対して $`\ell_j`$ を持つため、変数ごとの変化スケールを学習できます。

小さな lengthscale は、その変数方向で関数が大きく変化することを意味します。

ただし、単純に lengthscale の大小だけで因果的な変数重要度を判断すべきではありません。

## 2.14 「ノンパラメトリック」とは何か

GPはしばしば **nonparametric model** と呼ばれます。ただし、
「パラメータが存在しない」という意味ではありません。
lengthscale、outputscale、noise level などの有限個の hyperparameter は存在します。

ここでいう nonparametric は、未知関数そのものを線形回帰の係数ベクトルのような
固定個数のパラメータだけで表現する必要がなく、データが増えるにつれて
扱う関数値や共分散構造も拡張される、という意味です。

## 2.15 SingleTaskGP との対応

robotorchan の標準 GP は `SingleTaskGP` です。

基本的な流れは次の通りです。

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import SingleTaskGP

train_X = torch.rand(20, 2, dtype=torch.double)
train_Y = torch.sin(train_X[:, :1] * 6.0)

model = SingleTaskGP(train_X=train_X, train_Y=train_Y)

mll = model.make_mll()
fit_gpytorch_mll(mll)

posterior = model.posterior(torch.rand(5, 2, dtype=torch.double))

mean = posterior.mean
variance = posterior.variance
```

理論上の

$$
\mu(x),\sigma^2(x)
$$

に対応するものが `posterior.mean` と `posterior.variance` です。

## 2.16 Multi-output との違い

標準 `SingleTaskGP` は基本的に単一タスクの GP です。

複数の出力や複数の関連タスクを扱う場合は、

- `ModelListGP`
- `MultiTaskGP`
- `KroneckerMultiTaskGP`

などを利用します。

これらの違いは主に「出力・タスク間の共分散をどう扱うか」にあります。

## 2.17 GPの強みと弱み

### 強み

- 少量データでも比較的使いやすい
- 不確実性を自然に得られる
- kernel を通して事前知識を組み込める
- BOとの相性が良い

### 弱み

- Exact GP はデータ数が増えると重い
- 高次元では学習が難しくなりやすい
- kernel の仮定が不適切だと性能が落ちる
- 外挿には基本的に弱い

## 2.18 この章で覚えておくこと

最初は次の流れを押さえると、GPの全体像を見失いにくくなります。

~~~text
入力点
  ↓
kernel で点どうしの共分散を定義
  ↓
複数点の関数値に joint Gaussian prior を置く
  ↓
観測データで conditioning
  ↓
posterior mean + posterior covariance
  ↓
BO が予測値と不確実性を利用
~~~

GPの中心は単に「滑らかな曲線を引くこと」ではなく、
**入力点間の共分散構造を通じて、未知関数についての確率分布を更新すること**です。

## 2.19 次に読む章

GPの性質は kernel によって大きく決まります。

次章では、kernel が何を意味し、RBF・Matérn・ARD などが予測にどう影響するかを説明します。

→ [3. Kernel](03_kernel.md)

## 参考文献

1. Rasmussen, C. E., and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.
2. Williams, C. K. I., and Rasmussen, C. E. (1996).
   Gaussian processes for regression.
   *Advances in Neural Information Processing Systems 8*.
3. Rasmussen, C. E. (2004).
   Gaussian processes in machine learning.
   *Advanced Lectures on Machine Learning*, 63-71.
