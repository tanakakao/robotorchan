# 2. Gaussian Process

## 2.1 Gaussian Process とは

Gaussian Process（GP）は、**関数そのものに確率分布を置くベイズ的な回帰モデル**です。

通常の回帰では、例えば

\[
y = w^T x + b
\]

のように有限個のパラメータ `w`, `b` を学習します。

一方、GPでは未知関数 `f(x)` に対して

\[
f(x) \sim \mathcal{GP}(m(x), k(x,x'))
\]

という事前分布を仮定します。

ここで

- `m(x)` は平均関数
- `k(x,x')` は共分散関数（kernel）

です。

GPの重要な特徴は、任意の有限個の入力点

\[
X = \{x_1,\ldots,x_n\}
\]

における関数値ベクトル

\[
\mathbf{f} = [f(x_1),\ldots,f(x_n)]^T
\]

が多変量正規分布に従うことです。

\[
\mathbf{f} \sim \mathcal{N}(\mathbf{m}, K)
\]

ここで

\[
\mathbf{m}_i = m(x_i)
\]

\[
K_{ij} = k(x_i,x_j)
\]

です。

## 2.2 GPを直感的に理解する

GPでは「近い入力点では似た関数値を取りやすい」といった仮定を kernel によって表現します。

例えば、温度が 500 ℃ と 501 ℃ の条件では、目的特性も似る可能性が高いと考えます。

このとき、入力点同士の類似度を

\[
k(x,x')
\]

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

## 2.3 観測ノイズ

実験値や測定値にはノイズが含まれることがあります。

一般に

\[
y_i = f(x_i) + \epsilon_i
\]

とし、

\[
\epsilon_i \sim \mathcal{N}(0,\sigma_n^2)
\]

と仮定します。

このとき観測値 `y` の共分散行列は

\[
K_y = K + \sigma_n^2 I
\]

となります。

ここで `I` は単位行列です。

ノイズを入れることで、すべての観測点を完全に補間するのではなく、観測誤差を許容した回帰ができます。

## 2.4 事前分布と事後分布

GPでは、データを見る前には未知関数に対して事前分布を持ちます。

\[
f \sim \mathcal{GP}(m,k)
\]

観測データ

\[
\mathcal{D}=\{X,y\}
\]

を得ると、ベイズ則により関数の分布を更新します。

更新後の分布を posterior と呼びます。

BOでは、この posterior を使って次の候補点を選択します。

## 2.5 予測分布の導出

学習入力を `X`、予測したい点を `x_*` とします。

学習点における観測値と、予測点における未知関数値の同時分布は

\[
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
\]

と書けます。

ここで

\[
k_* =
\begin{bmatrix}
k(x_1,x_*)\\
\vdots\\
k(x_n,x_*)
\end{bmatrix}
\]

\[
k_{**}=k(x_*,x_*)
\]

です。

多変量正規分布の条件付き分布から、posterior mean は

\[
\mu(x_*)
=
m(x_*)
+
k_*^T(K+\sigma_n^2I)^{-1}(y-m(X))
\]

となります。

posterior variance は

\[
\sigma^2(x_*)
=
k_{**}
-
k_*^T(K+\sigma_n^2I)^{-1}k_*
\]

です。

平均関数を 0 とすれば

\[
\mu(x_*) = k_*^T(K+\sigma_n^2I)^{-1}y
\]

と簡略化できます。

## 2.6 予測平均の意味

posterior mean

\[
\mu(x_*)
\]

は、既知の観測値を kernel による類似度で重み付けした予測と解釈できます。

予測点 `x_*` と強く相関する観測点ほど、その予測に大きく影響します。

そのため、kernel の設計は予測形状に直接影響します。

## 2.7 予測分散の意味

posterior variance

\[
\sigma^2(x_*)
\]

は「その点についてモデルがどれだけ不確かか」を表します。

観測点の近くでは一般に不確実性が小さくなり、観測点から離れると大きくなります。

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

## 2.8 GPと補間

ノイズがほぼ 0 の Exact GP では、学習点付近で posterior variance が非常に小さくなり、観測点をほぼ補間します。

ただし実験データに測定誤差がある場合、ノイズを 0 と仮定しすぎると過学習につながる可能性があります。

そのため実務では、

- ノイズをモデルに推定させる
- 既知の測定分散 `train_Yvar` を与える

といった扱いが重要です。

## 2.9 Hyperparameter

Kernel には lengthscale や outputscale などの hyperparameter があります。

例えば RBF kernel では

\[
k(x,x')
=
\sigma_f^2
\exp\left(
-\frac{\|x-x'\|^2}{2\ell^2}
\right)
\]

と書けます。

ここで

- `\ell` : lengthscale
- `\sigma_f^2` : outputscale

です。

これらは通常、データから学習します。

## 2.10 Marginal Log Likelihood

Exact GP では hyperparameter を Marginal Log Likelihood（MLL）最大化によって推定することが一般的です。

観測値 `y` の周辺尤度は

\[
p(y\mid X,\theta)
=
\mathcal{N}(y; m(X), K_\theta + \sigma_n^2I)
\]

です。

その対数は

\[
\log p(y\mid X,\theta)
=
-\frac{1}{2}(y-m)^TK_y^{-1}(y-m)
-\frac{1}{2}\log|K_y|
-\frac{n}{2}\log(2\pi)
\]

となります。

この式には、

- データへの適合
- モデル複雑度へのペナルティ

の両方が含まれています。

そのため、GPは単純な誤差最小化とは異なる形で hyperparameter を学習します。

## 2.11 Cholesky 分解と計算量

実装上は、

\[
(K+\sigma_n^2I)^{-1}
\]

を明示的に逆行列として計算するのではなく、通常 Cholesky 分解を用いて線形方程式を解きます。

Exact GP の主要計算量は概ね

\[
\mathcal{O}(n^3)
\]

メモリ量は

\[
\mathcal{O}(n^2)
\]

です。

このためデータ数 `n` が非常に大きくなると Exact GP は重くなります。

その場合は Variational GP などの近似手法を検討します。

## 2.12 ARD

入力次元ごとに異なる lengthscale を持たせることを Automatic Relevance Determination（ARD）と呼びます。

\[
k(x,x')
=
\sigma_f^2
\exp\left(
-\frac{1}{2}
\sum_{j=1}^d
\frac{(x_j-x'_j)^2}{\ell_j^2}
\right)
\]

各次元に対して `\ell_j` を持つため、変数ごとの変化スケールを学習できます。

小さな lengthscale は、その変数方向で関数が大きく変化することを意味します。

ただし、単純に lengthscale の大小だけで因果的な変数重要度を判断すべきではありません。

## 2.13 SingleTaskGP との対応

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

\[
\mu(x),\sigma^2(x)
\]

に対応するものが `posterior.mean` と `posterior.variance` です。

## 2.14 Multi-output との違い

標準 `SingleTaskGP` は基本的に単一タスクの GP です。

複数の出力や複数の関連タスクを扱う場合は、

- `ModelListGP`
- `MultiTaskGP`
- `KroneckerMultiTaskGP`

などを利用します。

これらの違いは主に「出力・タスク間の共分散をどう扱うか」にあります。

## 2.15 GPの強みと弱み

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

## 2.16 次に読む章

GPの性質は kernel によって大きく決まります。

次章では、kernel が何を意味し、RBF・Matérn・ARD などが予測にどう影響するかを説明します。

→ [3. Kernel](03_kernel.md)
