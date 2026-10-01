# 3. Kernel

## 3.1 Kernel とは

Gaussian Process（GP）では、入力 $`x`$ と $`x'`$ の関係を kernel（共分散関数）

$$
k(x,x')
$$

によって表します。

ここで重要なのは、kernel を単に「入力どうしの類似度」と考えるだけでは不十分なことです。
GPにおける kernel は、2つの入力に対応する**関数値の共分散**を定義します。

Kernel は、2つの入力点における関数値

$$
f(x), f(x')
$$

がどの程度一緒に変化しやすいか、つまり共分散を定義します。

$$
\operatorname{Cov}(f(x),f(x'))=k(x,x')
$$

GPでは、**どのような関数を事前にもっともらしいと考えるか**の多くが kernel によって決まります。

したがって、モデル名が同じ `SingleTaskGP` でも kernel の選択・hyperparameter によって予測の性質は大きく変わります。

## 3.2 Kernel matrix

学習点

$$
X=\{x_1,\ldots,x_n\}
$$

に対して kernel matrix $`K`$ を

$$
K_{ij}=k(x_i,x_j)
$$

と定義します。

$$
K=
\begin{bmatrix}
k(x_1,x_1) & \cdots & k(x_1,x_n)\\
\vdots & \ddots & \vdots\\
k(x_n,x_1) & \cdots & k(x_n,x_n)
\end{bmatrix}
$$

GPではこの行列が学習点間の共分散構造を表します。

有効な covariance kernel は、任意の有限入力集合に対してこの行列が
**positive semidefinite（PSD; 半正定値）**になる必要があります。つまり任意の
実ベクトル $`a`$ に対して

$
a^T K a \ge 0
$

が成立しなければなりません。

これは単なる数学上の形式ではありません。$`a^TKa`$ は kernel が定める
線形結合の分散に対応するため、負になってはいけません。したがって、
「似ていそうな値を返す関数」なら何でも GP kernel にできるわけではありません。

対称な PSD kernel を使うことで、$`K`$ を有効な共分散行列として扱えます。

## 3.3 Stationary・isotropic という考え方

Kernelを読むときは、「何に依存して共分散が決まるか」を区別すると理解しやすくなります。

**stationary kernel** は、絶対的な位置ではなく入力間の差 $`x-x'`$ に依存します。

$
k(x,x') = \kappa(x-x')
$

さらに **isotropic kernel** は、その差の方向ではなく距離だけに依存します。

$
k(x,x') = \kappa(\|x-x'\|)
$

RBFや標準的なMatérn kernelは代表的な stationary kernel であり、
単一lengthscaleでユークリッド距離を使う形は isotropic です。
ARDでは次元ごとに異なるlengthscaleを使うため、単純な isotropic 距離から拡張されます。

一方、入力位置によって振る舞い自体が変わる問題では stationary 仮定が
不十分な場合があります。そのような問題は後の Nonstationary GP で扱います。

## 3.4 距離と類似性

多くの代表的 kernel は入力間距離

$$
r=\|x-x'\|
$$

が小さいほど大きな値を取ります。

直感的には

```text
入力が近い
   ↓
Kernel値が大きい
   ↓
関数値の相関が強い
   ↓
一方の観測が他方の予測に強く影響する
```

という関係です。

ただし、すべての kernel が単純なユークリッド距離だけで定義されるわけではありません。カテゴリ変数、タスク、階層構造、グラフなどに応じて異なる類似度を設計できます。

### Kernel値と相関係数は同じではない

$`k(x,x')`$ は covariance であり、一般には相関係数そのものではありません。
対応する相関係数は、分散が正なら

$
\rho(x,x')
=
\frac{k(x,x')}
{\sqrt{k(x,x)k(x',x')}}
$

です。例えば outputscale を大きくすると covariance の絶対値は変わりますが、
正規化された correlation は変わらない場合があります。

## 3.5 RBF kernel

代表的な kernel の1つが RBF（Radial Basis Function）kernel、あるいは Squared Exponential kernel です。

$$
k_{\mathrm{RBF}}(x,x')
=
\sigma_f^2
\exp\left(
-\frac{\|x-x'\|^2}{2\ell^2}
\right)
$$

ここで

- $`\ell`$ : lengthscale
- $`\sigma_f^2`$ : outputscale

です。

RBF kernel は非常に滑らかな関数を仮定します。

数学的には、RBF GP のサンプル関数は非常に高い滑らかさを持ちます。

そのため、目的関数が滑らかに変化すると考えられる場合には自然ですが、急激な変化や粗い応答を持つ場合には過度に滑らかな仮定となることがあります。

## 3.6 Matérn kernel

ベイズ最適化では Matérn kernel がよく使われます。

Matérn kernel は RBF より滑らかさの仮定が弱く、実務上柔軟です。

一般形は

$$
k_{\nu}(r)
=
\sigma_f^2
\frac{2^{1-\nu}}{\Gamma(\nu)}
\left(\frac{\sqrt{2\nu}r}{\ell}\right)^\nu
K_\nu\left(\frac{\sqrt{2\nu}r}{\ell}\right)
$$

です。

ここで $`K_\nu`$ は第2種変形 Bessel 関数です。

実際には $`\nu=3/2`$ や $`\nu=5/2`$ がよく使われます。

### Matérn 3/2

$$
k(r)=
\sigma_f^2
\left(1+\frac{\sqrt{3}r}{\ell}\right)
\exp\left(-\frac{\sqrt{3}r}{\ell}\right)
$$

### Matérn 5/2

$$
k(r)=
\sigma_f^2
\left(
1+\frac{\sqrt{5}r}{\ell}
+\frac{5r^2}{3\ell^2}
\right)
\exp\left(-\frac{\sqrt{5}r}{\ell}\right)
$$

`\nu` が大きいほど滑らかな関数を仮定します。

## 3.7 Lengthscale

Lengthscale $`\ell`$ は、関数が入力空間でどの程度の距離スケールで変化するかを表します。

### 小さい lengthscale

$$
\ell \downarrow
$$

では、少し入力が変わっただけでも相関が急速に低下します。

つまり、関数が細かく変化することを許します。

### 大きい lengthscale

$$
\ell \uparrow
$$

では、離れた点でも高い相関を保ちます。

つまり、より滑らかでゆっくり変化する関数を仮定します。

Lengthscale は「入力変数の単位」に強く依存するため、入力のスケーリングは重要です。

## 3.8 Outputscale

Outputscale $`\sigma_f^2`$ は関数値がどの程度の振幅で変化するかを表します。

Kernel を

$$
k(x,x')=\sigma_f^2\tilde{k}(x,x')
$$

と考えると、$`\tilde{k}`$ が相関構造、$`\sigma_f^2`$ が全体の変動スケールを担当します。

出力の標準化は、この hyperparameter の推定を安定化させる上でも重要です。

## 3.9 ARD

複数次元入力では、次元ごとに異なる lengthscale を持たせることがあります。

$$
r^2(x,x')
=
\sum_{j=1}^d
\frac{(x_j-x'_j)^2}{\ell_j^2}
$$

これを Automatic Relevance Determination（ARD）と呼びます。

各変数が異なるスケールで目的関数へ影響する場合に有効です。

例えば

```text
温度      : lengthscale = 0.15
圧力      : lengthscale = 0.70
保持時間  : lengthscale = 1.20
```

であれば、正規化された入力空間では温度方向により細かい変化を許していると解釈できます。

ただし、lengthscale はデータ分布・他変数との相関・prior の影響も受けるため、単純な「変数重要度」と同一視しないことが重要です。

## 3.10 入力の正規化

Kernel は入力間距離を使うことが多いため、入力スケールが極端に異なると問題が起きます。

例えば

```text
温度: 300 ～ 1500
圧力: 0.1 ～ 1.0
```

をそのまま入れると、ユークリッド距離は温度に支配されやすくなります。

BOでは一般に、各連続変数を

$$
[0,1]
$$

付近へ正規化して扱うことが推奨されます。

BoTorch では input transform として `Normalize` が利用できます。

## 3.11 出力の標準化

出力についても

$$
y' = \frac{y-\mu_y}{\sigma_y}
$$

のように標準化すると、GP hyperparameter の初期値や prior と整合しやすくなります。

BoTorch では outcome transform として `Standardize` が利用できます。

## 3.12 Kernel の和

Kernel 同士を加算して新しい kernel を作ることができます。
$`k_1`$ と $`k_2`$ が有効な PSD kernel なら、その和も PSD なので
GPの covariance kernel として利用できます。

$$
k(x,x')=k_1(x,x')+k_2(x,x')
$$

これは関数を

$$
f(x)=f_1(x)+f_2(x)
$$

という複数成分の和としてモデル化することに対応します。

例えば

```text
長期的な滑らかな傾向
+
短周期の局所変動
```

のような構造を表現できます。

Additive GP は、この加法的構造を高次元問題に利用します。

## 3.13 Kernel の積

Kernel の積も有効な kernel です。$`k_1`$ と $`k_2`$ が PSD kernel なら、
点ごとの積によって得られる Gram matrix も Schur product theorem により PSD です。

$$
k(x,x')=k_1(x,x')k_2(x,x')
$$

積 kernel は、複数の構造が同時に成立するような共分散を表せます。

Multi-task GP では、入力 covariance と task covariance を組み合わせる際に積構造が重要になります。

## 3.14 Multi-task kernel

Multi-task GP では、単純な入力 $`x`$ だけでなく task $`t`$ も考えます。

典型的には

$$
k((x,t),(x',t'))
=
k_X(x,x')\,k_T(t,t')
$$

のような構造を考えます。

ここで

- $`k_X`$ : 入力間 covariance
- $`k_T`$ : task 間 covariance

です。

この考え方により、あるタスクの観測を別タスクの予測に利用できます。

`MultiTaskGP` や `KroneckerMultiTaskGP` の理論を理解する際の基礎になります。

## 3.15 Kronecker structure

全タスクが同じ入力点で観測される block design では、共分散行列に

$$
K = K_X \otimes K_T
$$

という Kronecker product 構造が現れます。

これは `KroneckerMultiTaskGP` の重要な考え方です。

単純な SingleTask GP の

$$
K_X
$$

に、タスク covariance

$$
K_T
$$

を追加した形と考えると理解しやすくなります。

## 3.16 Categorical kernel

カテゴリ変数では、数値の差をそのままユークリッド距離として扱うことは一般に適切ではありません。

例えば

```text
触媒A = 0
触媒B = 1
触媒C = 2
```

と符号化しても、

```text
A と C の距離は A と B の2倍
```

という意味はありません。

Mixed GP ではカテゴリ変数に適した covariance 構造を用いて、連続変数とカテゴリ変数を組み合わせます。

robotorchan では `MixedSingleTaskGP` がこの用途に対応します。

## 3.17 Conditional / Hierarchical kernel

探索空間によっては、親変数の値によって有効な子変数が変わります。

例えば

```text
製法 = A → 温度、時間が有効
製法 = B → 圧力、流量が有効
```

という場合です。

通常の距離では「そもそも存在しない変数」を適切に扱えません。

Hierarchical kernel は、変数が有効かどうかも含めて covariance を設計します。

robotorchan では

- `HierarchicalConditionalKernelGP`
- `HierarchicalConditionalKernelMultiTaskGP`

がこの考え方に対応します。

## 3.18 Kernel とモデル選択

GPモデルの違いのかなりの部分は、**どのような covariance structure を仮定するか**として理解できます。

```text
SingleTaskGP
  └─ 入力間 covariance

MixedSingleTaskGP
  └─ 連続 + category covariance

MultiTaskGP
  └─ 入力 covariance × task covariance

KroneckerMultiTaskGP
  └─ K_X ⊗ K_T

HierarchicalConditionalKernelGP
  └─ 条件付き探索空間を考慮した covariance

Contextual GP
  └─ context 間構造を含む covariance
```

この視点を持つと、新しい GP モデルを「完全に別物」として覚える必要がなくなります。

## 3.19 Kernel とベイズ最適化

Kernel は surrogate model の予測平均だけでなく、posterior uncertainty にも影響します。

そのため kernel の選択は acquisition function にも間接的に影響します。

例えば lengthscale が極端に大きい場合、広い領域でモデルが似た予測をし、不確実性も過度に小さくなることがあります。

逆に lengthscale が極端に小さい場合、観測点から少し離れただけで不確実性が大きくなり、探索を過度に促進する場合があります。

つまり、kernel learning は単なる予測問題ではなく、**BO の探索戦略そのものに影響する**重要な要素です。

## 3.20 実務上の基本方針

通常の連続変数 BO では Matérn 系 kernel は有力なベースラインです。
ただし「常に Matérn が最良」という意味ではありません。既知の周期性、加法構造、
非定常性、カテゴリ構造などがある場合は、それを表す kernel や専用モデルを
比較対象にする方が自然です。

目安としては以下です。

| 問題構造 | 検討する考え方 |
|---|---|
| 通常の連続変数 | Matérn / RBF |
| 変数ごとに感度が違う | ARD |
| カテゴリを含む | Mixed / categorical kernel |
| タスク間相関がある | task covariance |
| 高次元で加法構造がある | additive kernel |
| 条件付き変数がある | hierarchical kernel |
| Context がある | contextual covariance |

## 3.21 robotorchan との関係

robotorchan では、利用者が毎回 kernel をゼロから組み立てなくても代表的な問題構造を扱えるよう、モデル単位で共通 API を提供します。

一方で、各モデルの違いを理解するためには

> 「このモデルは、どの入力・出力・タスク間の covariance をどう仮定しているか」

を見ることが重要です。

これは今後説明する Mixed、Multi-Fidelity、Multi-task、Hierarchical、Contextual GP の共通の見方になります。

## 3.21 この章で覚えておくこと

Kernelについて最初に押さえるべき点は、次の関係です。

~~~text
問題についての仮定
  ↓
kernel が入力間の covariance を定義
  ↓
kernel matrix K
  ↓
GP prior / posterior の相関構造
  ↓
予測平均・不確実性
  ↓
Bayesian Optimization の候補選択にも影響
~~~

Kernelは単なる距離関数ではなく、**未知関数がどのように変化すると考えるかを
共分散構造として記述するもの**です。RBF、Matérn、ARDなどの名前を暗記するより、
「このkernelはどのような関数を事前にもっともらしいと考えるのか」を読むことが重要です。

## 3.22 次に読む章

GPが予測平均と不確実性を返す仕組みを理解したら、次はそれらを使って「どの点を次に評価するか」を決める必要があります。

次章では acquisition function を説明します。

→ [4. Acquisition Function](04_acquisition_function.md)

## 参考文献

1. Rasmussen, C. E., and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.
2. Stein, M. L. (1999).
   *Interpolation of Spatial Data: Some Theory for Kriging*. Springer.
3. GPyTorch documentation.
   Kernel implementations used by the upstream GP stack.
