# 3. Kernel

本章では、GPの共分散を設計するために、カーネルの妥当性、長さ尺度、
滑らかさ、和と積の意味を学びます。[GPの基本章](02_gaussian_process.md)の
事前分布・事後分布と、行列の半正定値性の基礎を前提とします。
以下では実数値の関数を扱い、入力を $x,x'\in\mathbb R^d$ とします。

## 3.1 Kernel とは

Gaussian Process（GP）では、入力 $`x`$ と $`x'`$ の関係を kernel（共分散関数）

$$
k(x,x')
$$

によって表します。

ここで重要なのは、kernel を単に「入力どうしの類似度」と考えるだけでは不十分なことです。
GPにおける kernel は、2つの入力に対応する**関数値の共分散**を定義します。

具体的には、2つの入力点における関数値

$$
f(x), f(x')
$$

がどの程度一緒に変化しやすいか、つまり共分散を定義します。

$$
\operatorname{Cov}(f(x),f(x'))=k(x,x')
$$

GPでは、**どのような関数を事前にもっともらしいと考えるか**の多くが kernel によって決まります。

したがって、同じ単一タスクGPでも、カーネルとそのハイパーパラメータによって予測の性質は大きく変わります。

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

GPではこの $n\times n$ 行列が学習点間の共分散構造を表します。

有効な covariance kernel は、任意の有限入力集合に対してこの行列が
**positive semidefinite（PSD; 半正定値）**になる必要があります。つまり任意の
実ベクトル $`a`$ に対して

$$
a^T K a \ge 0
$$

が成立しなければなりません。

これは単なる数学上の形式ではありません。$`a^TKa`$ は kernel が定める
線形結合の分散に対応するため、負になってはいけません。したがって、
「似ていそうな値を返す関数」なら何でも GP kernel にできるわけではありません。

例えば特徴写像 $\phi:\mathcal X\to\mathbb R^p$ により
$k(x,x')=\phi(x)^\top\phi(x')$ と置くと、
$a^\top Ka=\|\sum_i a_i\phi(x_i)\|^2\ge0$ です。
この構成は線形カーネルを含み、類似度の候補が共分散として妥当であることを示す一つの方法です。

対称なPSDカーネルを使うことで、$K$ を有効な共分散行列として扱えます。
半正定値性は可逆性までは保証しません。同じ入力の重複などで $K$ が特異でも、
独立な正の観測ノイズ分散を加えた $K+\sigma_n^2I$ は正定値になります。
ノイズを加える理由と数値計算上の安定化については[GP章](02_gaussian_process.md)を参照してください。

## 3.3 Stationary・isotropic という考え方

Kernelを読むときは、「何に依存して共分散が決まるか」を区別すると理解しやすくなります。

**stationary kernel** は、絶対的な位置ではなく入力間の差 $`x-x'`$ に依存します。

$$
k(x,x') = \kappa(x-x')
$$

さらに **isotropic kernel** は、その差の方向ではなく距離だけに依存します。

$$
k(x,x') = \kappa(\|x-x'\|)
$$

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

$$
\rho(x,x')
=
\frac{k(x,x')}
{\sqrt{k(x,x)k(x',x')}}
$$

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

数学的には、RBF共分散は任意の階数で平均二乗微分可能なGPを定めます。

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

ここで $r=\|x-x'\|\ge0$、$\ell>0$ は長さ尺度、$\sigma_f^2>0$ は分散、
$\nu>0$ は滑らかさを制御するパラメータです。
$\Gamma$ はガンマ関数、$K_\nu$ は第2種変形Bessel関数であり、
第3.2節の共分散行列 $K$ とは異なる記号です。
$r=0$ では連続な極限を取り、$k_\nu(0)=\sigma_f^2$ とします。

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

$\nu$ が大きいほど滑らかな関数を仮定します。正の整数 $p$ に対し、
Matérn共分散が定める過程は $\nu>p$ のとき $p$ 回平均二乗微分可能です。
したがって $\nu=3/2$ は1回、$\nu=5/2$ は2回の平均二乗微分を許します。
長さ尺度 $\ell$ は変化する距離を調整し、$\nu$ は微分可能性を調整するため、
両者は別の性質を表します。

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

つまり、より長い距離にわたってゆっくり変化する関数を仮定します。

例えば分散1のRBFでは、距離が $r=\ell$ のとき相関は
$e^{-1/2}\simeq0.607$、$r=2\ell$ では $e^{-2}\simeq0.135$ です。
長さ尺度は相関の減衰を読むための基準であり、「この距離で相関がなくなる」という境界ではありません。
また、RBFでは $\ell$ を変えても無限回の平均二乗微分可能性は変わりません。

長さ尺度は入力変数の単位に依存するため、入力のスケーリングは重要です。

## 3.8 Outputscale

Outputscale $`\sigma_f^2`$ は関数値の変動の分散尺度です。振幅の標準偏差に対応するのは $\sigma_f$ です。

Kernel を

$$
k(x,x')=\sigma_f^2\tilde{k}(x,x')
$$

と考えると、$\tilde k(x,x)=1$ の場合は $\tilde k$ が相関構造、$\sigma_f^2$ が事前分散を表します。
一般のカーネルでは対角値が入力依存の場合もあり、この分解だけで相関と分散が分離されるとは限りません。

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

## 3.11 出力の標準化

出力についても

$$
y' = \frac{y-\mu_y}{\sigma_y}
$$

のように標準化すると、ハイパーパラメータの尺度や事前分布を設定しやすくなります。
ここで $\mu_y$ と $\sigma_y>0$ は学習データなどから定めた中心と尺度です。
変換後の共分散と観測ノイズ分散はともに $1/\sigma_y^2$ 倍になります。
予測平均を物理単位へ戻す場合は $\sigma_y$ 倍してから $\mu_y$ を加え、
予測分散は $\sigma_y^2$ 倍します。平均・標準偏差・分散の変換を混同しないことが重要です。

## 3.12 Kernel の和

Kernel 同士を加算して新しい kernel を作ることができます。
$`k_1`$ と $`k_2`$ が有効な PSD kernel なら、その和も PSD なので
GPの covariance kernel として利用できます。

$$
k(x,x')=k_1(x,x')+k_2(x,x')
$$

これは独立なGP成分 $f_1,f_2$ を用いて関数を

$$
f(x)=f_1(x)+f_2(x)
$$

という複数成分の和としてモデル化することに対応します。
共分散の双線形性により、独立な成分間の交差共分散がゼロとなるため、
$\operatorname{Cov}(f(x),f(x'))=k_1(x,x')+k_2(x,x')$ を得ます。
成分が相関していれば交差項も必要です。

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

積カーネルは、複数の構造が同時に成立するような共分散を表せます。
例えば $x=(u,v)$ に対し $k_u(u,u')k_v(v,v')$ と置くと、
どちらかの共分散が小さい組では全体の共分散も小さくなり、二つの変数群の相互作用を表せます。
ただし、このカーネルを持つGPと、独立GPの積 $f_1(x)f_2(x)$ は同じ分布ではありません。
独立な零平均GPの積は積カーネルと同じ共分散を持ちますが、一般にガウス過程ではありません。

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

マルチタスク GP や Kronecker 構造のマルチタスク GP の理論を理解する際の基礎になります。

## 3.15 Kronecker structure

第3.14節の分離可能なカーネルを用い、全タスクを同じ入力点で観測するブロック配置では、潜在共分散行列に

$$
K = K_X \otimes K_T
$$

という Kronecker product 構造が現れます。

ここでは入力点を外側、タスクを内側に並べます。タスク数を $T$ とすると、
$K_X$ は $n\times n$、$K_T$ は $T\times T$、全体は $nT\times nT$ です。
観測順序やノイズ構造による違いは[多出力GP章](07_multitask_multioutput.md)で扱います。

単一タスクGPの

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

この考え方は、単一応答の階層モデルにも、階層構造とタスク間相関を組み合わせたモデルにも利用できます。

## 3.18 Kernel とモデル選択

GPモデルの違いのかなりの部分は、**どのような covariance structure を仮定するか**として理解できます。

```text
単一タスク GP
  └─ 入力間 covariance

混合入力 GP
  └─ 連続 + category covariance

マルチタスク GP
  └─ 入力 covariance × task covariance

Kronecker 構造のマルチタスク GP
  └─ K_X ⊗ K_T

Hierarchical Conditional カーネル GP
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

## 3.21 Kernel の仮定とモデル構造

カーネル選択は、入力・出力・タスク間でどのような共分散を仮定するかを
決定します。連続入力の平滑性、カテゴリ間の類似性、タスク間の相関、
階層条件による有効次元などは、それぞれ異なる構造的仮定です。

カーネルの和や積が正半定値性を保持することは、複合モデルを構成する
基本原理です。ただし任意の距離やマスク処理を既存のカーネルに適用しても、
正半定値性が保証されるわけではありません。

## 3.22 この章で覚えておくこと

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

## 3.23 次に読む章

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
