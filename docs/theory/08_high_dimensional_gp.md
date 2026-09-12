# 8. High-dimensional Gaussian Process

## 8.1 この章で扱うこと

入力次元が増えると、通常の Gaussian Process（GP）と Bayesian Optimization（BO）は急速に難しくなります。

```text
低次元
  └─ 少数の観測でも空間全体をある程度カバーできる

高次元
  └─ 同じ観測数では空間がほとんど埋まらない
```

例えば、10点の観測を

- 1次元空間
- 10次元空間
- 100次元空間

で比較すると、同じ10点でも情報密度はまったく異なります。

本章では、robotorchanで利用できる

- `SaasFullyBayesianSingleTaskGP`
- `SaasFullyBayesianMultiTaskGP`
- `AdditiveMapSaasSingleTaskGP`
- `EnsembleMapSaasSingleTaskGP`
- `OrthogonalAdditiveGP`

を中心に、高次元 GP / BO をどのように考えるかを説明します。

重要なのは、**「高次元だから特別なモデルを使う」のではなく、高次元問題にどのような低次元構造が潜んでいると仮定するか**です。

---

## 8.2 高次元で何が難しくなるのか

入力を

\[
x\in\mathbb{R}^d
\]

とします。

次元 `d` が増えると、BOでは主に次の3つが難しくなります。

1. surrogate modelの学習
2. acquisition functionの最適化
3. 有効な探索点を少数の評価で見つけること

特に、評価回数が少ない高次元BOでは

```text
dimension ↑
    ↓
観測空間が疎になる
    ↓
lengthscale推定が難しくなる
    ↓
posterior uncertaintyの識別が難しくなる
    ↓
acquisition optimizationも難しくなる
```

という問題が重なります。

---

## 8.3 Curse of Dimensionality

高次元問題を考える上で中心となるのが **次元の呪い（curse of dimensionality）** です。

各変数を10分割するとします。

```text
1次元: 10 grid points
2次元: 100 grid points
5次元: 100,000 grid points
10次元: 10,000,000,000 grid points
```

一般には各次元を `k` 分割すると

\[
k^d
\]

個の格子点が必要です。

BOはgrid searchよりはるかに効率的ですが、**空間体積が指数的に増えるという本質的問題から完全には逃れられません**。

---

## 8.4 高次元でもBOが使える理由

高次元BOが成立する重要な理由は、多くの実問題で

\[
d_{\text{effective}} \ll d
\]

となることです。

つまり名目上は100変数あっても、実際には

- 5変数だけが強く効く
- 一部の低次元部分空間だけで関数が変化する
- 各変数の効果がほぼ加法的である

といった構造を持つ場合があります。

これを **effective dimension** と呼びます。

```text
100 dimensions
    ↓
実際に重要なのは5 dimensions
    ↓
その5次元を識別できればBOは成立しやすい
```

SAASはこの考え方を、GPのlengthscale priorとして組み込みます。

---

## 8.5 ARD lengthscale

標準的なGPでは、各入力次元に別々のlengthscaleを持つAutomatic Relevance Determination（ARD）を使えます。

RBF kernelなら

\[
k(x,x')
=
\sigma_f^2
\exp\left(
-\frac{1}{2}
\sum_{j=1}^d
\frac{(x_j-x_j')^2}{\ell_j^2}
\right)
\]

です。

ここで

\[
\ell_j
\]

が第 `j` 次元のlengthscaleです。

直感的には

```text
lengthscale 小
  └─ その変数を少し動かすだけで関数が変化する
     → 重要度が高い可能性

lengthscale 大
  └─ その変数を動かしても関数があまり変化しない
     → 重要度が低い可能性
```

と解釈できます。

---

## 8.6 ARDだけでは高次元で不十分なことがある

ARDは有用ですが、高次元・少数データでは各lengthscaleを自由に推定すると不安定になりやすい問題があります。

例えば

```text
observations = 30
input dimensions = 100
```

のような場合、100個のlengthscaleをデータだけから安定に識別するのは難しいです。

結果として

- irrelevant dimensionにも短いlengthscaleが付く
- uncertaintyが不自然になる
- marginal likelihoodに複数の局所解が生じる
- acquisition functionが無関係な方向にも探索する

といった問題が起こり得ます。

そこで、**多くの次元は重要ではないというprior**を明示的に入れるのがSAASです。

---

## 8.7 SAASとは

SAASは

**Sparse Axis-Aligned Subspace**

の略です。

中心となる仮定は

> 高次元入力のうち、本当に関数へ強く影響する軸は少数である

というものです。

例えば100次元でも

```text
x1   important
x2   unimportant
x3   unimportant
x4   important
...
x83  important
...
```

のように、少数の軸だけが有効だと考えます。

SAASはこのsparsityを、inverse lengthscaleへの階層priorで表現します。

---

## 8.8 inverse lengthscaleで考える

lengthscaleを

\[
\ell_j
\]

とすると、inverse lengthscaleを

\[
\rho_j=\frac{1}{\ell_j}
\]

と考えます。

すると

```text
重要な変数
  → lengthscale 小
  → inverse lengthscale 大

重要でない変数
  → lengthscale 大
  → inverse lengthscale ほぼ0
```

となります。

つまりSAASでは、ほとんどの

\[
\rho_j
\]

を0付近へ縮小できれば、重要変数だけを残せます。

---

## 8.9 SAAS prior

SAASでは、典型的にはhierarchical half-Cauchy priorを使います。

概念的には

\[
\tau \sim \operatorname{HalfCauchy}(\beta)
\]

\[
\rho_j \sim \operatorname{HalfCauchy}(\tau)
\]

です。

ここで

- `τ` は全次元に共通するglobal shrinkage
- `ρ_j` は各次元のinverse lengthscale

です。

Half-Cauchyは0付近に強い質量を持ちながらheavy tailも持つため、

```text
大多数のρ_j
  → 0付近へ shrink

本当に重要な少数のρ_j
  → heavy tailを使って大きな値へ escape
```

という挙動を実現できます。

これがSAASの核心です。

---

## 8.10 sparsityは変数選択そのものではない

SAASはLassoのように変数を厳密に0/1で選択するモデルではありません。

posterior上で

- 重要な変数は短いlengthscale
- 重要でない変数は長いlengthscale

になるように強く誘導します。

したがって

```text
SAAS = Bayesian variable selection
```

と完全に同一視するのではなく、

> 高次元GPのlengthscaleにsparsity-inducing priorを置く

と理解するのが正確です。

---

## 8.11 Fully Bayesian inference

`SaasFullyBayesianSingleTaskGP`では、通常のGPのように1組のhyperparameterをMAP推定するのではなく、hyperparameter posteriorをサンプリングします。

通常のGPでは概念的に

\[
\hat\theta
=
\arg\max_\theta p(\theta\mid D)
\]

のような1点を使います。

Fully Bayesianでは

\[
p(\theta\mid D)
\]

そのものを扱い、

\[
p(f_*\mid D)
=
\int
p(f_*\mid D,\theta)
p(\theta\mid D)
\,d\theta
\]

とhyperparameter uncertaintyを積分します。

実際にはこの積分をMCMC sampleで近似します。

---

## 8.12 NUTSによる学習

BoTorchのFully Bayesian SAASは、`fit_fully_bayesian_model_nuts`を使って学習します。

```python
from botorch.fit import fit_fully_bayesian_model_nuts

fit_fully_bayesian_model_nuts(model)
```

NUTSはNo-U-Turn Samplerで、Hamiltonian Monte Carlo（HMC）の一種です。

robotorchanでは

```python
model.supports_mll
```

が`False`です。

つまり

```python
model.make_mll()
```

による通常のExact GP fittingとは異なります。

---

## 8.13 SAAS posteriorはmixtureになる

MCMCで複数のhyperparameter sample

\[
\theta_1,\theta_2,\ldots,\theta_S
\]

を得ると、それぞれに対応するGP posteriorがあります。

そのためposteriorは概念的に

\[
p(f_*\mid D)
\approx
\frac{1}{S}
\sum_{s=1}^S
p(f_*\mid D,\theta_s)
\]

というmixtureになります。

BoTorchではMCMC sampleに対応するbatch dimensionを持つため、通常GPとposterior shapeが異なる点に注意します。

---

## 8.14 `SaasFullyBayesianSingleTaskGP`

robotorchanではBoTorchのFully Bayesian SAASを薄くwrapしています。

```python
import torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.transforms import Normalize, Standardize
from robotorchan.models import SaasFullyBayesianSingleTaskGP

torch.set_default_dtype(torch.double)

train_X = torch.rand(30, 50)
train_Y = torch.sin(6.0 * train_X[:, :1]) + 0.5 * train_X[:, 3:4]

model = SaasFullyBayesianSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=50),
    outcome_transform=Standardize(m=1),
)

fit_fully_bayesian_model_nuts(
    model,
    warmup_steps=256,
    num_samples=128,
    thinning=16,
    disable_progbar=True,
)
```

50次元入力でも真に効く変数が少数なら、SAAS priorが有効に働く可能性があります。

---

## 8.15 `median_lengthscale`

SAAS modelでは、MCMC sampleごとにlengthscaleがあります。

BoTorchでは代表値としてmedian lengthscaleを確認できます。

```python
model.median_lengthscale
```

例えば

```text
x0  0.4
x1  80.0
x2  120.0
x3  1.2
x4  95.0
```

なら、`x0`と`x3`が他より重要である可能性があります。

ただし、lengthscaleは説明変数の因果的重要度ではありません。

入力scaleや相関、観測領域にも依存するため、feature importanceとして過度に解釈しないことが重要です。

---

## 8.16 入力正規化が重要

lengthscale priorを使う高次元GPでは、入力scaleの統一が特に重要です。

例えば

```text
x1: 0 ～ 1
x2: 0 ～ 1,000,000
```

をそのまま使うとlengthscaleの比較が意味を失います。

基本的には

\[
x_j\in[0,1]
\]

へ正規化します。

```python
from botorch.models.transforms import Normalize

input_transform = Normalize(d=train_X.shape[-1])
```

SAASを使う場合は特に、入力scaleをモデル設計の一部として扱います。

---

## 8.17 Fully Bayesian SAASの利点

Fully Bayesian SAASの主な利点は

1. 高次元でirrelevant variableを強くshrinkできる
2. hyperparameter uncertaintyをposteriorへ反映できる
3. 少数データでもMAP estimateの一点推定に依存しにくい

ことです。

特に

```text
dimension 高い
observation 少ない
effective dimension 小さい
```

という条件で有力です。

---

## 8.18 Fully Bayesian SAASの弱点

一方で計算は重くなります。

NUTSでは多数のMCMC stepが必要です。

さらに各stepでGP likelihood計算を行うため、データ数 `n` が増えると負荷が大きくなります。

```text
高次元 + 少数データ
  → SAAS向き

高次元 + 大量データ
  → NUTSコストが問題になりやすい
```

SAASBOは「評価が高価でデータ数は少ない」というBO本来の条件に特に適しています。

大規模データ問題では、次章のVariational GPなど別の方法も検討します。

---

## 8.19 Fully BayesianとMAPの違い

高次元GPでは、Fully Bayesian SAAS以外にMAP-SAASもあります。

```text
Fully Bayesian SAAS
  └─ hyperparameter posteriorをMCMCでサンプリング

MAP-SAAS
  └─ SAAS priorを使いながら代表的なhyperparameterを最適化
```

一般に

```text
Fully Bayesian
  + uncertaintyをより完全に扱える
  - 計算が重い

MAP
  + fittingが軽い
  - posterior hyperparameter uncertaintyを簡略化
```

という違いがあります。

---

## 8.20 MAP-SAAS

MAP-SAASでは、SAASのsparsity-inducing priorを維持しつつ、NUTSを使わずMAP fittingします。

robotorchanでは

- `AdditiveMapSaasSingleTaskGP`
- `EnsembleMapSaasSingleTaskGP`

を利用できます。

これらは通常のExact GP wrapperと同じく

```python
mll = model.make_mll()
fit_gpytorch_mll(mll)
```

で学習できます。

---

## 8.21 `AdditiveMapSaasSingleTaskGP`

このモデルは複数のSAAS prior scaleを用いたadditiveな構造を使います。

robotorchan APIでは

```python
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import AdditiveMapSaasSingleTaskGP

model = AdditiveMapSaasSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    num_taus=4,
)

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

のように利用します。

`num_taus`はSAAS shrinkage scaleの表現に関係します。

Fully Bayesian SAASより軽量な代替として検討できます。

---

## 8.22 `EnsembleMapSaasSingleTaskGP`

`EnsembleMapSaasSingleTaskGP`は、複数のSAAS MAP modelをensembleとして扱うモデルです。

```python
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import EnsembleMapSaasSingleTaskGP

model = EnsembleMapSaasSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    num_taus=4,
)

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

`taus`を明示的に渡すこともできます。

```python
model = EnsembleMapSaasSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    taus=taus,
)
```

考え方としては、単一のMAP solutionだけに依存するより、複数のshrinkage設定を持つmodel ensembleで不確実性をある程度表現します。

---

## 8.23 Fully Bayesian SAASとMAP-SAASの選択

実務上は次のように考えられます。

| 状況 | 第一候補 |
|---|---|
| 評価回数が少なくmodel fitting時間を許容 | Fully Bayesian SAAS |
| 高次元だがNUTSが重い | MAP-SAAS |
| hyperparameter uncertaintyを重視 | Fully Bayesian SAAS |
| fitting速度を重視 | MAP-SAAS |
| 通常GPでも十分安定 | `SingleTaskGP` |

高次元だから必ずSAASを使う必要はありません。

30次元でも全変数が本当に重要なら、SAASのsparsity仮定が合わない可能性があります。

---

## 8.24 `SaasFullyBayesianMultiTaskGP`

高次元とMulti-taskを組み合わせたい場合、robotorchanでは

```python
SaasFullyBayesianMultiTaskGP
```

を利用できます。

これは

```text
Multi-task structure
    +
SAAS prior on data dimensions
```

を組み合わせる考え方です。

例えば

```text
入力: 80 process variables
Task 0: 装置A
Task 1: 装置B
Task 2: 装置C
```

で、実際に重要なprocess variableが少数なら有力です。

---

## 8.25 Multi-task SAASの入力形式

通常の`MultiTaskGP`と同じように、task featureを含むlong-formatを使います。

```python
model = SaasFullyBayesianMultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=-1,
)

fit_fully_bayesian_model_nuts(model)
```

robotorchan wrapperでは

- `task_feature`
- `output_tasks`
- `rank`
- `all_tasks`

などのBoTorch APIを維持しています。

---

## 8.26 SAASが仮定するaxis-aligned sparsity

SAASの重要な制約は **axis-aligned** であることです。

つまり

```text
重要な方向 = x1, x7, x20
```

のように、元の座標軸に沿った少数変数が重要だと仮定します。

一方、真の関数が

\[
z=x_1+x_2+x_3+x_4
\]

のような回転した低次元部分空間だけで変化する場合、SAASの軸方向sparsityは必ずしも最適ではありません。

この場合は

- random embedding
- learned embedding
- BAxUS
- latent-space BO

など別の高次元BO手法も候補になります。

robotorchanの現在の高次元GP群は、主にaxis-aligned sparsityとadditive structureを扱います。

---

## 8.27 Additive structure

高次元を扱うもう1つの強い仮定が **加法構造** です。

例えば

\[
f(x)
=
c
+f_1(x_1)
+f_2(x_2)
+\cdots
+f_d(x_d)
\]

と仮定します。

この場合、本来の `d` 次元関数を直接学習する代わりに、`d` 個の1次元関数を学習できます。

```text
通常GP
  └─ 1つの高次元関数 f(x1,...,xd)

Additive GP
  └─ f1(x1) + f2(x2) + ... + fd(xd)
```

これによりsample efficiencyが大きく改善する可能性があります。

---

## 8.28 `OrthogonalAdditiveGP`

robotorchanではBoTorchの`OrthogonalAdditiveGP`をwrapしています。

```python
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import OrthogonalAdditiveGP

model = OrthogonalAdditiveGP(
    train_X=train_X,
    train_Y=train_Y,
)

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

このモデルはOrthogonal Additive Kernelを用いて、各入力次元の寄与を分解します。

高次元問題でも関数がほぼadditiveなら、通常GPより少数データで学習しやすくなります。

---

## 8.29 second-order interaction

完全なadditive modelでは

\[
f(x)=c+\sum_i f_i(x_i)
\]

ですが、現実には変数間interactionがあります。

`OrthogonalAdditiveGP`では

```python
second_order = True
```

を使うことで、概念的に

\[
f(x)
=
c
+\sum_i f_i(x_i)
+\sum_{i<j}f_{ij}(x_i,x_j)
\]

という2次interactionを扱えます。

ただしinteraction数は

\[
\frac{d(d-1)}{2}
\]

で増えるため、高次元では計算量とモデル複雑度に注意が必要です。

まずfirst-order additiveで始め、必要な場合にsecond-orderを検討するのが実務的です。

---

## 8.30 SAASとAdditive GPは仮定が違う

両者はどちらも高次元問題を扱えますが、仮定が異なります。

```text
SAAS
  └─ 多数の変数のうち少数だけが重要

Additive GP
  └─ 多数の変数が効いてもよいが、効果が主に加法的
```

例えば

```text
100変数中5変数だけ重要
  → SAASが自然

30変数すべてが少しずつ効くがinteractionが弱い
  → Additive GPが自然
```

です。

---

## 8.31 effective sparsityとadditivity

実際の問題は両者の中間にあります。

例えば

\[
f(x)
=f_1(x_1)+f_4(x_4)+f_{12}(x_1,x_2)
\]

なら

- effective dimensionは低い
- additive / low-order interaction structureもある

という問題です。

高次元BOでは、単にモデル名を選ぶのではなく

> どの低次元構造を信じるか

を明示することが重要です。

---

## 8.32 acquisition function側も高次元で難しくなる

高次元問題ではsurrogate modelだけでなく、acquisition functionの最適化も難しくなります。

例えば

\[
x^*=\arg\max_x \alpha(x)
\]

を100次元空間で解く必要があります。

すると

- random startsが局所最適解へ入りやすい
- raw sample数が不足しやすい
- gradient optimizationが不安定になりやすい

といった問題があります。

したがってSAASを使えば高次元BO問題がすべて解決するわけではありません。

modelingとcandidate optimizationの両方を考えます。

---

## 8.33 Trust Regionという別アプローチ

高次元BOではsearch space全体ではなく、現在の有望領域だけを局所探索するTrust Region方式も有力です。

TuRBOでは概念的に

```text
global domain
    ↓
current best周辺のtrust region
    ↓
local GP / local acquisition optimization
```

とします。

BAxUSでは低次元subspaceを使いながら必要に応じて次元を拡張します。

これらはSAASとは異なるアプローチです。

```text
SAAS
  → model priorでeffective dimensionを学ぶ

TuRBO
  → search regionを局所化

BAxUS
  → subspace dimensionalityを適応的に拡張
```

これらを組み合わせる研究も可能です。

---

## 8.34 PCAで次元削減すればよいのか

高次元ならPCAを使えばよいように見えますが、BOでは注意が必要です。

PCAは

\[
\operatorname{Var}(X)
\]

を大きく説明する方向を残します。

しかしBOで重要なのは

\[
Y
\]

に効く方向です。

したがって

```text
Xの分散が大きい方向
```

と

```text
objectiveに重要な方向
```

は一致しない可能性があります。

PCAは有力なpreprocessingですが、高次元BOの万能解ではありません。

---

## 8.35 Deep Kernel Learningとの違い

Deep Kernel Learning（DKL）では

\[
z=g_\phi(x)
\]

というneural network embeddingを作り、latent space上でGPを使います。

```text
high-dimensional x
    ↓ neural network
low-dimensional z
    ↓ GP
posterior
```

これはSAASとは異なります。

```text
SAAS
  └─ 元入力空間の軸方向sparsityを学習

DKL
  └─ nonlinear feature representationを学習
```

画像・組成・構造descriptorなど非常に複雑な入力ではDKLが有力になることがあります。

---

## 8.36 高次元でのデータ数

「何次元から高次元か」という固定の境界はありません。

重要なのは

\[
\frac{n}{d}
\]

だけでもなく、

- effective dimension
- noise
- smoothness
- interaction structure
- evaluation budget

です。

例えば

```text
50 dimensions, n=30
```

でも5変数だけが重要ならSAASが有効かもしれません。

一方

```text
20 dimensions, n=200
```

でも20変数すべてに複雑なinteractionがあれば難しい問題です。

---

## 8.37 SAASを使う典型条件

SAASを第一候補にしやすいのは

- 入力次元が比較的高い
- 評価回数が少ない
- 多くの変数は無関係または弱いと考えられる
- 重要変数が元座標軸に沿って存在すると考えられる
- model fittingにMCMC時間を使える

という場合です。

```text
high d
small n
sparse relevance
expensive objective
```

が典型です。

---

## 8.38 SAASを避けた方がよい場合

一方、次のような場合は慎重に使います。

- ほぼ全変数が重要
- 重要な方向が回転したsubspaceにある
- データ数が非常に多い
- NUTSの計算時間を許容できない
- 強いcategorical / hierarchical structureが中心

この場合は

- standard GP
- Additive GP
- Variational GP
- DKL
- subspace BO
- trust-region BO

などを検討します。

---

## 8.39 MAP-SAASを使う典型条件

MAP-SAASは

```text
SAAS priorは使いたい
    +
NUTSは重すぎる
```

という場合に有力です。

特に反復回数が多いBOでは、毎iterationのMCMC fittingが全体runtimeの大部分を占めることがあります。

その場合はMAP-SAASをbaselineとして比較する価値があります。

---

## 8.40 Additive GPを使う典型条件

`OrthogonalAdditiveGP`を検討しやすいのは

- 多数の変数が効く
- 主効果が強い
- interactionが比較的弱い
- 各変数の寄与を解釈したい

場合です。

component-wise functionを理解しやすい点も重要です。

```text
prediction accuracy
      +
interpretability
```

を両立しやすい高次元GPです。

---

## 8.41 raw-data API

Fully Bayesian SAAS wrapperでも、robotorchanのraw-data規約を維持します。

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
```

ただし

```python
model.supports_mll is False
```

です。

MAP-SAAS / OrthogonalAdditiveGPはExact GP系なので

```python
model.make_mll()
```

を利用できます。

---

## 8.42 学習方法の整理

| Model | 学習方法 |
|---|---|
| `SingleTaskGP` | Exact MLL |
| `SaasFullyBayesianSingleTaskGP` | NUTS |
| `SaasFullyBayesianMultiTaskGP` | NUTS |
| `AdditiveMapSaasSingleTaskGP` | Exact MLL / MAP |
| `EnsembleMapSaasSingleTaskGP` | Exact MLL / MAP ensemble |
| `OrthogonalAdditiveGP` | Exact MLL |

高次元モデルを選ぶときは、**surrogate structureだけでなくfitting methodも重要**です。

---

## 8.43 モデル選択フロー

高次元連続入力なら、次のように考えられます。

```text
入力次元が高い
    ↓
少数の変数だけが重要そう？
 ├─ Yes
 │    ↓
 │   NUTSコストを許容？
 │    ├─ Yes → SaasFullyBayesianSingleTaskGP
 │    └─ No  → MAP-SAAS
 │
 └─ No
      ↓
      ほぼ加法的？
       ├─ Yes → OrthogonalAdditiveGP
       └─ No
            ↓
            standard GP / DKL / TuRBO / BAxUS 等を検討
```

Multi-taskなら

```text
high-dimensional + related tasks
    → SaasFullyBayesianMultiTaskGP
```

も候補です。

---

## 8.44 実務上の比較表

| 状況 | 第一候補 | 理由 |
|---|---|---|
| 高次元・少数重要変数 | Fully Bayesian SAAS | sparsity prior |
| 高次元・SAASを軽量化したい | MAP-SAAS | NUTS不要 |
| 高次元・関連taskあり | Multi-task SAAS | task sharing + sparsity |
| 多数変数が加法的に効く | OrthogonalAdditiveGP | additive structure |
| 少数次元・十分なデータ | SingleTaskGP | simpler |
| 大量データ | Variational GPも検討 | Exact GP / NUTSの計算負荷 |
| 非線形なlatent低次元構造 | DKL等 | learned representation |
| 局所探索が有効 | TuRBO等 | trust region |

---

## 8.45 よくある誤解

### 「20次元を超えたらSAAS」

固定の閾値はありません。重要なのはeffective dimensionとデータ数です。

### 「SAASは自動特徴選択をする」

厳密な離散的特徴選択ではありません。inverse lengthscaleをsparseにするBayesian priorです。

### 「lengthscaleが短い変数は因果的に重要」

違います。GP上の感度を表しますが、因果関係を意味しません。

### 「Fully Bayesian SAASは常にMAP-SAASより良い」

必ずしもそうではありません。計算予算とデータ条件によります。

### 「高次元問題ではadditive GPが安全」

強い高次interactionがある場合、additive仮定はbiasになります。

### 「SAASを使えばacquisition optimizationも簡単になる」

modelingは改善しても、高次元acquisition optimization自体の難しさは残ります。

---

## 8.46 high-dimensional BOで確認したい診断

実務では次を確認します。

1. posterior predictive accuracy
2. median lengthscaleの分布
3. important dimensionの安定性
4. independent restart間の再現性
5. acquisition candidateが境界へ偏っていないか
6. standard GPとの比較
7. MAP-SAASとFully Bayesian SAASの比較
8. objective improvement per wall-clock time

単にbest valueだけでなく、modelの挙動も確認します。

---

## 8.47 dimension importanceの安定性

SAASではMCMCやiterationごとに重要dimensionが変わることがあります。

特に初期データが少ない場合、これは自然です。

```text
iteration 1
  x3, x8が重要

iteration 5
  x3, x12が重要

iteration 20
  x3, x8, x12へ収束
```

この変化自体がmodel uncertaintyを反映しています。

早い段階で特定dimensionを完全に固定してしまうと探索を誤る可能性があります。

---

## 8.48 BOとの関係

高次元GPはsurrogate modelです。

最終的なBO loopは通常と同じです。

```text
observations
    ↓
high-dimensional GP
    ↓
posterior
    ↓
acquisition function
    ↓
optimize acquisition
    ↓
next candidate
```

SAAS専用のacquisition functionが必須なわけではありません。

BoTorchのMonte Carlo acquisition functionと組み合わせられます。

例えば

- qLogEI
- qLogNEI
- qUCB

などが候補です。

---

## 8.49 `qLogEI`との組み合わせ例

```python
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf

acqf = qLogExpectedImprovement(
    model=model,
    best_f=train_Y.max(),
)

bounds = torch.stack(
    [
        torch.zeros(train_X.shape[-1], dtype=train_X.dtype),
        torch.ones(train_X.shape[-1], dtype=train_X.dtype),
    ]
)

candidate, acq_value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=512,
)
```

高次元では`raw_samples`や`num_restarts`も計算コストと探索品質のトレードオフになります。

---

## 8.50 高次元モデルの責務

モデルと最適化の責務を分けると整理しやすくなります。

```text
SAAS / Additive GP
  └─ 高次元関数をどうmodelingするか

Acquisition Function
  └─ posteriorを使って次点価値をどう定義するか

Acquisition Optimizer
  └─ 高次元空間でacquisitionをどう最大化するか
```

この3つを混同しないことが重要です。

---

## 8.51 まとめ

高次元BOで最も重要なのは、**名目次元ではなく、どのような低次元構造が問題に存在するか**です。

```text
高次元
  ↓
どんな構造を仮定する？

少数重要変数
  → SAAS

少数重要変数 + 軽量化
  → MAP-SAAS

関連task + 少数重要変数
  → Multi-task SAAS

加法構造
  → OrthogonalAdditiveGP
```

理論上の中心は、SAASでは

\[
\rho_j=1/\ell_j
\]

にsparsity-inducing hierarchical priorを置き、多くのinverse lengthscaleを0付近へshrinkすることです。

一方、Additive GPでは

\[
f(x)=c+\sum_i f_i(x_i)
\]

のような構造を仮定して高次元性を緩和します。

特に覚えておきたい点は次の通りです。

1. 高次元BOではeffective dimensionが重要
2. ARDだけでは少数データでlengthscale推定が不安定になり得る
3. SAASはaxis-aligned sparsityをpriorとして導入する
4. Fully Bayesian SAASはNUTSを使うため`make_mll()`では学習しない
5. MAP-SAASは計算コストを下げる選択肢
6. Additive GPは「少数重要変数」とは異なる構造仮定
7. 高次元ではacquisition optimization自体も難しい
8. model selectionは次元数だけで決めない

次章では、データ数が増えてExact GPの計算が重くなる場合に使う [Variational GP](09_variational_gp.md) を説明します。

## 参考

- Eriksson, Jankowiak, *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces*, UAI 2021
- BoTorch `SaasFullyBayesianSingleTaskGP`
- BoTorch `SaasFullyBayesianMultiTaskGP`
- BoTorch MAP-SAAS models
- BoTorch `OrthogonalAdditiveGP`
- [Gaussian Process](02_gaussian_process.md)
- [Kernel](03_kernel.md)
- [Acquisition Function](04_acquisition_function.md)
