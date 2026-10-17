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

本章では、SAAS、MAP-SAAS、Multi-task SAAS、Additive GPを中心に、高次元GP / BOの構造仮定を説明します。

重要なのは、**「高次元だから特別なモデルを使う」のではなく、高次元問題にどのような低次元構造が潜んでいると仮定するか**です。

---

### 「何次元から高次元か」に固定の境界はない

10次元なら低次元、100次元なら高次元、という普遍的な閾値はありません。
難しさは少なくとも

~~~text
入力次元 d
観測数 n
探索領域の広さ
関数の滑らかさ
実際に効いている方向の数
acquisitionを最適化する難しさ
~~~

の組合せで決まります。

例えば100次元でも実際に効く軸が3つなら構造を利用できる可能性があります。
一方、20次元でも20変数すべてが複雑に相互作用し、観測が10点しかなければ
非常に難しい問題です。

したがって本章では「名目次元が何個か」より、**データ量に対してどのような
低次元構造を仮定できるか**を中心に考えます。

## 8.2 高次元で何が難しくなるのか

入力を

$$
x\in\mathbb{R}^d
$$

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

$$
k^d
$$

個の格子点が必要です。

BOはgrid searchよりはるかに効率的ですが、**空間体積が指数的に増えるという本質的問題から完全には逃れられません**。

---

## 8.4 高次元でもBOが使える理由

高次元BOが成立する重要な理由は、多くの実問題で

$$
d_{\text{effective}} \ll d
$$

となることです。

つまり名目上は100変数あっても、実際には

- 5変数だけが強く効く
- 一部の低次元部分空間だけで関数が変化する
- 各変数の効果がほぼ加法的である

といった構造を持つ場合があります。

このような考え方を広い意味で **effective dimension** と呼びます。
ただし文献によってeffective dimensionの厳密な定義は異なるため、
単一の指標として固定せず「関数変動を説明する実質的な自由度」と捉えます。

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

$$
k(x,x')
=
\sigma_f^2
\exp\left(
-\frac{1}{2}
\sum_{j=1}^d
\frac{(x_j-x_j')^2}{\ell_j^2}
\right)
$$

です。

ここで

$$
\ell_j
$$

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

ただしARD lengthscaleは単独のfeature importance scoreではありません。入力scale、
入力間相関、kernel、観測範囲などにも依存するため、短いlengthscaleをそのまま
「重要度が高い」と順位付けするのは避けます。

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

### 高次元BOには少なくとも2つの難しさがある

高次元対策は大きく分けると、

~~~text
surrogate側
  └─ 少ない観測から高次元関数をどう学習するか

candidate search側
  └─ 高次元空間でacquisitionの良い点をどう探すか
~~~

があります。

SAASやAdditive GPは主に前者です。TuRBO、BAxUS、embedding系の探索戦略などは
主に後者、または両者を組み合わせた設計です。

したがって「SAASを使ったので高次元BO対策は完了」とは限りません。

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

$$
\ell_j
$$

とすると、inverse lengthscaleを

$$
\rho_j=\frac{1}{\ell_j}
$$

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

$$
\rho_j
$$

を0付近へ縮小できれば、重要変数だけを残せます。

---

## 8.9 SAAS prior

SAASでは、典型的にはhierarchical half-Cauchy priorを使います。

概念的には

$$
\tau \sim \operatorname{HalfCauchy}(\beta)
$$

$$
\rho_j \sim \operatorname{HalfCauchy}(\tau)
$$

です。

ここで

- `τ` は全次元に共通するglobal shrinkage
- `ρ_j` は各次元のinverse lengthscale

です。

階層priorによりglobal shrinkageと各次元の局所的なescapeを組み合わせます。
Half-Cauchyのheavy tailによって、

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

通常のGPでハイパーパラメータを周辺尤度最大化する場合、一般に得られるのは点推定です。MAP推定では事前分布も考慮した事後モードを用います。Fully Bayesian推論ではハイパーパラメータ $\theta$ の事後分布を積分し、

$$
p(f_*\mid D)=\int p(f_*\mid D,\theta)p(\theta\mid D)\,d\theta
$$

として予測します。実際にはMCMC標本で積分を近似します。

---

## 8.12 NUTSによる学習

NUTS（No-U-Turn Sampler）はHamiltonian Monte Carloの一種で、軌道長を適応的に選ぶMCMC手法です。SAASの階層的なハイパーパラメータ事後分布を近似する際に利用できます。混合の良否、収束、事後サンプル数が予測の信頼性に影響します。

---

## 8.13 SAAS posteriorはmixtureになる

MCMCで複数のhyperparameter sample

$$
\theta_1,\theta_2,\ldots,\theta_S
$$

を得ると、それぞれに対応するGP posteriorがあります。

そのためposteriorは概念的に

$$
p(f_*\mid D)
\approx
\frac{1}{S}
\sum_{s=1}^S
p(f_*\mid D,\theta_s)
$$

というmixtureになります。

BoTorchではMCMC sampleに対応するbatch dimensionを持つため、通常GPとposterior shapeが異なる点に注意します。

---

## 8.14 SAASの高次元・少数データにおける位置付け

SAASは、入力次元が多くても実効的に効く変数が少ないという事前仮定を導入します。軸方向の疎性が適切なら少数観測でも有効ですが、重要変数が多数ある場合や回転した低次元部分空間に依存する場合は近似の偏りに注意が必要です。

---

## 8.15 Lengthscaleの事後要約

Fully Bayesian SAASでは、各次元のlengthscaleは単一の推定値ではなく事後分布を持ちます。中央値などで重要そうな次元を概観できますが、入力尺度、変数間相関、観測領域に依存するため、因果的重要度と解釈してはいけません。

---

## 8.16 入力正規化が重要

Lengthscaleの事前分布を比較可能にするため、入力変数の尺度を揃えることが重要です。例えば、各入力を学習データに基づいて $[0,1]$ へ正規化すると、逆lengthscaleの縮小を次元間で比較しやすくなります。学習時と予測時で同じ変換を使用し、検証データから正規化範囲を推定しないようにします。

---

## 8.17 Fully Bayesian SAASの利点

Fully Bayesian SAASの主な利点は

1. 高次元でirrelevant variableを強くshrinkできる
2. hyperparameter uncertaintyをposteriorへ反映できる
3. hyperparameterの一点推定だけに依存せずposterior uncertaintyを伝播できる

ことです。

特に

```text
dimension 高い
observation 少ない
axis-alignedなeffective sparsityを期待できる
```

という条件と整合しやすいモデルです。

ただしこれは性能保証ではありません。SAAS priorの仮定と真の関数構造が
合っているかはデータと比較実験で確認します。

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

MAP-SAASは、疎性を誘導する事前分布を用いながら、MCMCによる事後積分の代わりにハイパーパラメータを点推定する方法です。計算負荷を抑えやすい一方、ハイパーパラメータの事後不確実性は直接積分しません。

---

## 8.21 MAP-SAASにおける縮小の推定

MAP-SAASでは、SAAS型の縮小事前分布を用いながら点推定でハイパーパラメータを求めます。複数の縮小強度を組み合わせる設計もあり、完全ベイズ推論の事後積分とは異なります。

---

## 8.22 複数のMAP-SAAS推定の統合

異なる縮小強度のMAP推定を統合すると、単一設定への依存を緩和できる場合があります。ただし複数の点推定の統合は、NUTSなどによるハイパーパラメータ事後分布の積分と同一ではありません。

---

## 8.23 Fully Bayesian SAASとMAP-SAASの違い

比較するときは「どちらが上位か」ではなく、

~~~text
Fully Bayesian SAAS
  → MCMCでhyperparameter uncertaintyを積分

MAP-SAAS系
  → optimizationを中心にSAAS型priorを利用
~~~

という**推論contractの違い**を先に理解します。計算時間だけでなく、
posterior shapeや下流acquisitionとの組合せも確認します。

高次元だから必ずSAASを使う必要はありません。

30次元でも全変数が本当に重要なら、SAASのsparsity仮定が合わない可能性があります。

---

## 8.24 Multi-task構造とSAASの統合

Multi-task SAASは、タスク間の情報共有と、説明変数に対する疎な事前仮定を組み合わせます。共有すべき変動とタスク固有の変動を区別し、タスク間の相関を適切に表現する必要があります。

---

## 8.25 Multi-task SAASの共分散構造

タスクを離散的な入力として扱う場合、データ次元とタスク次元の役割は異なります。SAASによる縮小は説明変数側に適用し、タスク間相関は別の共分散構造で表現します。詳細は[Multi-task GP](07_multitask_multioutput.md)を参照してください。

---

## 8.26 SAASが仮定するaxis-aligned sparsity

SAASの重要な制約は **axis-aligned** であることです。

つまり

```text
重要な方向 = x1, x7, x20
```

のように、元の座標軸に沿った少数変数が重要だと仮定します。

一方、真の関数が

$$
z=x_1+x_2+x_3+x_4
$$

のような回転した低次元部分空間だけで変化する場合、SAASの軸方向sparsityは必ずしも最適ではありません。

この場合は

- random embedding
- learned embedding
- BAxUS
- latent-space BO

など別の高次元BO手法も候補になります。

この章で中心に扱うsurrogate群は、主にaxis-aligned sparsityとadditive structureです。
次元削減、ALEBO系、高次元探索など別の構造仮定もあるため、
それらは後続の高次元理論章で分けて扱います。

---

### Axis-aligned sparsityとlatent subspaceは別の仮定

例えば

~~~text
f(x) ≈ g(x_2, x_8)
~~~

なら元の軸そのものが重要なのでSAASの仮定と整合します。一方、

~~~text
f(x) ≈ g(Ax)
~~~

のように複数変数の線形結合や非線形表現に重要方向がある場合は、
元の軸をsparseにする仮定とは異なります。

この違いが、SAASとdimension reduction / embedding系手法を使い分ける
理論上の重要な境界です。

## 8.27 Additive structure

高次元を扱うもう1つの強い仮定が **加法構造** です。

例えば

$$
f(x)
=
c
+f_1(x_1)
+f_2(x_2)
+\cdots
+f_d(x_d)
$$

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

## 8.28 Orthogonal Additive Kernelの考え方

Additive Kernelは、入力次元ごとの寄与を足し合わせることで高次元の関数を構成します。Orthogonalな分解では寄与項間の識別性を高め、主効果と相互作用を整理しやすくします。モデル化の前提は関数の加法構造であり、SAASの軸方向疎性とは異なります。

---

## 8.29 second-order interaction

完全な加法モデルでは $f(x)=c+\sum_i f_i(x_i)$ ですが、変数間相互作用がある場合は

$$
f(x)=c+\sum_i f_i(x_i)+\sum_{i<j}f_{ij}(x_i,x_j)
$$

のように二次相互作用を導入できます。相互作用候補は $d(d-1)/2$ 個に増えるため、データ数とモデル複雑度のトレードオフに注意が必要です。

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

$$
f(x)
=f_1(x_1)+f_4(x_4)+f_{12}(x_1,x_2)
$$

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

$$
x^*=\arg\max_x \alpha(x)
$$

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

$$
\operatorname{Var}(X)
$$

を大きく説明する方向を残します。

しかしBOで重要なのは

$$
Y
$$

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

$$
z=g_\phi(x)
$$

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

$$
\frac{n}{d}
$$

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

加法的なGPは、多数の変数が寄与する一方で主効果が支配的な場合に適しています。各成分の解釈可能性はモデルの識別性や制約に依存し、予測精度だけでなく相互作用の残差も確認する必要があります。

---

## 8.41 推論方法とモデル構造の分離

高次元モデルの事前仮定と、そのハイパーパラメータをどのように推論するかは別の設計選択です。Exact GP、MAP、MCMCを比較する際は、予測精度だけでなく事後不確実性と計算コストを評価します。

---

## 8.42 高次元GPの推論方法の比較

| 推論方式 | 理論上の特徴 |
|---|---|
| 周辺尤度最大化 | ハイパーパラメータの点推定 |
| MAP | 事前分布を含む点推定 |
| Fully Bayesian MCMC | ハイパーパラメータの事後不確実性を積分 |
| 複数MAP推定の統合 | 複数設定の予測を統合するが、厳密な事後積分ではない |

具体的なモデルAPIや学習手順は[モデルドキュメント](../models/README.md)を参照してください。

---

## 8.43 構造仮定を選ぶフロー

高次元モデルは、まず関数に置く構造仮定で選びます。少数の座標軸が重要ならSAAS、加法性が強ければAdditive GP、回転した低次元部分空間に依存するなら埋め込み・次元削減を検討します。次に、推論コストと不確実性の扱いに応じてMAP推定とFully Bayesian推論を比較します。

---

## 8.44 構造仮定の比較表

| 想定する構造 | 対応する考え方 |
|---|---|
| 少数の元変数だけが効く | SAAS / MAP-SAAS系 |
| 関連task + 少数の元変数が効く | Multi-task SAAS |
| 多数変数が主に加法的に効く | Orthogonal Additive GP |
| latentな低次元方向で変化する | reduction / embedding系 |
| 局所領域に探索を集中したい | trust-region系 |
| データ数そのものが大きい | Variational GPも検討 |

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

## 8.49 高次元での獲得関数最適化

高次元の獲得関数は局所解や平坦な領域が増えやすく、初期点数と再始動回数が探索品質と計算時間に影響します。獲得関数の定義と最大化アルゴリズムは区別し、具体的な実行手順は[最適化ドキュメント](../optimization/README.md)で扱います。

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

## 8.51 この章で覚えておくこと

高次元BOで最も重要なのは、**名目次元だけで手法を決めず、どのような低次元構造を
問題に仮定できるか**です。

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

$$
\rho_j=1/\ell_j
$$

にsparsity-inducing hierarchical priorを置き、多くのinverse lengthscaleを0付近へshrinkすることです。

一方、Additive GPでは

$$
f(x)=c+\sum_i f_i(x_i)
$$

のような構造を仮定して高次元性を緩和します。

特に覚えておきたい点は次の通りです。

1. 高次元BOではeffective dimensionが重要
2. ARDだけでは少数データでlengthscale推定が不安定になり得る
3. SAASはaxis-aligned sparsityをpriorとして導入する
4. Fully Bayesian SAASではMCMCによりハイパーパラメータの事後分布を近似する
5. MAP-SAASは計算コストを下げる選択肢
6. Additive GPは「少数重要変数」とは異なる構造仮定
7. 高次元ではacquisition optimization自体も難しい
8. model selectionは次元数だけで決めない

次章では、データ数が増えてExact GPの計算が重くなる場合に使う [Variational GP](09_variational_gp.md) を説明します。

## 参考文献

1. Eriksson, D., and Jankowiak, M. (2021).
   High-dimensional Bayesian optimization with sparse axis-aligned subspaces.
   *Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence*.
2. BoTorch SAAS, MAP-SAAS, and `OrthogonalAdditiveGP` documentation.
3. [Gaussian Process](02_gaussian_process.md)
4. [Kernel](03_kernel.md)
5. [Acquisition Function](04_acquisition_function.md)
