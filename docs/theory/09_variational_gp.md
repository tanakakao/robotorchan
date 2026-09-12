# 9. Variational Gaussian Process

## 9.1 この章で扱うこと

前章では、高次元入力に対して SAAS や Additive GP を使う考え方を説明しました。

本章では別の問題、**データ数が増えて Exact GP の計算が重くなる問題**を扱います。

ここで重要なのは、

```text
高次元問題
  └─ 入力次元 d が大きい

大規模データ問題
  └─ 観測数 n が大きい
```

は別の問題だということです。

例えば

```text
n = 50, d = 100
```

は高次元・少数データ問題ですが、

```text
n = 100,000, d = 5
```

は低次元・大規模データ問題です。

後者では、通常の Exact GP が必要とする共分散行列の計算がボトルネックになります。

robotorchan では BoTorch の

```python
SingleTaskVariationalGP
```

を thin wrapper として提供しています。

本章では、

- Exact GP の計算量
- inducing point
- sparse GP
- variational distribution
- ELBO
- stochastic variational inference
- minibatch 学習
- inducing point allocation
- `SingleTaskVariationalGP`
- Bayesian Optimization での利用

までを順に整理します。

---

## 9.2 Exact GP の基本計算

通常の Gaussian Process regression では、学習データ

\[
X = [x_1, x_2, \dots, x_n]
\]

に対して共分散行列

\[
K_{XX}\in\mathbb{R}^{n\times n}
\]

を作ります。

観測ノイズを含めると

\[
K_y = K_{XX} + \sigma_n^2 I
\]

です。

posterior mean や marginal likelihood の計算では、典型的にこの行列の Cholesky 分解

\[
K_y = LL^\top
\]

を使います。

---

## 9.3 Exact GP の計算量

密な `n x n` 共分散行列に対する Cholesky 分解は、概ね

\[
O(n^3)
\]

の計算量を必要とします。

メモリも

\[
O(n^2)
\]

です。

したがって

```text
n = 1,000
```

程度なら Exact GP が現実的な場合がありますが、

```text
n = 10,000
n = 100,000
```

と増えるにつれて急速に厳しくなります。

もちろん実際の限界は

- GPU / CPU
- dtype
- batch size
- kernel
- number of outputs
- approximate linear algebra

などに依存するため、固定の閾値はありません。

---

## 9.4 なぜ Bayesian Optimization でも大規模 GP が必要になるのか

BO は一般に「評価が高価でデータ数が少ない」問題に向いています。

しかし実務では、

- 過去実験データが大量にある
- simulator を安く多数回評価できる
- active learning を長期間回してデータが蓄積する
- online optimization で観測が継続的に増える
- surrogate model を単なる BO 以外にも使う

というケースがあります。

このとき Exact GP の `O(n^3)` が問題になります。

---

## 9.5 Sparse GP の基本思想

大規模 GP では、全 `n` 点を同じ重さで直接扱う代わりに、少数の代表点を使って latent function を近似します。

この代表点を **inducing points** と呼びます。

入力位置を

\[
Z = [z_1, z_2, \dots, z_m]
\]

とし、通常

\[
m \ll n
\]

とします。

対応する latent function value を

\[
u = f(Z)
\]

とします。

```text
training points
x1 x2 x3 ... xn
        ↓
少数の inducing points Z
        ↓
latent summary u = f(Z)
        ↓
全体の GP を近似
```

というイメージです。

---

## 9.6 inducing point は単なる subsampling ではない

inducing point は、単に training data を `m` 点だけ残して残りを捨てる方法とは異なります。

単純な subsampling では、選ばれなかった観測を学習に使いません。

一方 Variational GP では、全観測データの likelihood を通じて

\[
q(u)
\]

という inducing variable の分布を学習します。

つまり

```text
subsampling
  → 観測データ自体を減らす

Variational GP
  → latent representation を m 個の inducing variables で近似しつつ
     全データの情報を ELBO 経由で利用する
```

という違いがあります。

---

## 9.7 inducing variables

inducing points `Z` に対応する latent values を

\[
u=f(Z)
\]

とします。

GP prior から

\[
p(u)=\mathcal{N}(0,K_{ZZ})
\]

が得られます。

また training latent values

\[
f=f(X)
\]

と `u` の joint distribution も Gaussian です。

Variational GP では、難しい posterior

\[
p(u\mid y)
\]

の代わりに

\[
q(u)
\]

を導入します。

---

## 9.8 Variational Distribution

代表的には

\[
q(u)=\mathcal{N}(m,S)
\]

と置きます。

ここで

- `m` は variational mean
- `S` は variational covariance

です。

これらを kernel hyperparameter や inducing point と一緒に最適化します。

Exact GP では posterior が解析的に得られる Gaussian regression でも、Variational GP ではあえて近似 posterior を使うことでスケーラビリティを得ます。

---

## 9.9 Variational Inference の目的

本当に欲しい posterior は

\[
p(u\mid y)
\]

ですが、これを直接扱う代わりに `q(u)` を近づけます。

理想的には

\[
q(u) \approx p(u\mid y)
\]

となるように学習したいわけです。

そのために使うのが Evidence Lower Bound、つまり **ELBO** です。

---

## 9.10 ELBO

log marginal likelihood は

\[
\log p(y)
\]

ですが、Variational GP ではこれを直接最大化する代わりに下限

\[
\mathcal{L}_{ELBO}
\le
\log p(y)
\]

を最大化します。

基本形は

\[
\mathcal{L}_{ELBO}
=
\mathbb{E}_{q(f)}[\log p(y\mid f)]
-
\operatorname{KL}(q(u)\|p(u))
\]

です。

2項の役割を分けると、

```text
期待 log likelihood
  → 観測データへよく適合させる

KL divergence
  → variational posterior が prior から不必要に離れすぎないようにする
```

となります。

---

## 9.11 ELBO を直感的に見る

ELBO は概念的には

```text
データへの適合
    -
近似 posterior の複雑さ
```

です。

これは通常の marginal likelihood にある

```text
fit
  +
Occam's razor
```

と似た考え方を variational approximation の中で実現しています。

---

## 9.12 Variational GP の計算量

inducing point 数を `m`、観測数を `n` とすると、標準的な sparse variational GP では典型的に

\[
O(nm^2 + m^3)
\]

程度の計算が中心になります。

`m << n` なら Exact GP の

\[
O(n^3)
\]

より大幅に軽くできます。

mini-batch size を `b` とすると、1 step あたりのデータ依存部分は概ね

\[
O(bm^2)
\]

の形になり、全データを毎 step 使わずに学習できます。

厳密な計算量は variational strategy や implementation に依存しますが、重要なのは

> `n` ではなく小さい `m` を中心とした線形代数に置き換える

という点です。

---

## 9.13 `m` は重要な hyperparameter

inducing point 数 `m` を増やすと、一般に近似精度は改善しやすくなります。

しかし

- 計算量が増える
- メモリが増える
- optimization が重くなる

ため、トレードオフがあります。

```text
m 小
  + 高速
  - 近似が粗い可能性

m 大
  + Exact GP に近づきやすい
  - 計算が重い
```

です。

---

## 9.14 BoTorch の default inducing point 数

BoTorch 0.18.1 の `SingleTaskVariationalGP` では、`inducing_points` を明示しない場合、heuristic として

\[
m \approx 0.25n
\]

つまり training points の 25% が使われます。

例えば

```text
n = 400
```

なら、おおよそ

```text
m = 100
```

です。

ただしこれは一般解ではありません。

大規模データでは `m` を明示的に設定して精度と計算時間を検証することが重要です。

---

## 9.15 inducing point allocation

inducing point はどこに置くかも重要です。

BoTorch の `SingleTaskVariationalGP` では、default allocator として

```python
GreedyVarianceReduction
```

が使われます。

これは GP posterior / kernel variance を効率よく表現するように inducing point を配置する考え方です。

BoTorch では downstream task に合わせた custom allocator も利用できます。

---

## 9.16 inducing point を学習する

`SingleTaskVariationalGP` には

```python
learn_inducing_points = True
```

があります。

これにより inducing point の位置自体も学習対象にできます。

```text
初期配置
  ↓
variational training
  ↓
Z の位置も更新
  ↓
データをよりよく近似する位置へ移動
```

という挙動です。

ただし inducing point の optimization が不安定になるケースもあるため、固定点との比較も有効です。

---

## 9.17 robotorchan の `SingleTaskVariationalGP`

robotorchan では BoTorch のモデルを薄く wrap しています。

constructor は主に

```python
from robotorchan.models import SingleTaskVariationalGP

model = SingleTaskVariationalGP(
    train_X=train_X,
    train_Y=train_Y,
    inducing_points=64,
)
```

のように利用できます。

主な引数は

- `train_X`
- `train_Y`
- `likelihood`
- `num_outputs`
- `learn_inducing_points`
- `covar_module`
- `mean_module`
- `variational_distribution`
- `variational_strategy`
- `inducing_points`
- `inducing_point_allocator`
- `outcome_transform`
- `input_transform`

です。

---

## 9.18 Exact GP と学習 API が違う

通常の `SingleTaskGP` は Exact Marginal Log Likelihood を使います。

```python
mll = model.make_mll()
fit_gpytorch_mll(mll)
```

一方 `SingleTaskVariationalGP` の `make_mll()` は

```python
VariationalELBO
```

を返します。

robotorchan では

```python
mll = model.make_mll()
```

という共通 API を維持しつつ、内部の objective は Exact GP と異なります。

---

## 9.19 `make_mll()` の実体

robotorchan の wrapper では概念的に

```python
from gpytorch.mlls import VariationalELBO

mll = VariationalELBO(
    likelihood=model.likelihood,
    model=model.model,
    num_data=num_data,
)
```

を作ります。

`num_data` を省略した場合は

```python
model.raw_train_X.shape[-2]
```

を使います。

---

## 9.20 `num_data` が重要な理由

mini-batch 学習では、1回の forward に使う batch size と全データ数が異なります。

例えば

```text
全データ n = 100,000
mini-batch b = 512
```

でも ELBO は全データを代表する objective でなければなりません。

そのため

```python
mll = model.make_mll(num_data=100_000)
```

のように全観測数を渡します。

`num_data=512` としてしまうと、KL term と likelihood term の相対スケーリングが誤ります。

---

## 9.21 基本的な full-batch 学習例

```python
import torch
from robotorchan.models import SingleTaskVariationalGP

torch.set_default_dtype(torch.double)

train_X = torch.rand(500, 3)
train_Y = torch.sin(2 * torch.pi * train_X[:, :1]) + 0.2 * train_X[:, 1:2] - 0.1 * train_X[:, 2:3]

model = SingleTaskVariationalGP(
    train_X=train_X,
    train_Y=train_Y,
    inducing_points=64,
)

mll = model.make_mll()
```

Variational GP は通常、gradient optimizer を使って ELBO を反復最適化します。

---

## 9.22 optimizer による学習

概念的には次のように学習します。

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
model.likelihood.train()

for _ in range(200):
    optimizer.zero_grad()
    output = model(train_X)
    loss = -mll(output, train_Y.squeeze(-1))
    loss.backward()
    optimizer.step()
```

GPyTorch の likelihood や output shape に応じて target shape は調整が必要です。

重要なのは、Exact GP のように「MLL を一度 optimizer helper に渡す」だけではなく、SGD / Adam 系 optimizer を用いた反復学習が自然だという点です。

---

## 9.23 Stochastic Variational GP

Variational GP の大きな利点は、likelihood term がデータ点ごとに分解できるため minibatch 化しやすいことです。

```text
100,000 points
    ↓
mini-batch 512 points
    ↓
ELBO gradient
    ↓
parameter update
    ↓
次の mini-batch
```

これにより大規模データでも通常の neural network に近い training loop を使えます。

---

## 9.24 minibatch 学習の考え方

DataLoader を使うと概念的には

```python
for batch_X, batch_Y in loader:
    optimizer.zero_grad()
    output = model(batch_X)
    loss = -mll(output, batch_Y.squeeze(-1))
    loss.backward()
    optimizer.step()
```

となります。

このとき `mll` は

```python
mll = model.make_mll(num_data=len(dataset))
```

として、全データ数を使う必要があります。

---

## 9.25 `train_X` は全データでなくてもよい

BoTorch の `SingleTaskVariationalGP` は、constructor の `train_X` が必ずしも全学習データである必要はありません。

これは inducing point allocation と初期化に使うデータとして扱えます。

大規模データでは

```text
全データ
  ↓
初期化用 subset
  ↓
inducing point allocation
  ↓
その後 minibatch training
```

という使い方もできます。

ただし robotorchan の `raw_train_X` は constructor に渡した tensor の snapshot なので、全データそのものとは限りません。

---

## 9.26 raw-data API の注意

robotorchan では

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
```

を持ちます。

`SingleTaskVariationalGP` の upstream constructor には `train_Yvar` がないため、

```python
model.raw_train_Yvar is None
```

です。

また `raw_train_X` / `raw_train_Y` は constructor 入力の provenance であり、後続 minibatch で学習した全データ履歴ではありません。

---

## 9.27 `train_Y` は optional

BoTorch 0.18.1 の `SingleTaskVariationalGP` では `train_Y` は optional です。

これは model structure と inducing point を `train_X` だけで構築し、その後別途 training loop で target を与える使い方ができるためです。

例えば

```python
model = SingleTaskVariationalGP(
    train_X=initial_X,
    inducing_points=64,
)
```

のような構築も可能です。

---

## 9.28 Gaussian likelihood 以外

BoTorch のドキュメントでは `SingleTaskVariationalGP` は大量データだけでなく、non-Gaussian response を扱う場合にも候補とされています。

Variational inference は Gaussian likelihood で解析的 posterior が得られないケースにも拡張しやすいためです。

ただし Bayesian Optimization で custom likelihood を使う場合は、

- posterior semantics
- observation noise
- acquisition function compatibility

を確認する必要があります。

---

## 9.29 multiple outputs

`SingleTaskVariationalGP` は

```python
num_outputs > 1
```

も扱えます。

ただし BoTorch 0.18.1 では複数 output を **independent な batch** として扱います。

つまり

```text
output 1 ─ GP
output 2 ─ GP
output 3 ─ GP
```

であり、`MultiTaskGP` のように cross-output covariance を積極的に共有するモデルではありません。

関連出力間で情報共有したい場合は、別の multi-task model を検討します。

---

## 9.30 Variational Strategy

GPyTorch では inducing variables をどう parameterize し、posterior approximation をどう構成するかを

```python
VariationalStrategy
```

で表現します。

BoTorch の default は whitening を利用する `VariationalStrategy` です。

whitening により variational optimization の conditioning を改善しやすくなります。

---

## 9.31 Variational Distribution

default では

```python
CholeskyVariationalDistribution
```

が使われます。

これは

\[
q(u)=\mathcal{N}(m,S)
\]

の full covariance `S` を Cholesky parameterization で表現します。

inducing point 数が増えると `S` の parameter 数も増えるため、`m` を極端に大きくすることにはコストがあります。

---

## 9.32 input / outcome transform の注意

BoTorch 0.18.1 は `SingleTaskVariationalGP` で minibatch training を行う場合、learnable transform に注意するよう明示しています。

例えば

```python
Normalize(...)
Standardize(...)
```

の統計量が mini-batch ごとに更新されると、batch ごとに scale が変わってしまいます。

これは望ましくありません。

---

## 9.33 大規模データでは事前変換を推奨

minibatch training では、全データから事前に

- input normalization parameters
- output mean / standard deviation

を計算し、dataset 自体を pre-transform しておく方が安全です。

```text
全 training data
    ↓
一度だけ normalization / standardization
    ↓
固定済み transformed dataset
    ↓
mini-batch training
```

とします。

これにより各 mini-batch で統計量が変化する問題を避けられます。

---

## 9.34 Exact GP と Variational GP の違い

| 観点 | Exact GP | Variational GP |
|---|---|---|
| posterior | exact | approximate |
| 主な行列サイズ | `n x n` | `m x m` |
| training objective | exact MLL | ELBO |
| 計算量の目安 | `O(n^3)` | `O(nm^2 + m^3)` |
| minibatch | 基本的に不向き | 得意 |
| inducing points | 不要 | 必要 |
| optimization | MLL最適化 | gradient-based ELBO optimization |
| 大量データ | 苦手 | 得意 |

---

## 9.35 Approximation error

Variational GP は近似モデルなので、Exact GP と比べて posterior が変わります。

特に `m` が小さすぎると

- posterior mean が過度に滑らかになる
- local structure を捉えられない
- uncertainty が不正確になる
- BO candidate が変わる

可能性があります。

したがって prediction RMSE だけでなく、uncertainty calibration も確認する必要があります。

---

## 9.36 BO では uncertainty が特に重要

通常の回帰なら posterior mean が十分正確なら実用になる場合があります。

しかし BO では acquisition function が

\[
\mu(x)
\]

だけでなく

\[
\sigma(x)
\]

を使います。

Variational approximation が uncertainty を歪めると、探索と活用のバランスが変わります。

そのため BO 用 Variational GP では

> mean prediction の精度だけでなく posterior variance の品質

が特に重要です。

---

## 9.37 inducing point allocation と BO

global regression accuracy に適した inducing points と、optimum 周辺を正確に表現する inducing points は必ずしも同じではありません。

BoTorch の設計でも custom `InducingPointAllocator` を渡せます。

BO では

```text
全領域を均一に近似
```

よりも

```text
改善が期待できる領域を高精度に近似
```

した方がよい場合があります。

これは今後 robotorchan で BO-oriented allocator を拡張する余地があります。

---

## 9.38 Variational GP と Active Learning

Active Learning では大量 pool に対して posterior uncertainty を計算することがあります。

Variational GP は training data が大きい場合だけでなく、

```text
large training data
    +
large candidate pool
```

の組み合わせでも有用です。

ただし posterior evaluation cost も inducing point 数に依存するため、pool size が非常に大きい場合は batch evaluation を使います。

---

## 9.39 Variational GP と高次元 GP は別軸

前章の SAAS と本章の Variational GP は、解決する問題が異なります。

```text
SAAS
  → d が大きく、少数変数だけが重要

Variational GP
  → n が大きく、Exact GP が重い
```

したがって

```text
high d + small n
```

なら SAAS が自然ですが、

```text
low d + large n
```

なら Variational GP が自然です。

---

## 9.40 high d + large n は難しい

もし

```text
high dimensional
    +
large dataset
```

なら、SAAS と通常 Variational GP のどちらか一方だけで完全に解決できるとは限りません。

この場合は

- sparse variational model
- structured kernel
- additive structure
- dimensionality reduction
- DKL
- local GP

など複数の工夫が必要になる場合があります。

---

## 9.41 Variational GP と Random Forest / Neural Network

大量データになると GP 以外の surrogate も候補になります。

Variational GP の利点は、

- GP 的な posterior uncertainty
- kernel prior
- BoTorch acquisition function との統合

を維持しやすい点です。

一方で単純な point prediction だけなら tree model や neural network がより高速な場合があります。

BO では uncertainty quality と optimization performance を含めて比較します。

---

## 9.42 inducing point 数の選び方

`m` を選ぶときは、固定ルールより実験的比較が重要です。

例えば

```text
m = 32
m = 64
m = 128
m = 256
```

を比較し、

- validation error
- negative log predictive density
- calibration
- training time
- memory
- BO regret / best value

を確認します。

`m` を増やしても性能がほとんど改善しなくなった地点が実務的な候補です。

---

## 9.43 inducing point の位置を確認する

低次元なら inducing point を可視化できます。

```text
training data cloud
    +
inducing points
```

を重ねて、

- domain 全体をカバーしているか
- boundary に偏っていないか
- optimum 周辺を表現できているか

を確認します。

高次元なら PCA / UMAP は診断表示として使えますが、元空間での距離情報が歪む点に注意します。

---

## 9.44 学習率と optimization

Variational GP は neural network と同様に iterative optimizer を使うため、

- learning rate
- number of epochs
- batch size
- scheduler
- initialization

の影響を受けます。

Exact GP 以上に「training procedure 自体」が性能へ影響します。

ELBO が安定していない状態で posterior を BO に使わないことが重要です。

---

## 9.45 convergence の確認

最低限、

```text
ELBO / loss
```

の推移を確認します。

さらに

- validation likelihood
- prediction RMSE
- posterior variance
- inducing point movement
- kernel hyperparameters

も確認するとよいです。

loss が下がっていても、BO に必要な uncertainty が良いとは限りません。

---

## 9.46 minibatch size

mini-batch size `b` も重要です。

```text
b 小
  + memory small
  + stochasticity large
  - gradient variance large

b 大
  + stable gradient
  - memory large
  - 1 step expensive
```

GP では kernel computation もあるため、GPU memory と inducing point 数を見ながら調整します。

---

## 9.47 noise の扱い

Variational GP でも likelihood を通じて observation noise を学習します。

Gaussian likelihood なら概念的には

\[
y=f(x)+\epsilon,
\quad
\epsilon\sim\mathcal{N}(0,\sigma_n^2)
\]

です。

大量データだから noise model が不要になるわけではありません。

むしろ多量の観測があると heteroscedasticity や non-Gaussian noise が見えやすくなる場合があります。

---

## 9.48 posterior prediction

学習後は Exact GP wrapper と同じ BoTorch model interface で

```python
posterior = model.posterior(test_X)
```

を使えます。

したがって acquisition function 側からは

```text
Exact GP
Variational GP
```

の違いを大きく意識せずに posterior を利用できます。

これは BoTorch-native wrapper にする大きな利点です。

---

## 9.49 acquisition function との組み合わせ

`SingleTaskVariationalGP` は BoTorch model interface を実装しているため、通常の MC acquisition function と組み合わせられます。

例えば

- qLogEI
- qLogNEI
- qUCB
- qPI

などです。

ただし acquisition value は approximate posterior に依存するため、surrogate approximation error が candidate selection に直接影響します。

---

## 9.50 large-data BO の loop

大規模データで BO を行う場合は、概念的に

```text
large historical dataset
    ↓
Variational GP fitting
    ↓
posterior
    ↓
acquisition optimization
    ↓
new experiment
    ↓
dataset append
    ↓
retraining / incremental-style update
```

となります。

毎 iteration でゼロから大量データを full training すると重いため、warm start や一定 epoch の更新も実務上重要になります。

---

## 9.51 Exact GP から切り替える判断

固定の `n` 閾値を決めるより、次を測るのがよいです。

1. Exact GP fitting time
2. posterior evaluation time
3. peak memory
4. BO iteration total time
5. prediction / calibration quality

例えば Exact GP が 1 iteration 数秒なら、近似モデルへ変える必要はありません。

一方 model fitting が実験時間より長くなってきたら、Variational GP を検討する価値があります。

---

## 9.52 小規模データでは Exact GP が第一候補

Variational GP は近似を導入するため、小規模データで無理に使う必要はありません。

```text
small n
  → Exact GP

large n
  → Variational GPを検討
```

が基本です。

Exact GP が十分速いなら、posterior approximation error を増やさない Exact GP の方が自然です。

---

## 9.53 モデル選択フロー

```text
回帰問題
  ↓
データ数 n が Exact GP で現実的？
  ├─ Yes
  │    └─ SingleTaskGP など Exact GP
  │
  └─ No
       ↓
       inducing-point approximation を許容？
       ├─ Yes
       │    └─ SingleTaskVariationalGP
       │
       └─ No
            └─ 別の scalable surrogate も検討
```

さらに高次元なら

```text
n large?
  ↓
Yes → Variational approximation の必要性

同時に d high?
  ↓
Yes → dimension structure の対策も別途必要
```

と二軸で考えます。

---

## 9.54 実務上の比較表

| 状況 | 第一候補 |
|---|---|
| 小規模な通常回帰 | `SingleTaskGP` |
| 高次元・少数データ | SAAS 系 |
| 大規模データ | `SingleTaskVariationalGP` |
| 大規模 + mini-batch 必須 | `SingleTaskVariationalGP` |
| 多出力だが独立でよい | `SingleTaskVariationalGP(num_outputs=...)` |
| 出力間相関を使いたい | Multi-task 系を検討 |
| 大規模 + non-Gaussian likelihood | Variational GP 系を検討 |

---

## 9.55 よくある誤解

### 「Variational GP は高次元用」

主目的は `n` が大きい場合の scalability です。高次元対策とは別です。

### 「inducing points は学習データの subsampling」

違います。latent GP を表現する variational variables です。

### 「inducing point は多いほどよい」

精度は改善しやすいですが、計算量と memory が増えます。

### 「ELBO は Exact MLL と同じ」

違います。ELBO は log marginal likelihood の lower bound です。

### 「mini-batch size が `num_data`」

違います。`num_data` は全 training dataset の観測数です。

### 「Variational GP なら uncertainty は Exact GP と同じ」

近似 posterior なので異なる可能性があります。

---

## 9.56 robotorchan で覚えておく API

```python
model = SingleTaskVariationalGP(
    train_X=train_X,
    train_Y=train_Y,
    inducing_points=64,
)

mll = model.make_mll()

model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.posterior(test_X)
```

mini-batch 学習なら

```python
mll = model.make_mll(num_data=n_total)
```

とします。

---

## 9.57 診断項目

Variational GP を使う場合は少なくとも次を確認します。

1. ELBO convergence
2. validation RMSE / MAE
3. predictive log likelihood
4. uncertainty calibration
5. inducing point 数に対する感度
6. inducing point 配置
7. learning rate
8. mini-batch size
9. Exact GP と比較できる subset での差
10. BO performance

特に 9 は有効です。

小さな subset 上で Exact GP と Variational GP を比較すると、近似による posterior distortion を確認できます。

---

## 9.58 理論上の整理

Variational GP の中心は、

\[
p(f\mid y)
\]

を直接扱う代わりに inducing variables `u` を導入し、

\[
q(u)=\mathcal{N}(m,S)
\]

を学習することです。

その objective が

\[
\mathcal{L}_{ELBO}
=
\mathbb{E}_{q(f)}[\log p(y\mid f)]
-
\mathrm{KL}(q(u)\|p(u))
\]

です。

これにより Exact GP の `n x n` 行列計算を、`m << n` の inducing representation に置き換えられます。

---

## 9.59 まとめ

Variational GP は、**大量データに対して GP の probabilistic surrogate を維持しながら計算量を下げる方法**です。

```text
Exact GP
  n x n covariance
  O(n^3)

Variational GP
  m inducing points
  m << n
  ELBO optimization
  mini-batch possible
```

robotorchan では

```python
SingleTaskVariationalGP
```

を使います。

特に覚えておきたい点は次の通りです。

1. 高次元問題と大規模データ問題は別
2. inducing point は単純 subsampling ではない
3. `q(u)` で posterior を variational approximation する
4. ELBO は log marginal likelihood の lower bound
5. `m` が近似精度と計算コストを支配する
6. mini-batch 学習では `num_data` に全データ数を渡す
7. minibatch 時の learnable transform には注意する
8. BO では posterior mean だけでなく uncertainty approximation が重要
9. Exact GP が十分速いなら無理に Variational GP を使わない

次章では、人の好みや比較データを扱う [Preference Learning](10_preference_learning.md) を説明します。

## 参考

- Hensman, Fusi, Lawrence, *Gaussian Processes for Big Data*, UAI 2013
- Burt, Rasmussen, van der Wilk, *Convergence of Sparse Variational Inference in Gaussian Process Regression*, JMLR 2020
- Moss, Ober, Picheny, *Inducing Point Allocation for Sparse Gaussian Processes in High-Throughput Bayesian Optimization*, AISTATS 2023
- BoTorch 0.18.1 `SingleTaskVariationalGP`
- GPyTorch `VariationalELBO`
- [Gaussian Process](02_gaussian_process.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
