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

具体的なモデルの生成・学習方法は [モデルドキュメント](../models/README.md) を参照してください。

本章では、

- Exact GP の計算量
- inducing point
- sparse GP
- variational distribution
- ELBO
- stochastic variational inference
- minibatch 学習
- inducing point allocation
- Bayesian Optimization での利用

までを順に整理します。

---

### まず「dが大きい」と「nが大きい」を分ける

前章と本章の役割を図にすると、

~~~text
入力次元 d が大きい
  → 関数構造・探索空間が難しい
  → SAAS / additive / reduction / high-dimensional search

観測数 n が大きい
  → n × n covariance の線形代数が重い
  → inducing-point variational approximation
~~~

です。

もちろん $`d`$ と $`n`$ が同時に大きい問題もあります。その場合、
Variational GPへ変えるだけでは入力次元側の難しさは解消されません。

## 9.2 Exact GP の基本計算

通常の Gaussian Process regression では、学習データ

$$
X = [x_1, x_2, \dots, x_n]
$$

に対して共分散行列

$$
K_{XX}\in\mathbb{R}^{n\times n}
$$

を作ります。

観測ノイズを含めると

$$
K_y = K_{XX} + \sigma_n^2 I
$$

です。

posterior mean や marginal likelihood の計算では、典型的にこの行列の Cholesky 分解

$$
K_y = LL^\top
$$

を使います。

---

## 9.3 Exact GP の計算量

密な `n x n` 共分散行列に対する Cholesky 分解は、概ね

$$
O(n^3)
$$

の計算量を必要とします。

メモリも

$$
O(n^2)
$$

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

この入力位置を **inducing points** と呼びます。

ただし「代表的なtraining rows」という理解だけでは不十分です。重要なのは、
その位置に対応する **inducing variables** を介してlatent GP posteriorを近似することです。

入力位置を

$$
Z = [z_1, z_2, \dots, z_m]
$$

とし、通常

$$
m \ll n
$$

とします。

対応する latent function value を

$$
u = f(Z)
$$

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

$$
q(u)
$$

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

$$
u=f(Z)
$$

とします。

GP prior から

$$
p(u)=\mathcal{N}(0,K_{ZZ})
$$

が得られます。

また training latent values

$$
f=f(X)
$$

と `u` の joint distribution も Gaussian です。

Variational GP では、難しい posterior

$$
p(u\mid y)
$$

の代わりに

$$
q(u)
$$

を導入します。

---

## 9.8 Variational Distribution

代表的には

$$
q(u)=\mathcal{N}(m,S)
$$

と置きます。

ここで

- `m` は variational mean
- `S` は variational covariance

です。

これらを kernel hyperparameter や inducing point と一緒に最適化します。

Exact GP では posterior が解析的に得られる Gaussian regression でも、Variational GP ではあえて近似 posterior
を使うことでスケーラビリティを得ます。

---

## 9.9 Variational Inference の目的

本当に欲しい posterior は

$$
p(u\mid y)
$$

ですが、これを直接扱う代わりに `q(u)` を近づけます。

理想的には

$$
q(u) \approx p(u\mid y)
$$

となるように学習したいわけです。

そのために使うのが Evidence Lower Bound、つまり **ELBO** です。

---

## 9.10 ELBO

log marginal likelihood は

$$
\log p(y)
$$

ですが、Variational GP ではこれを直接最大化する代わりに下限

$$
\mathcal{L}_{ELBO}
\le
\log p(y)
$$

を最大化します。

基本形は

$$
\mathcal{L}_{ELBO}
=
\mathbb{E}_{q(f)}[\log p(y\mid f)]
-
\operatorname{KL}(q(u)\|p(u))
$$

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

ELBOは概念的には

```text
観測を説明する期待 log likelihood
    -
q(u) を prior p(u) から動かすコスト
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

$$
O(nm^2 + m^3)
$$

程度の計算が中心になります。

`m << n` なら Exact GP の

$$
O(n^3)
$$

より大幅に軽くできます。

mini-batch size を `b` とすると、1 step あたりのデータ依存部分は概ね

$$
O(bm^2)
$$

の形になり、全データを毎 step 使わずに学習できます。

厳密な計算量は variational strategy や implementation に依存しますが、重要なのは

> `n` ではなく小さい `m` を中心とした線形代数に置き換える

という点です。

---

## 9.13 `m` は重要な hyperparameter

inducing point 数 `m` を増やすとvariational familyの表現力は増えますが、実際の予測精度が
単調に改善するとは限りません。inducing point配置やoptimizationも影響します。

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

## 9.14 Inducing point 数に一般的な固定比率はない

Inducing point 数 $m$ は、近似精度と計算負荷のトレードオフを決めるモデル選択パラメータです。
$m/n$ に普遍的な最適比率はなく、データの複雑さ・カーネル・誘導点配置に依存します。
予測精度と不確実性の較正を複数の $m$ で検証します。

---

## 9.15 inducing point allocation

Inducing point の配置は、近似事後分布の精度と計算負荷に影響します。観測点からの選択、空間充填、クラスタリング、カーネルに基づく配置などの選択肢があります。配置の良否は、近似したい関数の変動と入力分布に依存します。

---

## 9.16 inducing point を学習する

誘導点位置 $Z$ を固定する方法と、変分パラメータやカーネルパラメータとともに最適化する方法があります。位置を最適化すると表現力が上がる一方、非凸最適化による初期値依存性が増えるため、固定配置との比較が有用です。

---

## 9.17 Variational GP の推論対象

Inducing variables $u=f(Z)$ と変分分布 $q(u)$ を導入し、
$p(f\mid u)$ を用いて近似事後分布 $q(f)$ を構成します。
推論アルゴリズムとモデルの入力構造は独立した設計軸です。

---

## 9.18 Exact GP と Variational GP の目的関数

Gaussian likelihood を用いる Exact GP は周辺対数尤度 $\log p(y)$ を直接評価できます。
Variational GP は一般に $\log p(y)$ の下界である ELBO を最大化します。
両者の最適化目的を同一視してはいけません。

---

## 9.19 ELBO と KL 正則化

$
\mathcal L=\sum_{i=1}^{n}\mathbb E_{q(f_i)}[\log p(y_i\mid f_i)]
-\operatorname{KL}(q(u)\Vert p(u)).
$

第一項は観測データへの適合、第二項は誘導変数の変分分布と事前分布の乖離を表します。

---

## 9.20 Mini-batch ELBO のスケーリング

観測 $n$ 件から一様に抽出した mini-batch $B$（サイズ $b$）では、

$
\widehat{\mathcal L}_B
=\frac{n}{b}\sum_{i\in B}\mathbb E_{q(f_i)}[\log p(y_i\mid f_i)]
-\operatorname{KL}(q(u)\Vert p(u)).
$

これは独立観測・一様サンプリングのもとで全データ ELBO の不偏推定量です。
非一様サンプリングでは適切な重み付けが必要です。

---

## 9.21 Full-batch 最適化の理論

全観測を使う場合は ELBO の勾配を直接計算し、変分パラメータ、カーネルパラメータ、
必要に応じて誘導点位置を同時または交互に更新します。

---

## 9.22 Stochastic optimization

Mini-batch の確率的勾配を用いる場合、学習率、batch size、誘導点位置の更新が
近似事後分布の品質に影響します。ELBO の改善だけでなく予測分布を検証します。

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

Mini-batch 学習では、観測ごとの期待対数尤度をサンプリングして推定し、KL 項との相対スケールを保つ必要があります。独立観測から一様に抽出したサイズ $b$ の batch では、尤度項に $n/b$ を掛けます。導出は[9.20](#920-mini-batch-elbo-のスケーリング)を参照してください。

---

## 9.25 誘導点の初期化と観測データ

誘導点 $Z$ は観測点の部分集合に限定されません。
観測点からの初期化、クラスタリング、最適化可能な位置パラメータなどを考えられます。
初期化に用いるデータと ELBO の評価に用いる全観測は概念的に区別します。

---

## 9.26 データ来歴と統計的推論

モデルの初期化に使う観測、実際に ELBO に寄与する観測、予測評価に使う観測は
区別する必要があります。データ来歴の保持方法はモデル API の責務です。

---

## 9.27 Variational approximation と入力構造

変分近似は主として推論の計算量に関わります。
カテゴリ変数や階層入力の意味を保持するには、それとは別に適切な共分散を指定します。
近似推論を導入しただけで Mixed Variables に対応するわけではありません。

---

## 9.28 Gaussian likelihood 以外

Gaussian likelihood では共役性により Exact GP の事後分布を解析的に計算できます。一方、分類などの非 Gaussian likelihood では一般に事後分布が非共役となり、変分近似が有用です。ただし潜在関数の事後分布と観測ラベルの予測分布は区別する必要があります。

---

## 9.29 multiple outputs

複数出力を扱う場合、出力ごとに独立した変分 GP を置く構成と、共有潜在過程を使って出力間共分散をモデル化する構成は異なります。前者の出力間共分散はゼロであり、後者は cross-output uncertainty を表現できます。詳しくは [Multi-task / Multi-output](07_multitask_multioutput.md) を参照してください。

---

## 9.30 Variational Strategy

変分戦略は、誘導変数 $u$ と近似分布 $q(u)$ から $q(f)$ を構成する方法を定めます。Whitening は事前共分散で再パラメータ化することで、変分最適化の条件数を改善する場合があります。具体的なパラメータ化は推論手法の選択に依存します。

---

## 9.31 Variational Distribution

代表的な構成では `CholeskyVariationalDistribution` のようなGaussian
variational distributionを利用します。実際のdefaultは利用中のupstream versionと
constructor contractを確認します。

これは

$$
q(u)=\mathcal{N}(m,S)
$$

の full covariance `S` を Cholesky parameterization で表現します。

inducing point 数が増えると `S` の parameter 数も増えるため、`m` を極端に大きくすることにはコストがあります。

---

## 9.32 Mini-batch と変換の整合性

入力正規化や出力標準化の統計量が mini-batch ごとに変わると、
異なるスケールの尤度項を同じ目的関数に混在させることになります。
学習データから推定した変換を固定するなど、一貫した尺度を維持します。

---

## 9.33 学習データ内での変換推定

正規化・標準化の統計量は、検証データを混ぜず学習部分から推定します。
その統計量を固定して各 mini-batch に適用することで、
変換の不整合と検証データのリークを防ぎます。

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

$$
\mu(x)
$$

だけでなく

$$
\sigma(x)
$$

を使います。

Variational approximation が uncertainty を歪めると、探索と活用のバランスが変わります。

そのため BO 用 Variational GP では

> mean prediction の精度だけでなく posterior variance の品質

が特に重要です。

---

## 9.37 inducing point allocation と BO

全入力領域の平均的な予測精度に適した誘導点配置と、獲得関数が重視する領域の不確実性を正確に近似する配置は必ずしも一致しません。探索領域に偏った配置は局所精度を高める一方、未探索領域での不確実性を過小評価する可能性があります。

---

## 9.38 Variational GP と Active Learning

Active Learning では大量 pool に対して posterior uncertainty を計算することがあります。

Variational GPはtraining dataが大きい場合のscalable surrogateとして利用できます。
Active Learningでcandidate poolも大きい場合は、training scalabilityに加えて
posterior evaluation自体のbatchingやmemoryも別途考えます。

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

$$
y=f(x)+\epsilon,
\quad
\epsilon\sim\mathcal{N}(0,\sigma_n^2)
$$

です。

大量データだから noise model が不要になるわけではありません。

むしろ多量の観測があると heteroscedasticity や non-Gaussian noise が見えやすくなる場合があります。

---

## 9.48 posterior prediction

変分 GP の予測分布は、誘導変数の近似事後分布を介して得られます。近似誤差は予測平均だけでなく、予測分散にも現れます。特にベイズ最適化では獲得関数が不確実性に依存するため、分散の較正を検証する必要があります。

---

## 9.49 acquisition function との組み合わせ

獲得関数は近似事後分布から計算されるため、近似誤差が候補選択へ直接影響します。Expected Improvement や UCB など、予測分散に依存する基準では特にその影響が大きくなり得ます。近似モデルと Exact GP を小規模データで比較することが有用です。

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

モデル選択では、まず観測数 $n$ による計算負荷と、入力次元 $d$ による統計的な難しさを分けます。Exact GP が十分に扱える場合は近似の追加を避ける選択が合理的です。大規模データでは誘導点近似を検討し、高次元性には別途構造仮定や次元削減を検討します。

---

## 9.54 問題構造の比較表

| 問題 | 対応する考え方 |
|---|---|
| Exact GPが十分軽い | Exact GPを維持できる |
| large nでdense Exact GPがボトルネック | Variational GPを検討 |
| high d + small n | 高次元構造の対策を検討 |
| high d + large n | scalabilityとdimension structureの両方が必要 |
| mixed input + large n | Mixed Variational GPを検討 |
| independent multi-output | batched / independent output構成を検討 |
| cross-output sharingが必要 | multi-task構造を別途検討 |

---

## 9.55 よくある誤解

### 「Variational GP は高次元用」

主目的は `n` が大きい場合の scalability です。高次元対策とは別です。

### 「inducing points は学習データの subsampling」

違います。latent GP を表現する variational variables です。

### 「inducing point は多いほどよい」

精度は改善しやすいですが、計算量と memory が増えます。

### 「ELBO は Exact MLL と同じ」

違います。標準的な variational GP では ELBO は log marginal likelihood の lower bound です。
ただし、実装で追加の regularization や別 objective を組み合わせる場合まで
「常に同じ ELBO」と一般化しないことが重要です。

### 「mini-batch size が `num_data`」

違います。`num_data` は全 training dataset の観測数です。

### 「Variational GP なら uncertainty は Exact GP と同じ」

近似 posterior なので異なる可能性があります。

---

## 9.56 実装 API との境界

学習器の生成、ELBO オブジェクトの取得、mini-batch の指定、事後分布取得などの
ライブラリ固有の呼び出し方は[モデルドキュメント](../models/README.md)で扱います。
本章では $q(u)$、ELBO、近似誤差、計算量および不確実性の理論を扱います。

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

$$
p(f\mid y)
$$

を直接扱う代わりに inducing variables `u` を導入し、

$$
q(u)=\mathcal{N}(m,S)
$$

を学習することです。

その objective が

$$
\mathcal{L}_{ELBO}
=
\mathbb{E}_{q(f)}[\log p(y\mid f)]
-
\mathrm{KL}(q(u)\|p(u))
$$

です。

これにより Exact GP の `n x n` 行列計算を、`m << n` の inducing representation に置き換えられます。

---

## 9.59 この章で覚えておくこと

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

具体的なモデル API は [モデルドキュメント](../models/README.md) を参照してください。

特に覚えておきたい点は次の通りです。

1. 高次元問題と大規模データ問題は別
2. inducing point は単純 subsampling ではない
3. `q(u)` で posterior を variational approximation する
4. ELBO は log marginal likelihood の lower bound
5. `m` が近似精度と計算コストを支配する
6. mini-batch 学習では尤度項を全データ数に応じて正しくスケーリングする
7. minibatch 時の learnable transform には注意する
8. BO では posterior mean だけでなく uncertainty approximation が重要
9. Exact GP が十分速いなら無理に Variational GP を使わない

次章では、人の好みや比較データを扱う [Preference Learning](10_preference_learning.md) を説明します。

## 参考文献

1. Titsias, M. (2009).
   Variational learning of inducing variables in sparse Gaussian processes.
   *Proceedings of AISTATS*, 5, 567-574.
2. Hensman, J., Fusi, N., and Lawrence, N. D. (2013).
   Gaussian processes for big data. *Proceedings of UAI*.
3. Burt, D. R., Rasmussen, C. E., and van der Wilk, M. (2020).
   Convergence of sparse variational inference in Gaussian process regression.
   *Journal of Machine Learning Research*, 21.
4. BoTorch `SingleTaskVariationalGP` and GPyTorch `VariationalELBO` documentation.
5. [Gaussian Process](02_gaussian_process.md)
6. [High-dimensional GP](08_high_dimensional_gp.md)
