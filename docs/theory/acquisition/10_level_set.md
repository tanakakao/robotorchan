# 10. Level-set Acquisition

## 10.1 最適値ではなく境界を学ぶ

Level-set estimation は、未知関数 \(f(x)\) に対して指定した threshold / target level \(t\) の上下を分類したり、

\[
f(x)=t
\]

となる境界を推定したりする問題です。

例えば、

- 規格値を超える条件領域
- 相転移や臨界条件の境界
- 安全 / 非安全領域の境界
- 合格 / 不合格を分ける response threshold

などが対象になります。

Bayesian Optimization が

```text
どこが最適か
```

を問うのに対し、Level-set estimation は

```text
どこで target level を横切るか
```

を問います。

## 10.2 Level set と excursion set

target level \(t\) に対して、

\[
L_t
=
\{x\in\mathcal X:f(x)=t\}
\]

を level set と考えます。

また、

\[
E_t^+
=
\{x:f(x)\ge t\}
\]

のような threshold 以上の領域を excursion set と呼ぶことがあります。

実際の experimental design では、境界そのものだけでなく \(E_t^+\) とその補集合を正しく分類することが目的になる場合があります。

## 10.3 Posterior による境界 uncertainty

Gaussian posterior

\[
f(x)\mid\mathcal D_n
\sim
\mathcal N(\mu_n(x),\sigma_n^2(x))
\]

を考えます。

境界学習で価値が高い候補は典型的に、

- posterior mean が target \(t\) に近い
- posterior uncertainty が大きい

という二つの性質を持ちます。

mean が target に近くても uncertainty がほぼゼロなら、すでに分類が確定している可能性があります。

逆に uncertainty が大きくても mean が target から極端に離れていれば、その点が boundary classification を変える可能性は低い場合があります。

## 10.4 Classification uncertainty と latent boundary

threshold に対する exceedance probability は

\[
P(f(x)\ge t\mid\mathcal D_n)
=
\Phi\left(
\frac{\mu_n(x)-t}{\sigma_n(x)}
\right)
\]

です。

この確率が 0.5 に近い領域では threshold の上下分類が不確実です。

ただし level-set acquisition は必ずしも exceedance probability の entropy を直接最大化する必要はありません。Straddle のように mean-to-target distance と uncertainty を組み合わせる基準もあります。

## 10.5 Straddle

Straddle は boundary learning の代表的な acquisition criterion です。

robotorchan で扱う形は

\[
\alpha_{\mathrm{straddle}}(x)
=
\beta\sigma_n(x)
-
|\mu_n(x)-t|
\]

です。

第1項は uncertainty を評価し、第2項は posterior mean が target から離れることを penalty とします。

```text
sigma が大きい
    → まだよく分からない

|mu - t| が小さい
    → boundary に近そう

両方を満たす
    → 高い Straddle score
```

## 10.6 Straddle の幾何学的解釈

posterior credible interval を概念的に

\[
[
\mu_n(x)-\beta\sigma_n(x),
\;
\mu_n(x)+\beta\sigma_n(x)
]
\]

と考えると、target \(t\) がこの interval に入り得る点は boundary classification が未確定な候補です。

Straddle score が高い領域は、この「credible interval が target をまたぐ」性質と強く関係します。

ただし \(\beta\) の定義や平方根の置き方は文献・実装により異なるため、parameter 名だけで比較せず式を確認する必要があります。

## 10.7 beta の役割

\(\beta\) を大きくすると uncertainty の寄与が増え、boundary 候補をより広く探索します。

小さくすると posterior mean が target に近い領域へ集中します。

```text
large beta
    exploration of uncertain boundary candidates

small beta
    concentration near current estimated boundary
```

理論的 confidence schedule を用いる level-set algorithms と、固定 \(\beta\) を tuning parameter として使う実装は区別する必要があります。

## 10.8 複数 threshold

複数の target levels

\[
t_1,\ldots,t_K
\]

を同時に学びたい場合、各 target に対する boundary utility を定義し、最大・和・重み付き和などで統合する方法が考えられます。

ただし reduction の選択は、

- どれか一つの boundary を優先するのか
- すべての boundaries を均等に学ぶのか
- 特定 threshold を重視するのか

という problem semantics を変えます。

したがって multi-boundary reduction は単なる tensor operation ではありません。

## 10.9 Multi-output と output selection

multi-output surrogate で特定 output の level set を学ぶ場合、どの output を target とするかを明示する必要があります。

例えば

\[
f(x)=(f_1(x),f_2(x),f_3(x))
\]

に対して \(f_2(x)=t\) の boundary を学ぶなら、acquisition は output 2 の posterior marginal を利用します。

複数 output の joint boundary を学ぶ問題は、単一 output の Straddle を各 output に独立適用するだけでは一般に定義できません。

## 10.10 Randomized Straddle

Randomized Straddle は deterministic な exploration coefficient を固定する代わりに、selection round ごとにランダムな coefficient を導入する考え方です。

robotorchan の実装が参照する randomized straddle 系の考え方では、正の random variable から探索係数を生成し、

\[
\alpha_{\mathrm{RStraddle}}(x)
=
\sqrt{\beta_t}\,\sigma_n(x)
-
|\mu_n(x)-t|
\]

のような score を使います。

係数をランダム化することで、固定した exploration strength に依存しすぎない boundary exploration を行います。

## 10.11 1 selection round で係数を固定する理由

Randomized Straddle で重要なのは、**同じ candidate selection round 内では同じ sampled coefficient を使う**ことです。

acquisition optimizer が同じ候補を評価するたびに coefficient を再サンプルすると、

```text
x を少し動かした
    +
random coefficient も変わった
```

となり、最適化対象そのものが不必要に揺れます。

したがって概念的には、

```text
selection round 開始
    ↓
beta_t を1回 sample
    ↓
その beta_t で acquisition を最適化
    ↓
candidate 決定
    ↓
次 round で resample
```

という状態遷移になります。

これは randomized acquisition と acquisition optimization を整合させる重要な設計です。

## 10.12 Randomized Straddle の provenance

Randomized Straddle は robotorchan 固有の名称を付けた heuristic ではなく、randomized straddle に関する既存研究に基づく acquisition family として扱います。

一方、使用する random coefficient distribution、parameterization、round semantics は実装 contract として明示する必要があります。

Theory では手法の原理を説明し、現在の具体的な実装挙動は [Level-set guide](../../optimization/level_set_learning.md) と [Acquisition integration status](../../optimization/acquisition-status.md) を正とします。

## 10.13 BoundaryVariance

BoundaryVariance は robotorchan-specific heuristic です。

posterior variance

\[
v(x)=\sigma_n^2(x)
\]

を使い、

\[
\alpha_{\mathrm{BV}}(x)
=
v(x)
\exp\left[
-\frac12
\left(
\frac{\mu_n(x)-t}{\sqrt{v(x)}}
\right)^2
\right]
\]

という形で、uncertainty と target proximity を組み合わせます。

## 10.14 BoundaryVariance の解釈

指数項

\[
\exp\left[
-\frac12
\left(
\frac{\mu-t}{\sigma}
\right)^2
\right]
\]

は standardized distance

\[
z=
\frac{\mu-t}{\sigma}
\]

が 0 に近いほど大きくなります。

したがって BoundaryVariance は、

```text
posterior variance が大きい
    ×
target が posterior distribution の中心付近にある
```

候補を重視します。

Straddle と似た目的を持ちますが、同じ acquisition の再パラメータ化ではありません。

## 10.15 BoundaryVariance に beta がない理由

BoundaryVariance の定義には Straddle のような \(\beta\) exploration parameter はありません。

uncertainty の scale と target proximity は variance と standardized Gaussian-shaped weight の中に組み込まれています。

したがって API 上の対称性だけを理由に \(\beta\) を追加すると、定義されていない parameter を持つ別の heuristic に変わってしまいます。

Theory と public API の双方で、この違いを維持することが重要です。

## 10.16 Literature method と library-specific heuristic を分ける

robotorchan の theory docs では provenance を明確にします。

```text
Straddle
    established level-set / boundary-learning criterion

Randomized Straddle
    literature-based randomized acquisition family

BoundaryVariance
    robotorchan-specific heuristic
```

library-specific heuristic を、文献で確立した named method であるかのように記述しません。

逆に heuristic であることは無価値という意味ではありません。目的、数式、適用範囲、既知の制約を明示すれば、実験的 acquisition として評価できます。

## 10.17 PosteriorVariance との違い

PosteriorVariance は

\[
\alpha(x)=\sigma_n^2(x)
\]

なので、target \(t\) を考慮しません。

Level-set acquisition は target proximity を利用するため、関数全体の uncertainty reduction より boundary learning に集中できます。

```text
PosteriorVariance
    function learning

Straddle / BoundaryVariance
    boundary learning
```

という目的の違いがあります。

## 10.18 Constrained BO との関係

未知制約

\[
c(x)\le0
\]

の boundary

\[
c(x)=0
\]

を学ぶことは level-set estimation と数学的に近い問題です。

しかし最終目的は異なります。

```text
Level-set estimation
    boundary / excursion set の推定が目的

Constrained BO
    feasible region 内の objective optimization が目的
```

Constrained BO では boundary uncertainty が重要でも、それだけで objective utility は決まりません。

## 10.19 Classification Active Learning との関係

binary classification の decision boundary learning も直感的には level-set estimation と似ています。

latent function \(g(x)\) に対して

\[
g(x)=0
\]

を class boundary とみなせるモデルもあります。

ただし classification では likelihood が non-Gaussian であり、class probability、latent posterior、label entropy など複数の uncertainty notion があります。

Gaussian regression の Straddle を classification acquisition とそのまま同一視しないことが重要です。

## 10.20 Batch Level-set Learning

q > 1 で複数点を同時選択する場合、pointwise Straddle score の上位 q 点では候補が同じ boundary region に集中する可能性があります。

真の batch level-set acquisition を設計するには、

- candidate correlation
- boundary information redundancy
- joint uncertainty reduction

などを定義する必要があります。

robotorchan-specific q=1 acquisitions が存在することは、任意 q の joint semantics が自動的に定義されることを意味しません。

## 10.21 Structured output と ensemble posterior

scalar Gaussian posterior を前提にした boundary score を、structured output や ensemble posterior へ拡張するには reduction semantics が必要です。

例えば ensemble members の variance を、

- within-member posterior uncertainty
- between-member disagreement

のどちらとして扱うかで意味が変わります。

そのため generic tensor reduction を追加するだけでは、level-set acquisition の意味を保証できません。

Theory 上の一般化可能性と、現在の public implementation contract を区別します。

## 10.22 Stopping criterion

Level-set estimation では、boundary classification uncertainty を使った stopping rule が考えられます。

例えば、

- target をまたぐ credible interval を持つ点がなくなる
- excursion-set classification probability が十分高くなる
- boundary acquisition の最大値が閾値以下になる
- boundary location uncertainty が要求精度以下になる

などです。

ただし acquisition score の絶対値は parameterization に依存するため、停止条件として利用する場合は calibration が必要です。

## 10.23 BoTorch / robotorchan との対応

robotorchan は level-set / boundary learning 用に Straddle、RandomizedStraddle、BoundaryVariance を提供します。

これらは現在、scalarized q=1 regression acquisition として明示的な適用範囲を持ち、ensemble posterior など未定義の reduction semantics は拒否します。

Theory は将来の拡張可能性を説明できますが、現在の利用可能範囲については次を参照してください。

- [Level-set learning guide](../../optimization/level_set_learning.md)
- [BoundaryVariance guide](../../optimization/boundary-variance.md)
- [Acquisition integration status](../../optimization/acquisition-status.md)

## 10.24 まとめ

Level-set acquisition は、

\[
f(x)=t
\]

という boundary を効率よく学ぶための観測価値を定義します。

```text
Straddle
    uncertainty - target distance

Randomized Straddle
    randomized exploration coefficient を持つ literature-based family

BoundaryVariance
    variance × standardized target proximity
    robotorchan-specific heuristic
```

という provenance と数式上の違いを明確にすることが重要です。

また、Level-set estimation、Constrained BO、Classification AL は boundary という共通要素を持ちますが、最終的な意思決定目的は異なります。
