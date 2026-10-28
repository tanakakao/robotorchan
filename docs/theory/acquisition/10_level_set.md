# 10. Level-set Acquisition

[能動学習章](09_active_learning.md)の学習目的と、正規分布の確率計算を前提とします。
閾値を $t$、潜在関数の平均と標準偏差を $\mu_n,\sigma_n$ とし、観測ラベルの分類とは区別します。

## 10.1 Function learningとLevel-set learningを分ける

~~~text
Function learning
    関数やtarget predictionsを広く学ぶ

Level-set estimation
    指定response levelの上下・境界を学ぶ
~~~

PosteriorVarianceのようなfunction-learning criterionは「どこが不確実か」を見ます。一方、level-set
criterionは「そのuncertaintyが指定thresholdの分類を変え得るか」まで見ます。

## 10.2 最適値ではなく境界を学ぶ

Level-set estimation は、未知関数 $`f(x)`$ に対して指定した threshold / target level $`t`$ の上下を分類したり、

$$
f(x)=t
$$

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

## 10.3 Level setとexcursion set

target level $`t`$ に対して、

$$
L_t
=
\{x\in\mathcal X:f(x)=t\}
$$

を level set と考えます。

また、

$$
E_t^+
=
\{x:f(x)\ge t\}
$$

のような threshold 以上の領域を excursion set と呼ぶことがあります。

実際の experimental design では、境界そのものだけでなく $`E_t^+`$ とその補集合を正しく分類することが目的になる場合があります。

## 10.4 Posterior による境界 uncertainty

Gaussian posterior

$$
f(x)\mid\mathcal D_n
\sim
\mathcal N(\mu_n(x),\sigma_n^2(x))
$$

を考えます。

境界学習で価値が高い候補は典型的に、

- posterior mean が target $`t`$ に近い
- posterior uncertainty が大きい

という二つの性質を持ちます。

mean が target に近くても uncertainty がほぼゼロなら、すでに分類が確定している可能性があります。

逆に uncertainty が大きくても mean が target から極端に離れていれば、その点が boundary classification を変える可能性は低い場合があります。

## 10.5 Classification uncertainty と latent boundary

threshold に対する exceedance probability は

$$
P(f(x)\ge t\mid\mathcal D_n)
=
\Phi\left(
\frac{\mu_n(x)-t}{\sigma_n(x)}
\right)
$$

です。

上の式は $\sigma_n(x)>0$ の場合です。分散零なら確定値で判定し、
$P(f(x)\ge t\mid D)=\mathbf1[\mu_n(x)\ge t]$ とします。
分散が正でこの確率が0.5に近い領域では、閾値の上下分類が不確実です。

ただし level-set acquisition は必ずしも exceedance probability の entropy を直接最大化する必要はありません。Straddle のように
mean-to-target distance と uncertainty を組み合わせる基準もあります。

## 10.6 Straddle

Straddle は boundary learning の代表的な acquisition criterion です。

$$
\alpha_{\mathrm{straddle}}(x)
=
\beta\sigma_n(x)
-
|\mu_n(x)-t|
$$

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

## 10.7 Straddleの幾何学的解釈

posterior credible interval を概念的に

$$
[
\mu_n(x)-\beta\sigma_n(x),
\;
\mu_n(x)+\beta\sigma_n(x)
]
$$

と考えると、target $`t`$ がこの interval に入り得る点は boundary classification が未確定な候補です。

実際、$\beta\ge0$ ではStraddleスコアが非負であることと
$|\mu_n(x)-t|\le\beta\sigma_n(x)$、すなわち閾値がこの区間内にあることは同値です。
ただしこれは候補ごとの区間による解釈であり、領域全体・全時刻についての同時保証とは別です。

ただし $`\beta`$ の定義や平方根の置き方は文献・実装により異なるため、parameter 名だけで比較せず式を確認する必要があります。

## 10.8 betaの役割

$`\beta`$ を大きくすると uncertainty の寄与が増え、boundary 候補をより広く探索します。

小さくすると posterior mean が target に近い領域へ集中します。

```text
large beta
    exploration of uncertain boundary candidates

small beta
    concentration near current estimated boundary
```

理論的 confidence schedule を用いる level-set algorithms と、固定 $`\beta`$ を tuning parameter
として使う実装は区別する必要があります。

## 10.9 複数 threshold

複数の target levels

$$
t_1,\ldots,t_K
$$

を同時に学びたい場合、各 target に対する boundary utility を定義し、最大・和・重み付き和などで統合する方法が考えられます。

ただし reduction の選択は、

- どれか一つの boundary を優先するのか
- すべての boundaries を均等に学ぶのか
- 特定 threshold を重視するのか

という problem semantics を変えます。

したがって multi-boundary reduction は単なる tensor operation ではありません。

## 10.10 Multi-outputとoutput selection

multi-output surrogate で特定 output の level set を学ぶ場合、どの output を target とするかを明示する必要があります。

例えば

$$
f(x)=(f_1(x),f_2(x),f_3(x))
$$

に対して $`f_2(x)=t`$ の boundary を学ぶなら、acquisition は output 2 の posterior marginal を利用します。

複数 output の joint boundary を学ぶ問題は、単一 output の Straddle を各 output に独立適用するだけでは一般に定義できません。

## 10.11 Randomized Straddle

Randomized Straddle は deterministic な exploration coefficient を固定する代わりに、selection round ごとにランダムな
coefficient を導入する考え方です。

$$
\alpha_{\mathrm{RStraddle}}(x)
=
\sqrt{\beta_t}\,\sigma_n(x)
-
|\mu_n(x)-t|
$$

のような score を使います。

係数をランダム化することで、固定した exploration strength に依存しすぎない boundary exploration を行います。

## 10.12 1 selection roundで係数を固定する理由

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

## 10.13 Randomized Straddleのprovenance

Inatsuら（2024）の規約では、各選択ラウンドで $\beta_t\sim\chi^2_2$ を引き、係数を $\sqrt{\beta_t}$ とします。
二自由度のカイ二乗分布は $P(\beta_t>a)=e^{-a/2}$（$a\ge0$）なので、
固定した平均・分散と $\sigma_n(x)>0$ の下では

$$
P(\alpha_{RStraddle}(x)>0)
=\exp\left(-\frac{(\mu_n(x)-t)^2}{2\sigma_n^2(x)}\right).
$$

これは候補の潜在値が閾値を上回る確率ではなく、無作為な係数の下でスコアが正となる確率です。
同じ係数をラウンド内で共有することで、そのラウンドには一つの比較可能な目的関数が定まります。

## 10.14 分散と境界近接度の積による基準

posterior variance

$$
v(x)=\sigma_n^2(x)
$$

を使い、

$$
\alpha_{\mathrm{BV}}(x)
=
v(x)
\exp\left[
-\frac12
\left(
\frac{\mu_n(x)-t}{\sqrt{v(x)}}
\right)^2
\right]
$$

という形で、uncertainty と target proximity を組み合わせます。

$v(x)=0$ ではこのスコアを0と定めます。指数部で零除算せず、確定した関数値に学習価値を与えない連続極限です。
分散が正なら、前節のカイ二乗係数の下で
$\alpha_{BV}(x)=v(x)P(\alpha_{RStraddle}(x)>0)$ とも読めます。
ただし、これは無作為化Straddleスコア自体の期待値ではなく、二つが同じ選択基準になるわけでもありません。

## 10.15 分散と境界近接度の解釈

指数項

$$
\exp\left[
-\frac12
\left(
\frac{\mu-t}{\sigma}
\right)^2
\right]
$$

は standardized distance

$$
z=
\frac{\mu-t}{\sigma}
$$

が 0 に近いほど大きくなります。

したがって 分散と境界近接度の積による基準 は、

```text
posterior variance が大きい
    ×
target が posterior distribution の中心付近にある
```

候補を重視します。

Straddle と似た目的を持ちますが、同じ acquisition の再パラメータ化ではありません。

例えば平均が閾値と等しい二候補で、標準偏差が0.1と1なら、超過確率はどちらも0.5です。
一方、分散と近接度の積は0.01と1になり、潜在値の尺度で残る不確実性を区別します。
このスコアは上下分類のBernoulli分散 $p(1-p)$ と同じ量ではありません。

## 10.16 分散と近接度の積にbetaがない理由

分散と境界近接度の積による基準 の定義には Straddle のような $`\beta`$ exploration parameter はありません。

uncertainty の scale と target proximity は variance と standardized Gaussian-shaped weight
の中に組み込まれています。

## 10.17 Literature methodとlibrary-specific heuristicを分ける

ここで定義した分散と境界近接度の積は、Straddleと同じ手法でも、無作為化Straddleの期待値でもありません。
補助的なヒューリスティックを文献の手法と混同せず、目的、数式、適用範囲を定めて評価します。

## 10.18 PosteriorVarianceとの違い

事後分散基準 は

$$
\alpha(x)=\sigma_n^2(x)
$$

なので、target $`t`$ を考慮しません。

Level-set acquisition は target proximity を利用するため、関数全体の uncertainty reduction より boundary
learning に集中できます。

```text
事後分散基準
    function learning

Straddle / 分散と境界近接度の積による基準
    boundary learning
```

という目的の違いがあります。

## 10.19 Constrained BOとの関係

未知制約

$$
c(x)\le0
$$

の boundary

$$
c(x)=0
$$

を学ぶことは level-set estimation と数学的に近い問題です。

しかし最終目的は異なります。

```text
Level-set estimation
    boundary / excursion set の推定が目的

Constrained BO
    feasible region 内の objective optimization が目的
```

Constrained BO では boundary uncertainty が重要でも、それだけで objective utility は決まりません。

## 10.20 Classification Active Learningとの関係

binary classification の decision boundary learning も直感的には level-set estimation と似ています。

latent function $`g(x)`$ に対して

$$
g(x)=0
$$

を class boundary とみなせるモデルもあります。

ただし classification では likelihood が non-Gaussian であり、class probability、latent posterior、label
entropy など複数の uncertainty notion があります。

Gaussian regression の Straddle を classification acquisition とそのまま同一視しないことが重要です。

## 10.21 Batch Level-set Learning

q > 1 で複数点を同時選択する場合、pointwise Straddle score の上位 q 点では候補が同じ boundary region に集中する可能性があります。

真の batch level-set acquisition を設計するには、

- candidate correlation
- boundary information redundancy
- joint uncertainty reduction

などを定義する必要があります。

## 10.22 Structured outputとensemble posterior

scalar Gaussian posterior を前提にした boundary score を、structured output や ensemble posterior へ拡張するには
reduction semantics が必要です。

例えば ensemble members の variance を、

- within-member posterior uncertainty
- between-member disagreement

のどちらとして扱うかで意味が変わります。

そのため generic tensor reduction を追加するだけでは、level-set acquisition の意味を保証できません。

理論上の拡張にも、境界を定義する量と不確実性の意味を明示する必要があります。

## 10.23 Stopping criterion

Level-set estimation では、boundary classification uncertainty を使った stopping rule が考えられます。

例えば、

- target をまたぐ credible interval を持つ点がなくなる
- excursion-set classification probability が十分高くなる
- boundary acquisition の最大値が閾値以下になる
- boundary location uncertainty が要求精度以下になる

などです。

ただし、有限な点集合で閾値をまたぐ区間がなくても、未評価の連続領域全体が分類できたとは限りません。
スコアの絶対値は係数と出力単位にも依存するため、停止条件には対象領域と要求精度を対応させます。

## 10.25 この章で覚えておくこと

境界推定は指定した応答閾値の上下や境界を学ぶ問題であり、関数全体の学習や目的値の最大化とは異なります。
Straddleは閾値をまたぐ区間を重視し、無作為化版ではラウンド内に一つの係数を固定します。
分散と境界近接度の積は別の基準で、分類確率の分散やStraddleの期待値とは同じではありません。

複数出力や候補集合へ拡張するには、境界を定義する量と同時効用が必要です。
制約付きBOや分類の能動学習と境界が共通しても、最終目的と観測モデルは別に確認します。

## 10.26 まとめ

Level-set acquisition は、

$$
f(x)=t
$$

という boundary を効率よく学ぶための観測価値を定義します。

Straddleの区間による基準、ラウンドごとの無作為化、分散と境界近接度の積は、似た目的を異なる式で評価します。
標準偏差と確率の不確実性を区別し、係数の規約と適用条件を確認します。

また、Level-set estimation、Constrained BO、Classification AL は boundary という共通要素を持ちますが、最終的な意思決定目的は異なります。

## References

- Bryan, B. et al. (2006), *Active Learning for Identifying Function Threshold Boundaries*, for
  the Straddle level-set heuristic.
- Inatsu, Y., Takeno, S., Kutsukake, K., and Takeuchi, I. (2024), *Active Learning for Level Set
Estimation Using Randomized Straddle Algorithms*. Transactions on Machine Learning Research.
同論文は二自由度のカイ二乗分布から係数の二乗を引き、その平方根を信頼係数として使う。
