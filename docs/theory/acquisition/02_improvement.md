# 2. Improvement-based Acquisition

[基礎章](01_foundations.md)の期待効用を前提とします。
本章では、最大化に対する固定した比較基準を $b=f_{\mathrm{best}}$、予測標準偏差を $\sigma$ とします。
基準が不確実な場合は[ノイズ付き改善量](04_batch_noisy.md)で別に扱います。

## 2.1 PIとEIを一言で区別する

~~~text
PI: 基準を超える確率
EI: 基準をどれだけ超えるかという改善量の期待値
~~~

どちらも同じposteriorを利用できますが、候補を価値付けするutilityが異なります。
LogPI / LogEIは別の探索原理ではなく、数値計算を安定化するformulationです。

## 2.2 Improvement を直接評価する

Improvement-based acquisition は、「現在の基準よりどれだけ良い結果を得られるか」を候補の価値とします。

以下では最大化問題を考え、基準値を $`f_{\mathrm{best}}`$ とします。候補 $`x`$ の improvement は

$$
I(x)=\max(f(x)-f_{\mathrm{best}},0)
$$

です。

noiselessで関数値を直接観測する標準設定では、best valueを観測済み関数値の最良値として
扱えます。ただし、観測ノイズがある場合などに、その最大値が潜在関数の比較基準になるとは限りません。
noise、objective transform、constraint等によって改善基準の意味は変わります。

## 2.3 Expected Improvement

Expected Improvement（EI）は improvement の posterior expectation です。

$$
\operatorname{EI}(x)
=
\mathbb E[I(x)\mid\mathcal D_n]
$$

Gaussian posterior

$$
f(x)\mid\mathcal D_n
\sim
\mathcal N(\mu(x),\sigma^2(x))
$$

に対して、$\sigma(x)>0$ のとき

$$
\operatorname{EI}(x)
=
(\mu(x)-f_{\mathrm{best}})\Phi(z)
+\sigma(x)\phi(z),
\qquad
z=\frac{\mu(x)-f_{\mathrm{best}}}{\sigma(x)}
$$

となります。ここで $\Phi$ と $\phi$ は標準正規分布の累積分布関数と
確率密度関数です。$\sigma(x)=0$ では極限として
$\operatorname{EI}(x)=\max(\mu(x)-f_{\mathrm{best}},0)$ を用います。

標準正規変数 $Z$ を使って $f(x)=\mu+\sigma Z$ と書くと、改善が正なのは $Z>-z$ の範囲です。
したがって

$$
\operatorname{EI}(x)=\int_{-z}^{\infty}(\mu-b+\sigma t)\phi(t)\,dt
=(\mu-b)\Phi(z)+\sigma\phi(z).
$$

第一の積分では正規分布の対称性、第二の項では $\phi'(t)=-t\phi(t)$ を使っています。
この導出では潜在値が正規分布で、比較基準が固定されていることが重要です。

## 2.4 EI の二つの寄与

式の

$$
(\mu-f_{\mathrm{best}})\Phi(z)
$$

は予測平均による改善の寄与を、

$$
\sigma\phi(z)
$$

は posterior uncertainty により改善が起こり得る寄与を含みます。

このため EI は exploitation と exploration
を同じ改善量の期待値の中で扱います。ただし $\mu<b$ なら第一項は負であり、
独立した二つの非負スコアを足しているわけではありません。第二項と合わせて全体が非負になります。

## 2.5 EIを直感的に読む

posterior meanが基準値より高ければ、大きなimprovementが得られる可能性があります。
uncertaintyが大きい候補は、平均が基準以下でも上側tailが基準を超える可能性があります。

EIは両方をimprovementという一つの確率変数の期待値として評価します。
平均項と分散項を独立した二つのscoreだとは考えません。

## 2.6 Probability of Improvement

Probability of Improvement（PI）は改善量ではなく改善イベントの確率を評価します。

$$
\operatorname{PI}(x)
=
P(f(x)>f_{\mathrm{best}}\mid\mathcal D_n)
=
\Phi(z)
$$

PI は「改善するか」を直接評価しますが、大きな改善と小さな改善を区別しません。

例えば、わずかな改善を高確率で達成する候補と、大幅改善を中程度の確率で達成する候補では、PI は前者を好みやすく、EI は改善量も考慮します。

上のPIの閉形式は $\sigma>0$ の場合です。$\sigma=0$ では
$\operatorname{PI}=\mathbf1[\mu>b]$ であり、厳密な改善を使うなら $\mu=b$ では0です。
分散が正で平均が基準と等しい場合のPIは0.5なので、この境界で極限と確定値を混同しません。

例えば $b=0$ とし、候補Aが $\mathcal N(0.1,0.01^2)$、Bが $\mathcal N(0,1)$ なら、
AのPIはほぼ1、EIはほぼ0.1です。BのPIは0.5ですが、EIは $1/\sqrt{2\pi}\simeq0.399$ になります。
確実な小改善と、不確実だが大きな改善を両基準は異なる仕方で評価します。

固定した $\sigma>0$ でEIを微分すると $\partial\operatorname{EI}/\partial\mu=\Phi(z)$、
固定した平均での微分は $\partial\operatorname{EI}/\partial\sigma=\phi(z)>0$ です。
対照的にPIの標準偏差方向の微分は $-z\phi(z)/\sigma$ であり、平均が基準を上回る場合には負になります。
したがって「分散を増やすとどの改善基準も大きくなる」とは言えません。

## 2.7 LogEI / LogPI

EI や PI は理論的には単純ですが、改善が極めて起こりにくい領域では acquisition value と gradient が数値的に非常に小さくなることがあります。

LogEI系は、対応するimprovement criterionをlog-domainで数値的に安定して扱うformulationです。
正のEIに厳密な対数を取れば単調性から最大点は変わりません。
ただし $\log\mathbb E[I]$ と $\mathbb E[\log I]$ は別です。
正の分散を持つ正規予測では改善が零となる確率が正なので、後者は $-\infty$ になります。
また、極小のEIを計算してから対数を取るだけでは、途中のアンダーフローを避けられません。
バッチ版などで正部分や最大値を滑らかに近似する場合は、その有限の近似誤差も厳密な対数変換とは区別します。

実装上は、理論式の正しさだけでなく acquisition optimization に十分な gradient information が残ることが重要です。

## 2.8 Exploration parameter を追加する場合

EI / PI の定義には、基準値に margin $`\xi`$ を加えて

$$
I_\xi(x)
=
\max(f(x)-f_{\mathrm{best}}-\xi,0)
$$

とする流儀もあります。

$`\xi>0`$ は「単に現在値を超える」より大きな改善を要求します。この拡張では、改善イベントと改善量の双方の基準が $f_{\mathrm{best}}+\xi$ に変わります。

## 2.9 Minimize 問題

最小化問題では improvement を

$$
I(x)=\max(f_{\mathrm{best}}-f(x),0)
$$

と定義できます。

目的関数の符号を反転して最大化へ統一することもできます。数式を読む際には、最大化・最小化の convention を必ず確認する必要があります。

## 2.10 qEI と batch

q > 1 では、単純に EI の上位 q 点を選ぶのではなく、候補集合による共同 improvement を評価します。

最大化問題の概念的な q-improvement は

$$
I(X)
=
\max\left(
\max_{j=1,\ldots,q}f(x_j)-f_{\mathrm{best}},
0
\right)
$$

であり、

$$
q\operatorname{EI}(X)=\mathbb E[I(X)]
$$

を MC で評価できます。

候補間posterior dependenceがjoint utilityに影響するため、pointwise EI rankingとは異なります。
qEIをEI上位q点とは理解しません。詳細は [Batch / Noisy](04_batch_noisy.md) を参照してください。

## 2.11 Noise と NEI

observation noise があると、最大観測値が潜在関数の真の最良値とは限りません。この場合、固定した observed best を基準にする単純 EI は不安定な意思決定になり得ます。

Noisy Expected Improvement（NEI）はbaseline pointsのlatent function valuesについても
posterior uncertaintyを積分し、noise-awareなimprovementを評価します。

これは単にEI式のposterior standard deviationを大きくすることではありません。
best baseline自体が不確実という問題を扱います。

このため「noise があるので $`f_{\mathrm{best}}=\max y_i`$ とする」という単純化は避ける必要があります。NEI の詳細は [Batch /
Noisy](04_batch_noisy.md) で扱います。

## 2.13 使い分けの理解

EI と PI の本質的な違いは、

```text
PI
    改善イベントの確率

EI
    改善量の期待値
```

です。

Log formulation は別の意思決定原理ではなく、数値安定性を改善する実装上重要な formulation です。batch と noise はさらに joint posterior
と latent baseline uncertainty を導入するため、q / noisy variants を独立に理解する必要があります。

## 2.14 この章で覚えておくこと

PIは改善する確率、EIは改善量の期待値を評価します。同じ予測分布でも候補の順位が変わり得ます。
比較基準を固定する仮定と、分散零での定義を確認し、EIの二項を独立したスコアとみなしません。

厳密な対数変換は正の値の順位を保ちますが、期待値と対数の順序や有限の平滑化は区別します。
バッチでは候補集合の共同改善を、ノイズ下では基準の潜在値の不確実性も積分します。

## References

- Ament, S. et al. (2023), *Unexpected Improvements to Expected Improvement for Bayesian
  Optimization*. Advances in Neural Information Processing Systems 36.
- Jones, D. R., Schonlau, M., and Welch, W. J. (1998), *Efficient Global Optimization of
  Expensive Black-Box Functions*. Journal of Global Optimization, 13, 455–492.
