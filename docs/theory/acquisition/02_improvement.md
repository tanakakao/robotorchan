# 2. Improvement-based Acquisition

## 2.1 PIとEIを一言で区別する

~~~text
PI: 基準を超える確率
EI: 基準をどれだけ超えるかという改善量の期待値
~~~

どちらも同じposteriorを利用できますが、候補を価値付けするutilityが異なります。
LogPI / LogEIは別の探索原理ではなく、数値計算を安定化するformulationです。

## 2.3 Improvement を直接評価する

Improvement-based acquisition は、「現在の基準よりどれだけ良い結果を得られるか」を候補の価値とします。

以下では最大化問題を考え、基準値を $`f_{\mathrm{best}}`$ とします。候補 $`x`$ の improvement は

$
I(x)=\max(f(x)-f_{\mathrm{best}},0)
$

です。

noiselessで関数値を直接観測する標準設定では、best valueを観測済み関数値の最良値として
扱えます。ただしbest valueは常に観測yの最大値というAPI規則ではありません。
noise、objective transform、constraint等によって改善基準の意味は変わります。

## 2.2 Expected Improvement

Expected Improvement（EI）は improvement の posterior expectation です。

$
\operatorname{EI}(x)
=
\mathbb E[I(x)\mid\mathcal D_n]
$

Gaussian posterior

$
f(x)\mid\mathcal D_n
\sim
\mathcal N(\mu(x),\sigma^2(x))
$

に対して、\(\sigma(x)>0\) なら

$
z(x)
=
\frac{\mu(x)-f_{\mathrm{best}}}{\sigma(x)}
$

として

$
\operatorname{EI}(x)
=
(\mu(x)-f_{\mathrm{best}})\Phi(z)
+
\sigma(x)\phi(z)
$

と書けます。

$`\Phi`$ は標準正規 CDF、$`\phi`$ は標準正規 PDF です。

## 2.4 EI の二つの寄与

式の

$
(\mu-f_{\mathrm{best}})\Phi(z)
$

は予測平均による改善の寄与を、

$
\sigma\phi(z)
$

は posterior uncertainty により改善が起こり得る寄与を含みます。

このため EI は exploitation と exploration
を自然に両立する代表的な基準です。ただし、これは二つの項を独立な「活用スコア」「探索スコア」として最適化しているという意味ではありません。両者は同じ improvement 
expectation の解析式です。

## 2.5 EIを直感的に読む

posterior meanが基準値より高ければ、大きなimprovementが得られる可能性があります。
uncertaintyが大きい候補は、平均が基準以下でも上側tailが基準を超える可能性があります。

EIは両方をimprovementという一つの確率変数の期待値として評価します。
平均項と分散項を独立した二つのscoreだとは考えません。

## 2.6 Probability of Improvement

Probability of Improvement（PI）は改善量ではなく改善イベントの確率を評価します。

$
\operatorname{PI}(x)
=
P(f(x)>f_{\mathrm{best}}\mid\mathcal D_n)
=
\Phi(z)
$

PI は「改善するか」を直接評価しますが、大きな改善と小さな改善を区別しません。

例えば、わずかな改善を高確率で達成する候補と、大幅改善を中程度の確率で達成する候補では、PI は前者を好みやすく、EI は改善量も考慮します。

## 2.7 LogEI / LogPI

EI や PI は理論的には単純ですが、改善が極めて起こりにくい領域では acquisition value と gradient が数値的に非常に小さくなることがあります。

LogEI系は、対応するimprovement criterionをlog-domainで数値的に安定して扱うformulationです。
別の探索原理として扱いません。実際のlog-EI familyはunderflowやvanishing gradientを避ける
numerically robust formulationを含むため、対応する実装定義を確認します。

実装上は、理論式の正しさだけでなく acquisition optimization に十分な gradient information が残ることが重要です。

## 2.8 Exploration parameter を追加する場合

EI / PI の定義には、基準値に margin $`\xi`$ を加えて

$
I_\xi(x)
=
\max(f(x)-f_{\mathrm{best}}-\xi,0)
$

とする流儀もあります。

$`\xi>0`$ は「単に現在値を超える」より大きな改善を要求します。ただし、ライブラリ API がこの parameterization を採用しているとは限らないため、理論上の
parameter と実装引数を混同しないことが重要です。

## 2.9 Minimize 問題

最小化問題では improvement を

$
I(x)=\max(f_{\mathrm{best}}-f(x),0)
$

と定義できます。

実装では objective の符号を反転して最大化 convention に統一することもあります。数式を読む際には、最大化・最小化の convention を必ず確認する必要があります。

## 2.10 qEI と batch

q > 1 では、単純に EI の上位 q 点を選ぶのではなく、候補集合による共同 improvement を評価します。

最大化問題の概念的な q-improvement は

$
I(X)
=
\max\left(
\max_{j=1,\ldots,q}f(x_j)-f_{\mathrm{best}},
0
\right)
$

であり、

$
q\operatorname{EI}(X)=\mathbb E[I(X)]
$

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

## 2.12 BoTorch / robotorchan との対応

BoTorch には EI / PI とその log formulation、q / noisy variants が用意されています。robotorchan
はこれらを再実装せず、native path を利用します。

実際の推奨 API と現在の統合状況は [Standard acquisition](../../optimization/standard_acquisition.md) と
[Acquisition integration status](../../optimization/acquisition-integration.md) を参照してください。

### 2.10.1 現在の registry 名と LogEI 系

現在のrobotorchan capability registryでは、代表的なBoTorch-native integration contractとして
`qLogExpectedImprovement` と `qLogNoisyExpectedImprovement` を登録しています。

integration guideではnative LogEI / LogPI等も利用対象です。registryはBoTorch acquisition
全件のcatalogではないため、registryにないことを利用不可とは解釈しません。

これは robotorchan 独自の EI 実装を持つという意味ではありません。
実装 strategy は **BoTorch-native acquisition; no robotorchan wrapper** です。

両者は Monte Carlo posterior sample を要求し、output / black-box constraint composition を
サポートする acquisition として metadata 化されています。
candidate/input-space constraint の可否は acquisition metadata ではなく optimizer 側の責務です。

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

- PIは改善eventの確率、EIは改善量の期待値
- EIの平均・不確実性の寄与は同じimprovement expectationから生じる
- LogEI / LogPIは別の探索目的ではなくnumerical robustnessのためのformulation
- best valueの意味はnoiseやobjective compositionによって変わる
- qEIはpointwise EI上位q点ではなくjoint improvement
- NEIはlatent baseline uncertaintyを扱う
- registryはBoTorch acquisition全件のcatalogではない

## References

- Ament, S. et al. (2023), *Unexpected Improvements to Expected Improvement for Bayesian
Optimization*. arXiv:2310.20708. BoTorch's log-EI family follows this numerically robust 
formulation.
- BoTorch documentation, *Acquisition Functions*, for the current analytic / Monte Carlo
acquisition interfaces and log-space EI variants.
