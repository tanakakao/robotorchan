# 5. Information-theoretic Acquisition

## 5.1 EI / UCBとの違いから理解する

~~~text
EI / UCB
    次のcandidateで高い目的値を得られそうか

information-theoretic acquisition
    最適化問題についてどれだけ重要な情報を得られそうか
~~~

後者でも最終目的はoptimizationですが、candidate valueを直接objective valueだけで測りません。
「何についての情報か」を決め、その未知量のuncertainty reductionを価値にします。

## 5.2 情報量を観測価値として使う

Information-theoretic acquisition は、候補点の目的値そのものではなく、**特定の未知量について観測がどれだけ情報を与えるか**を候補価値として扱います。

重要なのは「情報量」という語だけでは不十分で、何についての情報かを明示することです。

例えば Bayesian Optimization では、

- optimizer location $`x^*`$
- optimum value $`f^*`$
- Pareto set / Pareto front

などが情報獲得対象になり得ます。

Active Learning の predictive information gain は別の対象を持つため、同じ mutual information を使っていても目的は異なります。

## 5.3 Entropy

離散確率変数 $`Z`$ の entropy は

$$
H(Z)
=
-\sum_z p(z)\log p(z)
$$

です。

連続変数では differential entropy

$$
H(Z)
=
-\int p(z)\log p(z)\,dz
$$

を考えます。

Entropyはuncertaintyの尺度ですが、acquisitionで重要なのは現在のentropyだけではありません。
**候補の観測でinformation targetのuncertaintyがどれだけ減るか**を評価します。

## 5.4 Mutual Informationはexpected entropy reduction

未知量 $`Z`$ と候補 $`x`$ で得られる将来観測 $`Y_x`$ の mutual information は

$$
I(Z;Y_x\mid\mathcal D_n)
=
H(Z\mid\mathcal D_n)
-
\mathbb E_{Y_x}
[
H(Z\mid\mathcal D_n,Y_x)
]
$$

と書けます。

これは次のexpected uncertainty reductionです。

~~~text
観測前のuncertainty - 観測後に残るexpected uncertainty
~~~

対称性から

$$
I(Z;Y_x\mid\mathcal D_n)
=
H(Y_x\mid\mathcal D_n)
-
\mathbb E_Z[
H(Y_x\mid\mathcal D_n,Z)
]
$$

とも書けます。実際のアルゴリズムでは、どちらの表現が計算しやすいかが重要になります。

## 5.5 optimizer locationとoptimum valueを分ける

最大化問題では、少なくとも次の二つは別のrandom quantityです。

~~~text
optimizer location x*: どこが最大か
optimum value f*:       最大値はいくつか
~~~

位置を知りたいことと値を知りたいことは同じではありません。この違いがPESとMESを
理解する入口になります。

## 5.6 Entropy Search の考え方

Entropy Search 系は、最適化問題を「未知の optimum について効率よく学習する問題」として捉えます。

optimizer location を

$$
x^*
=
\arg\max_{x\in\mathcal X} f(x)
$$

とすれば、Predictive Entropy Search（PES）などは $`x^*`$ に関する情報獲得を考えます。

この考え方は直接improvementを評価するEIと異なり、現在のobjective valueが高くなくても
optimum identificationに有益な候補を評価し得ます。

## 5.7 Max-value Entropy Search

Max-value Entropy Search（MES）は optimizer location $`x^*`$ ではなく、最大値

$$
f^*
=
\max_{x\in\mathcal X} f(x)
$$

について得られる情報を最大化します。

概念的には

$$
\alpha_{\mathrm{MES}}(x)
=
I(Y_x;f^*\mid\mathcal D_n)
$$

です。

optimizer の位置分布より scalar な optimum value の分布を扱うことで、情報理論的 BO をより計算しやすくすることが MES の重要な発想です。

## 5.8 MES の計算構造

MES では一般に、

1. posterior から optimum value $`f^*`$ の samples を得る。
2. 各 sampled optimum value の条件下で候補観測の entropy reduction を評価する。
3. samples について平均する。

という構造を持ちます。

実際の $`f^*`$ sampling には近似が必要です。探索空間の離散化、posterior function sampling、極値分布近似など、実装により計算方法は異なります。

したがって「MESのdecision criterion」と「optimum-value samplesの生成方法」は区別します。
後者の近似方法を変えることと、情報対象を # 5. Information-theoretic Acquisition

## 5.9 GIBBON

GIBBONはbatch Bayesian Optimizationを意識したinformation-theoretic acquisitionで、
Gaussian mutual informationとMES型のoptimum-value informationを組み合わせたtractableな
lower-bound formulationを利用します。

batch settingでは候補間のredundancyを無視すると似た候補を複数選びやすくなります。
GIBBONは候補集合のdependence structureを利用し、情報の重複を考慮します。

したがって GIBBON を「MES を q 点へ単純拡張しただけ」と理解するのは不十分です。

## 5.10 Information gain と exploration

Information-theoretic acquisition は exploration-oriented に見えますが、単なる posterior variance 最大化とは異なります。

posterior variance は

```text
その点の関数値がどれだけ不確実か
```

を評価します。

MES は

```text
その観測が optimum value についてどれだけ教えるか
```

を評価します。

不確実でも optimum と無関係な領域は、MES では価値が低くなり得ます。

## 5.11 Observation noise

noise がある場合、将来観測 $`Y_x`$ と latent function value # 5. Information-theoretic Acquisition

## 5.1 EI / UCBとの違いから理解する

~~~text
EI / UCB
    次のcandidateで高い目的値を得られそうか

information-theoretic acquisition
    最適化問題についてどれだけ重要な情報を得られそうか
~~~

後者でも最終目的はoptimizationですが、candidate valueを直接objective valueだけで測りません。
「何についての情報か」を決め、その未知量のuncertainty reductionを価値にします。

## 5.2 情報量を観測価値として使う

Information-theoretic acquisition は、候補点の目的値そのものではなく、**特定の未知量について観測がどれだけ情報を与えるか**を候補価値として扱います。

重要なのは「情報量」という語だけでは不十分で、何についての情報かを明示することです。

例えば Bayesian Optimization では、

- optimizer location $`x^*`$
- optimum value $`f^*`$
- Pareto set / Pareto front

などが情報獲得対象になり得ます。

Active Learning の predictive information gain は別の対象を持つため、同じ mutual information を使っていても目的は異なります。

## 5.3 Entropy

離散確率変数 $`Z`$ の entropy は

$$
H(Z)
=
-\sum_z p(z)\log p(z)
$$

です。

連続変数では differential entropy

$$
H(Z)
=
-\int p(z)\log p(z)\,dz
$$

を考えます。

Entropyはuncertaintyの尺度ですが、acquisitionで重要なのは現在のentropyだけではありません。
**候補の観測でinformation targetのuncertaintyがどれだけ減るか**を評価します。

## 5.4 Mutual Informationはexpected entropy reduction

未知量 $`Z`$ と候補 $`x`$ で得られる将来観測 $`Y_x`$ の mutual information は

$$
I(Z;Y_x\mid\mathcal D_n)
=
H(Z\mid\mathcal D_n)
-
\mathbb E_{Y_x}
[
H(Z\mid\mathcal D_n,Y_x)
]
$$

と書けます。

これは次のexpected uncertainty reductionです。

~~~text
観測前のuncertainty - 観測後に残るexpected uncertainty
~~~

対称性から

$$
I(Z;Y_x\mid\mathcal D_n)
=
H(Y_x\mid\mathcal D_n)
-
\mathbb E_Z[
H(Y_x\mid\mathcal D_n,Z)
]
$$

とも書けます。実際のアルゴリズムでは、どちらの表現が計算しやすいかが重要になります。

## 5.5 optimizer locationとoptimum valueを分ける

最大化問題では、少なくとも次の二つは別のrandom quantityです。

~~~text
optimizer location x*: どこが最大か
optimum value f*:       最大値はいくつか
~~~

位置を知りたいことと値を知りたいことは同じではありません。この違いがPESとMESを
理解する入口になります。

## 5.6 Entropy Search の考え方

Entropy Search 系は、最適化問題を「未知の optimum について効率よく学習する問題」として捉えます。

optimizer location を

$$
x^*
=
\arg\max_{x\in\mathcal X} f(x)
$$

とすれば、Predictive Entropy Search（PES）などは $`x^*`$ に関する情報獲得を考えます。

この考え方は直接improvementを評価するEIと異なり、現在のobjective valueが高くなくても
optimum identificationに有益な候補を評価し得ます。

## 5.7 Max-value Entropy Search

Max-value Entropy Search（MES）は optimizer location $`x^*`$ ではなく、最大値

$$
f^*
=
\max_{x\in\mathcal X} f(x)
$$

について得られる情報を最大化します。

概念的には

$$
\alpha_{\mathrm{MES}}(x)
=
I(Y_x;f^*\mid\mathcal D_n)
$$

です。

optimizer の位置分布より scalar な optimum value の分布を扱うことで、情報理論的 BO をより計算しやすくすることが MES の重要な発想です。

## 5.8 MES の計算構造

MES では一般に、

1. posterior から optimum value $`f^*`$ の samples を得る。
2. 各 sampled optimum value の条件下で候補観測の entropy reduction を評価する。
3. samples について平均する。

という構造を持ちます。

実際の $`f^*`$ sampling には近似が必要です。探索空間の離散化、posterior function sampling、極値分布近似など、実装により計算方法は異なります。

したがって「MESのdecision criterion」と「optimum-value samplesの生成方法」は区別します。
後者の近似方法を変えることと、情報対象を # 5. Information-theoretic Acquisition

## 5.9 GIBBON

GIBBONはbatch Bayesian Optimizationを意識したinformation-theoretic acquisitionで、
Gaussian mutual informationとMES型のoptimum-value informationを組み合わせたtractableな
lower-bound formulationを利用します。

batch settingでは候補間のredundancyを無視すると似た候補を複数選びやすくなります。
GIBBONは候補集合のdependence structureを利用し、情報の重複を考慮します。

したがって GIBBON を「MES を q 点へ単純拡張しただけ」と理解するのは不十分です。

## 5.10 Information gain と exploration

Information-theoretic acquisition は exploration-oriented に見えますが、単なる posterior variance 最大化とは異なります。

posterior variance は

```text
その点の関数値がどれだけ不確実か
```

を評価します。

MES は

```text
その観測が optimum value についてどれだけ教えるか
```

を評価します。

不確実でも optimum と無関係な領域は、MES では価値が低くなり得ます。

## 5.11 Observation noise

noise がある場合、将来観測 $`Y_x`$ と latent function value f(x)`$ は区別されます。

$$
Y_x=f(x)+\epsilon
$$

information gainは、noiseを含む観測からlatent optimumについて何を学べるかを評価します。
predictive entropyが高くても、その多くがirreducible observation noiseならoptimumに関する
情報価値は高いとは限りません。

## 5.12 Batch information gain

候補集合 $`X`$ に対しては

$$
I(Y_X;Z\mid\mathcal D_n)
$$

を考えます。

候補間に強いdependenceがある場合、各候補のpointwise information gainの和はjoint information
gainを過大評価し得ます。Gaussian formulationではcorrelation / covariance structureがこの
redundancyを表します。

このためinformation-theoretic acquisitionでも
[Batch / Noisy](04_batch_noisy.md) で扱ったjoint posteriorの考え方が重要です。

## 5.13 MES と Knowledge Gradient の違い

MES と Knowledge Gradient（KG）はどちらも myopic improvement より先を意識しますが、最適化対象は異なります。

```text
MES
    optimum value について得られる information

KG
    観測後の最終 decision value の expected improvement
```

MESはentropy / mutual information、KGはvalue of informationの考え方です。
KGは次章 [Lookahead](06_lookahead.md) で扱います。

## 5.14 Active Learning の information gain との違い

Active Learning でも mutual information を利用できますが、情報対象が異なります。

例えば predictive information gain では、target distribution 上の未知 prediction について候補観測が与える情報を評価します。

したがって、

```text
MES
    optimum value が情報対象

EPIG
    target prediction が情報対象
```

です。

EPIG は [Active Learning](09_active_learning.md) で扱います。

## 5.15 BoTorch / robotorchan との対応

BoTorchはMESとGIBBONを含むinformation-theoretic acquisitionを提供しています。
robotorchanはBoTorch-nativeで利用可能な標準手法を再実装しません。

現在のintegration guideはMES / GIBBONをnative pathとして扱います。一方、robotorchanの
capability registryはBoTorch acquisition全件のcatalogではなく、MES / GIBBONのentryがないことを
「利用不可」とは解釈しません。

またinformation-theoretic acquisitionのposterior requirementは手法ごとに確認します。
「information-theoreticだからPOSTERIOR_SAMPLES」と一括して決めず、optimum-value sampling、
conditional entropy、joint Gaussian structure等、実際の計算経路が要求するcontractを見ます。

利用上の位置付けは
[Information-theoretic optimization guide](../../optimization/information_theoretic.md)、
現在の統合状況は
[Acquisition integration status](../../optimization/acquisition-integration.md) を参照してください。

## 5.16 この章で覚えておくこと

- information-theoretic acquisitionでは最初にinformation targetを確認する
- entropyが大きい点を選ぶこととmutual informationを最大化することは同じではない
- optimizer locationとoptimum valueは別のunknown quantity
- PESはoptimizer location、MESはoptimum valueを代表的なinformation targetとする
- MES criterionとoptimum-value sampling approximationを分けて考える
- GIBBONはMESを単純にq点へ並べたものではなくbatch redundancyを考慮する
- observation noise由来のentropyが高くてもoptimum informationが大きいとは限らない
- MESとKGはinformation gainとdecision valueという異なる価値基準を持つ
- registryにentryがないことをBoTorch-native acquisitionの利用不可とは解釈しない

## 5.17 まとめ

Information-theoretic acquisition を理解するときは、必ず

$$
\text{information about what?}
$$

を確認します。

MES は optimum value、PES は optimizer location、predictive AL は prediction についての情報を扱います。

「uncertainty が大きい点を選ぶ」ことと「意思決定対象について情報を最大化する」ことは同じではありません。


## References

- Wang, Z. and Jegelka, S. (2017), *Max-value Entropy Search for Efficient Bayesian
  Optimization*. ICML.
- Moss, H. B. et al. (2021), *GIBBON: General-purpose Information-Based Bayesian
  Optimisation*. JMLR.
- BoTorch documentation, *Acquisition Functions*, for current native MES and lower-bound
  max-value entropy interfaces.
