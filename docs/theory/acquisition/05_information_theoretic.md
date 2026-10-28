# 5. Information-theoretic Acquisition

[基礎章](01_foundations.md)の期待値と[バッチ章](04_batch_noisy.md)の同時分布を前提とします。
本章では、情報対象を $Z$、候補で得る実際の観測を $Y_x$ とし、対数の底は自然対数、情報量の単位はnatとします。

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

とも書けます。いずれも期待値は現在のデータを与えた予測分布について取ります。
連続変数の共同密度を $p(z,y)$ と略記すると、相互情報量は

$$
I(Z;Y\mid\mathcal D_n)=
\iint p(z,y)\log\frac{p(z,y)}{p(z)p(y)}\,dz\,dy
=\operatorname{KL}(p(Z,Y)\Vert p(Z)p(Y))\ge0
$$

です。離散変数では対応する積分を和に置き換えます。
密度の積を分母に入れて対数を展開すると、前の二つのエントロピー差を得ます。
連続変数の微分エントロピー自体は負にもなり、座標の尺度にも依存しますが、
相互情報量は可逆な座標変換で変わりません。各項が有限など、差が定義できる条件を確認します。
また、期待する情報量が非負でも、一回の観測で事後エントロピーが必ず減るとは限りません。

## 5.5 optimizer locationとoptimum valueを分ける

最大化問題では、少なくとも次の二つは別のrandom quantityです。

~~~text
optimizer location x*: どこが最大か
optimum value f*:       最大値はいくつか
~~~

位置を知りたいことと値を知りたいことは同じではありません。この違いがPESとMESを
理解する入口になります。

最適値・最適点を確率変数として扱うには、最大値が達成されるなどの条件も必要です。
例えばコンパクトな領域上で連続な関数を考え、最大点が複数なら一貫した選択規則で $x^*$ を定めます。
その上で、値を知ることと、どの位置が最大かを知ることを別の推論対象として扱います。

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
後者の近似方法を変えても、情報対象が最適値であることは変わりません。

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

将来の観測 $`Y_x`$ と潜在関数値 $`f(x)`$ は、観測 noise がある場合には異なります。
情報獲得量は実際の観測 $`Y_x`$ と最適値 $`f^*`$ などの未知量との相互情報量として定義します。
潜在値を直接観測できると仮定した情報獲得量とは一般に一致しません。

### 正規分布で情報量と相関を読む例

情報対象 $Z$ と観測 $Y$ が二変量正規分布で、相関を $\rho$、$|\rho|<1$ とすると、
正規分布のエントロピー $\tfrac12\log(2\pi e v)$ と条件付き分散 $v_Z(1-\rho^2)$ から

$$
I(Z;Y)=-\frac12\log(1-\rho^2)
$$

を得ます。$\rho=0.8$ なら約0.511 natで、候補観測の分散が大きいだけではこの値は決まりません。
独立な測定ノイズは観測分散を増やし、潜在対象との相関を弱めるため、情報量を小さくし得ます。
ただしMESの最適値は一般に正規変数ではなく、この二変量公式をそのままMESに代入することはできません。

## 5.12 Batch information gain

候補集合 $`X`$ に対しては

$$
I(Y_X;Z\mid\mathcal D_n)
$$

を考えます。

候補間に強いdependenceがある場合、各候補のpointwise information gainの和はjoint information
gainを過大評価し得ます。Gaussian formulationではcorrelation / covariance structureがこの
redundancyを表します。

連鎖律では $I(Z;Y_1,Y_2)=I(Z;Y_1)+I(Z;Y_2\mid Y_1)$ となります。
既に一つを知っている条件での追加情報を評価する点が重要です。
なお、すべての情報問題で相互情報量が単純な重複削減になるわけではありません。
例えば独立な二つの公平なビットの排他的論理和を $Z$ とすると、各ビット単独の情報量は零でも、
両方を知れば $Z$ が決まるため、共同情報量は $\log2$ です。

このため情報理論的な獲得関数でも
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

## 5.16 この章で覚えておくこと

情報理論的な獲得関数では、最初に何について情報を得たいかを定めます。
PESの最適位置、MESの最適値、予測学習の目標応答は異なる対象です。
候補のエントロピーが高いことと、その対象との相互情報量が大きいことは同じではありません。

観測ノイズと候補間の依存も情報量を変えます。GIBBONの下界やMESの最適値標本の近似を、
元の情報基準そのものと区別します。KGは情報量ではなく、観測後の判断改善を価値にする方法です。

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
- Hernández-Lobato, J. M., Hoffman, M. W., and Ghahramani, Z. (2014), *Predictive Entropy
  Search for Efficient Global Optimization of Black-box Functions*. NeurIPS.
