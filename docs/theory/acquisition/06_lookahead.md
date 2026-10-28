# 6. Lookahead Acquisition

[情報量章](05_information_theoretic.md)の情報獲得と、[GP章](../02_gaussian_process.md)の観測による条件付けを前提とします。
本章では観測する点と、観測後に最終的に採用する点を分け、適応的な意思決定を考えます。

## 6.1 EI / MES / KGの違いから理解する

~~~text
EI   : candidate自身のimprovement
MES  : optimumについて得られるinformation
KG   : 観測後のterminal decision valueの改善
~~~

KGの中心は「何を知れるか」だけではなく、その情報によって**最後に下すdecisionがどれだけ
良くなるか**です。これがvalue of informationです。

## 6.2 現在の候補ではなく観測後の意思決定を見る

Lookahead acquisition は、候補 $`x`$ の現在の objective value だけでなく、

**その候補を観測した後、将来の意思決定がどれだけ良くなるか**

を評価します。

現在のデータを $`\mathcal D_n`$、候補 $`x`$ の未知観測を $`Y_x`$ とすると、観測後データは

$$
\mathcal D_{n+1}
=
\mathcal D_n\cup\{(x,Y_x)\}
$$

です。

しかし acquisition を評価する時点では $`Y_x`$ は未知です。そのため、possible outcomes に対して観測後の posterior と意思決定を平均する必要があります。

## 6.3 Value of Information

現在の最終意思決定価値を $`V(\mathcal D_n)`$ とします。

候補 $`x`$ を観測した後の expected decision value が

$$
\mathbb E_{Y_x}
[
V(\mathcal D_n\cup\{(x,Y_x)\})
]
$$

なら、one-step value of information は概念的に

$$
\alpha_{\mathrm{VOI}}(x)
=
\mathbb E_{Y_x}
[
V(\mathcal D_n\cup\{(x,Y_x)\})
]
-
V(\mathcal D_n)
$$

と書けます。

これは「候補そのものがどれだけ良いか」ではなく、「候補を知ることによって最終判断がどれだけ改善するか」です。

## 6.4 Knowledge Gradient

Knowledge Gradient（KG）は Bayesian Optimization における代表的な value-of-information criterion です。

観測ノイズがあっても、終端で潜在関数の期待値を最大にする点を推薦するなら、現在の最良判断の価値は

$$
V_n
=
\max_{x'\in\mathcal X}\mu_n(x')
$$

と、候補 $`x`$ を観測した後の posterior mean $`\mu_{n+1}`$ に基づく

$$
V_{n+1}
=
\max_{x'\in\mathcal X}\mu_{n+1}(x')
$$

の expected difference を評価します。

$$
\operatorname{KG}(x)
=
\mathbb E_{Y_x}[V_{n+1}]
-
V_n
$$

実際の formulation は observation noise、terminal value、candidate representation などに応じて変わりますが、本質は
**観測後の最終 decision quality の改善**です。

同じ終端の選択集合を用い、整合的なベイズ更新と有限な期待値を仮定すると、
条件付き期待値の反復則により

$$
\mathbb E_n[\max_{x'}\mu_{n+1}(x')]
\ge\max_{x'}\mathbb E_n[\mu_{n+1}(x')]
=\max_{x'}\mu_n(x').
$$

したがって理想的なKGは非負です。情報を得た後も元の判断を選べるため、期待する最適判断は悪化しません。
ただし、一回の観測後の最大事後平均が必ず上がるという意味ではありません。
また、ハイパーパラメータの再推定や不正確な内側最適化を含む数値手順には、別の誤差が生じます。

例えば推薦候補がA・Bの二点で、Aの値は既知の0、Bは現在 $\mathcal N(-0.5,1)$ とします。
Bをノイズなく観測すれば、その後は $\max(0,F_B)$ の良い方を推薦できます。
現在の推薦価値は0なので、KGは $\phi(0.5)-0.5\Phi(-0.5)\simeq0.198$ です。
これは現在Bの平均が劣っていても、観測後に判断を変える価値がある例です。

## 6.5 EI との違い

EI は候補 $`x`$ 自身が incumbent を改善する期待量を評価します。

KG は候補 $`x`$ の観測によって posterior 全体が更新され、その結果として最終的に選ぶ decision が改善する価値を評価します。

```text
EI
    candidate の function value による improvement

KG
    candidate を観測した後の decision quality の improvement
```

したがって、候補自身の objective value が高くなくても、他領域についての判断を大きく変える情報を持つなら KG が価値を認める場合があります。

## 6.6 Fantasy model

未知の観測結果を仮想的に生成し、それを観測したと仮定した posterior を **fantasy model** と呼びます。

概念的には

```text
current posterior
    ↓
candidate x
    ↓
sample possible observation y
    ↓
condition model on (x, y)
    ↓
fantasy posterior
    ↓
optimize future decision
```

という計算を複数の possible observations について行います。

GP では条件付き Gaussian structure を利用して fantasy posterior を効率よく構成できますが、通常の myopic acquisition
より計算負荷は高くなります。

## 6.7 Inner optimization

KG には nested optimization が現れます。

外側では「どこを観測するか」を最適化し、各 fantasy outcome の内側では「観測後にどこを最終選択するか」を最適化します。

$$
\max_x
\;
\mathbb E_{Y_x}
\left[
\max_{x'} \mu_{n+1}(x')
\right]
$$

この inner maximization が KG の計算コストと最適化難易度の大きな要因です。

## 6.8 qKG

複数候補を同時に観測する場合は、candidate batch $`X`$ の joint future observations を考えます。

$$
Y_X\sim p(Y_X\mid\mathcal D_n)
$$

qKG は batch を観測した後の expected terminal decision value を評価します。

終端価値を $V(\mathcal D)$ と置けば、one-step batch KG は概念的に

$$
q\operatorname{KG}(X)
=
\mathbb E_{Y_X\mid\mathcal D_n}
[V(\mathcal D_n\cup\{(X,Y_X)\})]
-
V(\mathcal D_n)
$$

と書けます。後段の最適な意思決定は各 fantasy outcome に応じて変わり、
同じ固定された future candidate を全シナリオに使うわけではありません。
観測費用や評価時間を組み込む場合は、この終端価値との差分だけでなく
対応する cost-aware utility を定義する必要があります。

qEIと同様に、qKGもpointwise KGの上位q点を選ぶものではありません。候補集合のjoint future
observationsと、それらから構成されるfantasy modelsがterminal valueへ影響します。

## 6.9 One-stepとMulti-step Lookahead

KG は代表的な one-step lookahead criterion です。

より一般には、現在の候補の後にさらに複数段階の adaptive decisions を行う policy を評価できます。

2-step の概念構造は

```text
x1 を選ぶ
    ↓
y1 を観測
    ↓
y1 に応じて x2(y1) を選ぶ
    ↓
y2 を観測
    ↓
terminal decision
```

です。

重要なのは、2点を最初から同時に選ぶ batch BO と異なり、**後段の候補が前段の観測結果に適応する**ことです。

## 6.10 Scenario tree と fantasy branching

Multi-step lookahead では future observation ごとに decision が分岐します。

```text
current
 ├─ fantasy y1(a)
 │    ├─ future y2(a1)
 │    └─ future y2(a2)
 └─ fantasy y1(b)
      ├─ future y2(b1)
      └─ future y2(b2)
```

この branching structure を増やすほど将来 policy を精密に近似できますが、計算量も急速に増えます。

そのため fantasy sample 数、branching factor、lookahead depth は accuracy と computation の trade-off になります。

## 6.11 One-shot formulationとauxiliary decision rows

一段先の期待値をS個の仮想観測で近似すると、観測点 $x$ と各標本での終端判断 $x'_s$ を一緒に最適化できます。

$$
\max_{x,x'_1,\ldots,x'_S}\frac1S\sum_{s=1}^S\mu_{n+1}^{(s)}(x'_s)-V_n.
$$

ここで $\mu_{n+1}^{(s)}$ は候補 $x$ での第s仮想観測による事後平均です。
補助変数 $x'_s$ を最適化することで、各シナリオの内側最適化を大きな一つの問題へ表現します。
実際に次に評価するのは観測点であり、補助的な終端判断すべてを同時に実験するわけではありません。
バッチ観測なら実評価点はq個であり、仮想観測の個数Sや将来判断の個数とは別です。

## 6.12 多段階先読み

これは特定の単一 closed-form acquisition というより、stage-wise value functions と fantasy branching を組み合わせる
framework として理解する方が適切です。

各段階の価値と終端価値の定義によって意思決定問題が変わります。
また、ある段階の判断はその時点までの観測だけに依存しなければなりません。
同じ観測履歴の枝に異なる判断を許したり、未来の結果を先に知って選ばせたりすると、
実行できない先読みの価値を計算してしまいます。

## 6.13 Batch と Multi-step の違い

両者は混同しやすいですが、明確に異なります。

### Batch

$$
(x_1,\ldots,x_q)
$$

を**観測結果を見る前に同時決定**します。

### Multi-step adaptive policy

$$
x_1
\rightarrow
Y_1
\rightarrow
x_2(Y_1)
$$

のように、後の decision が前の observation に依存します。

batch size q と lookahead depth は別の軸です。

## 6.14 Information-theoretic acquisition との違い

MES などは optimum についての information gain を直接評価します。

KG は terminal decision value の expected improvement を評価します。

```text
MES
    uncertainty reduction about optimum

KG
    expected improvement in terminal decision value
```

情報を得ること自体ではなく、その情報が最終 decision を改善する価値を持つかが KG の中心です。

## 6.15 Multi-Fidelity との関係

fidelity ごとに cost と information quality が異なる場合、KG の value-of-information interpretation は特に自然です。

候補 $`(x,s)`$ の観測が target fidelity の最終 decision をどれだけ改善するかを評価し、さらに evaluation cost を考慮できます。

この考え方は [Multi-Fidelity / Cost-aware](11_multifidelity_cost_aware.md) で扱います。

## 6.16 計算上の注意

Lookahead acquisition は通常、

- fantasy sampling
- posterior conditioning
- inner optimization
- outer acquisition optimization
- multi-step なら scenario branching

を含みます。

そのため myopic acquisition より高コストです。surrogate evaluation が安価でも acquisition optimization が支配的になることがあります。

lookahead depth を増やせば常に実用上優れるわけではなく、残り evaluation budget と acquisition computation cost のバランスが必要です。

## 6.18 この章で覚えておくこと

KGは観測後の最良判断の価値を平均し、現在の判断からの改善を評価します。
観測値への適応と整合的なベイズ更新が重要であり、情報の量を測るMESとも目的が異なります。
ノイズがある場合でも、終端価値を明確にすれば同じ意思決定の枠組みを使えます。

バッチの同時決定と多段階の適応的な方策は異なります。
一括した数値最適化で使う補助変数の全てを実験するわけではなく、各判断はその時点までの観測だけに依存させます。
標本数・内側最適化・分岐数の誤差と費用を考慮して先読みの深さを選びます。

## 6.19 まとめ

Lookahead acquisition の中心は

$$
\text{value of information for future decisions}
$$

です。

KG は one-step の代表例で、multi-step lookahead は future observation に応じて後段の decision が変わる adaptive
policy を扱います。

batch candidate selection と multi-step adaptive decision は別の概念であり、fantasy model
は未知の将来観測を現在の時点で評価するための中核的な仕組みです。

## References

- Frazier, P. I., Powell, W. B., and Dayanik, S. (2008), *A Knowledge-Gradient Policy for
Sequential Information Collection*. SIAM Journal on Control and Optimization.
- Jiang, S. et al. (2020), *Efficient Nonmyopic Bayesian Optimization via One-Shot Multi-Step
Trees*. NeurIPS.
