# 6. Lookahead Acquisition

## 6.1 現在の候補ではなく観測後の意思決定を見る

Lookahead acquisition は、候補 \(x\) の現在の objective value だけでなく、

**その候補を観測した後、将来の意思決定がどれだけ良くなるか**

を評価します。

現在のデータを \(\mathcal D_n\)、候補 \(x\) の未知観測を \(Y_x\) とすると、観測後データは

\[
\mathcal D_{n+1}
=
\mathcal D_n\cup\{(x,Y_x)\}
\]

です。

しかし acquisition を評価する時点では \(Y_x\) は未知です。そのため、possible outcomes に対して観測後の posterior と意思決定を平均する必要があります。

## 6.2 Value of Information

現在の最終意思決定価値を \(V(\mathcal D_n)\) とします。

候補 \(x\) を観測した後の expected decision value が

\[
\mathbb E_{Y_x}
[
V(\mathcal D_n\cup\{(x,Y_x)\})
]
\]

なら、one-step value of information は概念的に

\[
\alpha_{\mathrm{VOI}}(x)
=
\mathbb E_{Y_x}
[
V(\mathcal D_n\cup\{(x,Y_x)\})
]
-
V(\mathcal D_n)
\]

と書けます。

これは「候補そのものがどれだけ良いか」ではなく、「候補を知ることによって最終判断がどれだけ改善するか」です。

## 6.3 Knowledge Gradient

Knowledge Gradient（KG）は Bayesian Optimization における代表的な value-of-information criterion です。

noiseless な直感では、現在の posterior mean に基づく最良 decision value

\[
V_n
=
\max_{x'\in\mathcal X}\mu_n(x')
\]

と、候補 \(x\) を観測した後の posterior mean \(\mu_{n+1}\) に基づく

\[
V_{n+1}
=
\max_{x'\in\mathcal X}\mu_{n+1}(x')
\]

の expected difference を評価します。

\[
\operatorname{KG}(x)
=
\mathbb E_{Y_x}[V_{n+1}]
-
V_n
\]

実際の formulation は observation noise、terminal value、candidate representation などに応じて変わりますが、本質は **観測後の最終 decision quality の改善**です。

## 6.4 EI との違い

EI は候補 \(x\) 自身が incumbent を改善する期待量を評価します。

KG は候補 \(x\) の観測によって posterior 全体が更新され、その結果として最終的に選ぶ decision が改善する価値を評価します。

```text
EI
    candidate の function value による improvement

KG
    candidate を観測した後の decision quality の improvement
```

したがって、候補自身の objective value が高くなくても、他領域についての判断を大きく変える情報を持つなら KG が価値を認める場合があります。

## 6.5 Fantasy model

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

GP では条件付き Gaussian structure を利用して fantasy posterior を効率よく構成できますが、通常の myopic acquisition より計算負荷は高くなります。

## 6.6 Inner optimization

KG には nested optimization が現れます。

外側では「どこを観測するか」を最適化し、各 fantasy outcome の内側では「観測後にどこを最終選択するか」を最適化します。

\[
\max_x
\;
\mathbb E_{Y_x}
\left[
\max_{x'} \mu_{n+1}(x')
\right]
\]

この inner maximization が KG の計算コストと最適化難易度の大きな要因です。

BoTorch では fantasy points / fantasy samples と acquisition optimization を組み合わせてこの構造を扱います。

## 6.7 qKnowledgeGradient

複数候補を同時に観測する場合は、candidate batch \(X\) の joint future observations を考えます。

\[
Y_X\sim p(Y_X\mid\mathcal D_n)
\]

qKG は batch を観測した後の expected terminal decision value を評価します。

qEI と同様に、qKG も pointwise KG の上位 q 点を選ぶものではありません。候補間 correlation と joint fantasy outcomes が価値に影響します。

## 6.8 One-step と Multi-step Lookahead

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

## 6.9 Scenario tree と fantasy branching

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

## 6.10 qMultiStepLookahead

BoTorch の `qMultiStepLookahead` は、複数段階の fantasy decisions を含む lookahead acquisition を構成するための一般的な枠組みです。

これは特定の単一 closed-form acquisition というより、stage-wise value functions と fantasy branching を組み合わせる framework として理解する方が適切です。

各 stage で何を value とするか、terminal value をどう定義するかによって、構成される decision problem が変わります。

## 6.11 Batch と Multi-step の違い

両者は混同しやすいですが、明確に異なります。

### Batch

\[
(x_1,\ldots,x_q)
\]

を**観測結果を見る前に同時決定**します。

### Multi-step adaptive policy

\[
x_1
\rightarrow
Y_1
\rightarrow
x_2(Y_1)
\]

のように、後の decision が前の observation に依存します。

batch size q と lookahead depth は別の軸です。

## 6.12 Information-theoretic acquisition との違い

MES などは optimum についての information gain を直接評価します。

KG は terminal decision value の expected improvement を評価します。

```text
MES
    uncertainty reduction about optimum

KG
    expected improvement in terminal decision value
```

情報を得ること自体ではなく、その情報が最終 decision を改善する価値を持つかが KG の中心です。

## 6.13 Multi-Fidelity との関係

fidelity ごとに cost と information quality が異なる場合、KG の value-of-information interpretation は特に自然です。

候補 \((x,s)\) の観測が target fidelity の最終 decision をどれだけ改善するかを評価し、さらに evaluation cost を考慮できます。

この考え方は [Multi-Fidelity / Cost-aware](11_multifidelity_cost_aware.md) で扱います。

## 6.14 計算上の注意

Lookahead acquisition は通常、

- fantasy sampling
- posterior conditioning
- inner optimization
- outer acquisition optimization
- multi-step なら scenario branching

を含みます。

そのため myopic acquisition より高コストです。surrogate evaluation が安価でも acquisition optimization が支配的になることがあります。

lookahead depth を増やせば常に実用上優れるわけではなく、残り evaluation budget と acquisition computation cost のバランスが必要です。

## 6.15 BoTorch / robotorchan との対応

BoTorch は `qKnowledgeGradient` と `qMultiStepLookahead` を提供し、fantasy model を使った lookahead optimization をサポートします。

robotorchan はこれらの標準手法を再実装せず、BoTorch native path を基本とします。

利用上の位置付けは [Lookahead optimization guide](../../optimization/lookahead.md)、現在の統合状況は [Acquisition integration status](../../optimization/acquisition-status.md) を参照してください。

## 6.16 まとめ

Lookahead acquisition の中心は

\[
\text{value of information for future decisions}
\]

です。

KG は one-step の代表例で、multi-step lookahead は future observation に応じて後段の decision が変わる adaptive policy を扱います。

batch candidate selection と multi-step adaptive decision は別の概念であり、fantasy model は未知の将来観測を現在の時点で評価するための中核的な仕組みです。
