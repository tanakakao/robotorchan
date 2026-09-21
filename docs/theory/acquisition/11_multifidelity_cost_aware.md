# 11. Multi-Fidelity and Cost-aware Acquisition

## 11.1 評価の価値とコストを同時に考える

Multi-Fidelity Bayesian Optimization（MFBO）では、同じ設計変数 \(x\) を異なる fidelity \(s\) で評価できます。

\[
y=f(x,s)
\]

高 fidelity は target quantity に近い一方で高コスト、低 fidelity は安価だが近似誤差を持つ、という構造が典型です。

例として、

- 粗い / 高精度 simulation
- 短時間 / 長時間 experiment
- 少量 / 大量 data training
- 小規模 / 実機 validation
- coarse / fine mesh

などがあります。

MFBO の中心的な問いは、

```text
どこ x を評価するか
```

だけでなく、

```text
どの fidelity s で評価するか
```

です。

## 11.2 Target fidelity

通常、最終的に最適化したい target fidelity \(s^*\) を定義します。

\[
x^*
=
\arg\max_x f(x,s^*)
\]

低 fidelity の目的は、それ自体の optimum を見つけることではなく、target-fidelity optimum の探索に有益な情報を安価に得ることです。

したがって fidelity variable を通常の design variable と完全に同じ意味で扱うべきではありません。

## 11.3 Fidelity と task の違い

Multi-task と multi-fidelity は数学的に似た surrogate structure を使える場合がありますが、意思決定上の意味は異なります。

```text
task
    複数の関連する出力・条件・データ源

fidelity
    target quantity への近似品質 / 解像度 / 評価レベル
```

fidelity にはしばしば順序、target fidelity、evaluation cost が存在します。

したがって multi-task model が存在することだけで MFBO acquisition が自動的に定義されるわけではありません。

## 11.4 Multi-Fidelity surrogate と acquisition は別

MFBO では少なくとも二つの責務があります。

### Multi-Fidelity surrogate

\[
p(f(x,s)\mid\mathcal D)
\]

をモデル化し、異なる fidelity 間の相関を学習します。

### Multi-Fidelity acquisition

候補 \((x,s)\) を評価したとき、それが target-fidelity decision にどれだけ価値を与えるかを評価します。

したがって、

```text
MF model を使っている
    ≠
MF acquisition を使っている
```

です。

単に fidelity を入力特徴量へ追加し、通常の EI で \((x,s)\) を探索するだけでは、target fidelity と cost の意味を十分に利用しない場合があります。

## 11.5 Fidelity 間の information transfer

低 fidelity が有用なのは、target fidelity と相関しているからです。

低 fidelity observation \(Y_{x,s}\) が target quantity \(f(x',s^*)\) にほとんど情報を与えないなら、いくら安価でも価値は限定的です。

MFBO acquisition は概念的に、

\[
\text{target-fidelity information / decision value}
\]

と

\[
\text{evaluation cost}
\]

の trade-off を扱います。

## 11.6 Evaluation cost

fidelity-dependent cost を

\[
c(x,s)>0
\]

とします。

単純な場合は fidelity のみで

\[
c(s)
\]

とできます。

cost model は、

- known deterministic cost
- analytic cost function
- measured cost から学習する surrogate

などとして表現できます。

cost は objective value と別の量です。高 fidelity が高い objective value を持つという意味ではありません。

## 11.7 Cost-aware utility

候補の information / decision value を \(V(x,s)\) とすれば、直感的な cost-aware criterion は

\[
\frac{V(x,s)}{c(x,s)}
\]

です。

これは「単位コストあたりの価値」を表します。

ただし ratio utility があらゆる budgeted decision problem の厳密な最適 policy になるわけではありません。remaining budget、terminal value、並列性、固定 overhead などによって最適な cost treatment は変わります。

## 11.8 Knowledge Gradient と Multi-Fidelity

Knowledge Gradient（KG）は候補を観測した後の terminal decision value の改善を評価するため、MFBO と自然に結び付きます。

候補 \((x,s)\) を評価した後、target fidelity \(s^*\) での最終 decision quality がどれだけ改善するかを

\[
V(x,s)
=
\mathbb E[
\text{terminal value after observing }(x,s)
]
-
\text{current terminal value}
\]

として評価できます。

低 fidelity でも target posterior を大きく改善するなら高い value を持ち得ます。

## 11.9 Multi-Fidelity Knowledge Gradient

Multi-Fidelity KG（MF-KG）は、candidate fidelity での observation が target-fidelity decision に与える value of information を評価します。

概念的には

\[
\alpha_{\mathrm{MFKG}}(x,s)
=
\operatorname{VOI}_{s^*}(x,s)
\]

です。

cost-aware utility を組み合わせれば、

\[
\alpha_{\mathrm{cost}}(x,s)
\approx
\frac{
\operatorname{VOI}_{s^*}(x,s)
}{
c(x,s)
}
\]

のように理解できます。

実際の実装では fantasy model、projected target points、cost-aware utility などが組み合わされます。

## 11.10 Projection to target fidelity

MF-KG では候補 \((x,s)\) を観測しても、最終的に評価したい terminal decision は target fidelity \(s^*\) 上にあります。

そのため design \(x\) を target fidelity へ写像する

\[
P(x,s)=(x,s^*)
\]

のような projection が重要になります。

```text
candidate
    (x, low fidelity)

information source
    ↓

terminal decision
    (x, target fidelity)
```

という役割分離です。

projection は「低 fidelity の観測値を高 fidelity 値へ変換する」操作ではなく、terminal decision をどの fidelity で評価するかを acquisition に伝えるものです。

## 11.11 Current value

KG 系では candidate observation 後の terminal value だけでなく、現在の terminal value が必要です。

MF setting ではこの current value も target fidelity 上で定義される必要があります。

したがって current value の計算と candidate fidelity の acquisition optimization を混同しません。

## 11.12 Continuous fidelity

fidelity \(s\) が連続なら、

\[
s\in[s_{\min},s_{\max}]
\]

として design variables と一緒に acquisition optimization できます。

ただし fidelity dimension は通常の design dimension と意味が異なるため、target projection や cost model では fidelity index を明示的に扱います。

continuous fidelity だからといって、posterior が fidelity に対して単調であるとは限りません。必要な構造は surrogate model 側で表現する必要があります。

## 11.13 Discrete fidelity

fidelity が

\[
s\in\{s_1,s_2,\ldots,s_K\}
\]

のような離散集合の場合、continuous relaxation より各 fidelity choice を明示的に扱う optimization が自然です。

例えば各 discrete fidelity を固定した acquisition optimization を比較できます。

```text
s = low
    → x を最適化

s = medium
    → x を最適化

s = high
    → x を最適化

最も acquisition value が高い組を選択
```

この問題は mixed / discrete acquisition optimization と関係します。

## 11.14 Fidelity variable と categorical variable

discrete fidelity は値が離散でも、単なる nominal categorical variable とは限りません。

```text
categorical variable
    category間に順序を仮定しないことが多い

fidelity
    quality / resolution / cost の構造を持つ
```

したがって fidelity column を categorical kernel に入れるかどうかは、値が離散であるという理由だけで決めるべきではありません。

modeling semantics と acquisition semantics の両方を確認する必要があります。

## 11.15 Multiple fidelity dimensions

複数の fidelity controls

\[
s=(s_1,\ldots,s_k)
\]

を持つ場合もあります。

例えば simulation で、

- spatial resolution
- temporal resolution
- solver tolerance

を別々に変えられる場合です。

このとき target fidelity は vector \(s^*\) となり、cost model と information transfer も多次元になります。

すべての fidelity dimensions が同じ cost / accuracy structure を持つとは限りません。

## 11.16 Learned cost model

evaluation cost が未知または変動する場合、cost 自体を surrogate model で学習できます。

ただし cost-aware utility で denominator として使う場合、cost prediction は正である必要があります。

raw Gaussian prediction をそのまま cost とすると負値を取り得るため、

- log-cost modeling
- positive transform
- strictly positive analytic form

などを検討します。

cost model uncertainty を acquisition にどう伝播させるかも別の設計問題です。

## 11.17 Affine fidelity cost

単純な fidelity cost は

\[
c(s)
=
c_0
+
\sum_j w_j s_j
\]

のような affine form で表現できます。

fidelity が高いほど cost が増える問題では扱いやすい近似です。

ただし実際の cost が非線形、段階的、hardware-dependent なら affine assumption は近似にすぎません。

cost model は実験系の実態に合わせて選択します。

## 11.18 Inverse-cost weighting

cost-aware utility では、decision value を inverse cost で重み付けする考え方があります。

```text
same value
    ↓
cheaper evaluation を優先

same cost
    ↓
higher value evaluation を優先
```

ただし非常に安価だがほぼ情報を持たない fidelity を無限に優先しないよう、value calculation と cost scale の両方が適切である必要があります。

## 11.19 Budget と cost-aware acquisition

総 budget

\[
B
\]

が明示される場合、各 step の value/cost ratio を最大化する greedy policy と、budget 全体を考えた optimal policy は一般に同じではありません。

残り budget が少ないときには、高価な target-fidelity evaluation を実行する必要がある場合もあります。

したがって cost-aware acquisition を「常に最安 fidelity を選ぶ仕組み」と理解してはいけません。

## 11.20 Multi-Fidelity MES

information-theoretic acquisition も multi-fidelity へ拡張できます。

候補 \((x,s)\) の観測が target-fidelity optimum \(f^*_{s^*}\) について与える information gain を評価し、cost を考慮できます。

概念的には

\[
\frac{
I(Y_{x,s};f^*_{s^*}\mid\mathcal D)
}{
c(x,s)
}
\]

です。

MF-KG が terminal decision value を基準にするのに対し、MF-MES 系は optimum に関する information gain を基準にします。

## 11.21 Multi-Fidelity と Multi-Task の組合せ

実問題では task \(t\) と fidelity \(s\) の両方を持つことがあります。

\[
f(x,t,s)
\]

例えば複数材料系 / 装置 / 条件 task について、simulation fidelity も変えられる場合です。

このとき surrogate は task correlation と fidelity correlation の両方を表現する必要があります。

さらに acquisition は、

```text
どの x
どの task
どの fidelity
```

を意思決定するのか、あるいは task は固定され fidelity だけを選ぶのかを明示する必要があります。

model が表現できることと acquisition が選択対象にすることは別です。

## 11.22 Multi-Fidelity と Mixed Variables

design space に continuous / categorical variables があり、さらに fidelity variable がある場合、candidate optimization は mixed structure を持ちます。

ただし構造列としての fidelity を通常の categorical design variable と混同しません。

```text
design categorical variable
    製品種、材料種、装置種など

fidelity variable
    evaluation quality / resolution

task variable
    related task identity
```

それぞれの semantics を保ったまま model と acquisition optimizer を設計します。

## 11.23 Stopping と最終評価

MFBO では探索終了時に target fidelity で最終候補を確認することが重要です。

低 fidelity observations が多数あっても、最終 recommendation は target-fidelity posterior / observation に基づく必要があります。

stopping rule としては、

- budget exhaustion
- target-fidelity decision uncertainty
- expected value of information が cost に見合わない
- terminal recommendation の安定化

などが考えられます。

## 11.24 BoTorch / robotorchan との対応

BoTorch は Multi-Fidelity Knowledge Gradient、cost-aware utility、fidelity cost model、target-fidelity projection、および discrete fidelity を扱う acquisition optimization primitives を提供しています。

robotorchan はこれらの標準機能を再実装せず、BoTorch native path を基本とします。

利用上の位置付けは [Multi-Fidelity / Cost-aware optimization guide](../../optimization/multifidelity-cost-aware.md)、現在の統合状況は [Acquisition integration status](../../optimization/acquisition-status.md) を参照してください。

## 11.25 まとめ

Multi-Fidelity acquisition の中心は、

```text
cheap or expensive?
```

ではなく、

```text
その fidelity の観測が
target-fidelity decision にどれだけ価値を与え、
そのためにいくら cost が必要か
```

です。

特に、

```text
Multi-Fidelity surrogate
    fidelity間の posterior structure

Multi-Fidelity acquisition
    target-fidelity decisionへの観測価値

Cost model
    evaluation expense

Acquisition optimizer
    x と fidelity の候補探索
```

を分離して理解することが重要です。

次章 [Selection Guide](12_selection_guide.md) では、ここまでの acquisition families を problem setting から選択するための整理を行います。
