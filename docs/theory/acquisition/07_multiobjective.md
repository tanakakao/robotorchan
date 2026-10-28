# 7. Multi-objective Acquisition

[改善量章](02_improvement.md)と[多出力GP章](../07_multitask_multioutput.md)を前提とします。
目的数を $m$ とし、各目的の向きを最大化へそろえます。目的空間のPareto frontと、それを実現する入力のPareto setは区別します。

## 7.1 Multi-outputとMulti-objectiveを分ける

~~~text
Multi-output model
    1つのxに対して複数の出力を予測するmodel contract

Multi-objective optimization
    複数のobjectiveを同時に意思決定へ使い、trade-offを最適化するproblem setting
~~~

出力が複数あるだけでPareto optimizationになるわけではありません。複数出力をscalarizeして
単目的utilityへ変換する場合もあります。逆にMOBO acquisitionを使うには、どのmodel outputsを
objectivesとして扱うかというdecision semanticsが必要です。

## 7.2 単一の最良値からPareto集合へ

複数目的

$$
f(x)=\left(f_1(x),\ldots,f_m(x)\right)
$$

を同時に最大化する問題では、一般にすべての目的で唯一最良となる点は存在しません。

目的間に trade-off があるため、multi-objective Bayesian Optimization（MOBO）では単一の incumbent ではなく、Pareto
dominance と Pareto front を扱います。

以下では各目的を最大化する convention を使います。最小化目的は符号反転などにより同じ convention へ変換できます。

## 7.3 Pareto dominance

目的ベクトル $`y,y'\in\mathbb R^m`$ に対して、$`y`$ が $`y'`$ を Pareto dominate するとは、

$$
y_j\ge y'_j
\quad
\forall j
$$

かつ少なくとも一つの目的で strict inequality が成立することです。

他の feasible solution に dominate されない解を Pareto optimal solution と呼び、その目的空間上の集合が Pareto front です。

MOBO の目的は、通常この front を少ない評価回数で学習・改善することです。

## 7.4 Hypervolume

Pareto front の品質を測る代表的な指標が hypervolume（HV）です。

reference point $`r`$ を Pareto front より悪い点として設定し、non-dominated objective vectors と $`r`$
の間で支配される領域の体積を

$$
HV(\mathcal P;r)
$$

とします。

2目的なら面積、3目的なら体積、それ以上では高次元 volume です。

Hypervolume は、

- Pareto front が良い方向へ進むこと
- front が広い trade-off を覆うこと

の両方を一つの scalar measure に反映できます。

最大化では、基準点を上回る目的ベクトル $y$ ごとの箱 $[r_1,y_1]\times\cdots\times[r_m,y_m]$ の和集合の体積です。
どれかの成分が基準を下回る点の箱は空とし、重なる体積は一度だけ数えます。
例えば $r=(0,0)$、非支配集合が $\{(2,1),(1,2)\}$ なら、面積は $2+2-1=3$ です。
新たに $(1.5,1.5)$ を追加すると、未被覆の $0.5\times0.5=0.25$ だけ改善します。
点数が増えたことではなく、新しく覆う目的領域が価値になります。

## 7.5 Reference pointはhypervolumeの基準

reference point は単なる実装引数ではありません。どの目的領域を hypervolume として評価するかを決める尺度の一部です。

最大化conventionでは、reference pointはhypervolumeを測りたいobjective regionの下側基準です。
通常は関心のあるnon-dominated objective vectorsより各目的で悪い値を設定します。

ただしreference pointを「training Yの最小値から機械的に決めれば常に正しい」とは考えません。
objective transform後の尺度、実務上意味のある最低水準、比較したいregionを含めて決めます。

reference point が不適切だと、

- 有用な Pareto 改善を評価できない
- 特定の目的領域を過度に重視する
- hypervolume improvement の比較自体が変わる

可能性があります。

したがって reference point の設計は EHVI / NEHVI の理論的問題設定に含まれます。

## 7.6 Hypervolume Improvement

ここでは $\mathcal P_n$ を、観測済みの目的ベクトルから得られる
**固定された非支配集合**とします。以下の通常の EHVI は、
この基準集合を確定値として扱う場合の定義です。

現在の非支配な目的ベクトル集合を $`\mathcal P_n`$ とし、新しい目的ベクトル $`y`$ を追加したときの hypervolume improvement を

$$
HVI(y)
=
HV(\mathcal P_n\cup\{y\};r)
-
HV(\mathcal P_n;r)
$$

とします。

すでに Pareto front に dominate される点では improvement は 0 です。

## 7.7 Expected Hypervolume Improvement

候補 $`x`$ の目的ベクトルは未知なので、posterior expectation を取ります。

$$
EHVI(x)
=
\mathbb E[
HVI(f(x))
\mid\mathcal D_n
]
$$

これが Expected Hypervolume Improvement（EHVI）です。

単目的 EI が scalar improvement の期待値を扱うのに対し、EHVI は Pareto front の hypervolume improvement の期待値を扱います。

## 7.8 qEHVI

batch candidate set

$$
X=(x_1,\ldots,x_q)
$$

では、q 点を同時に追加した場合の joint hypervolume improvement を評価します。

$$
qEHVI(X)
=
\mathbb E[
HV(\mathcal P_n\cup f(X);r)
-
HV(\mathcal P_n;r)
]
$$

候補間 posterior correlation と、候補同士が作る新しい non-dominated region が相互作用するため、pointwise EHVI の上位 q
点を選ぶこととは異なります。

## 7.9 Noisy Expected Hypervolume Improvement

observation noise がある場合、観測された objective vectors から作った Pareto front が latent Pareto front
と一致するとは限りません。

Noisy Expected Hypervolume Improvement（NEHVI）は baseline points の latent objective values に関する
uncertainty も考慮し、noise-aware な hypervolume improvement を評価します。

これは単目的の NEI と同じく、

```text
observed incumbent / front
```

と

```text
latent incumbent / front
```

を区別する考え方です。

batch / noisy settingではqNEHVI系が重要になります。これは単にEHVIのposterior varianceを
大きくすることではなく、baseline objective valuesとfront自体のlatent uncertaintyを扱います。

baseline $B$ の latent objective vectors から得られる非支配集合を
$\mathcal P(f(B))$ とすれば、制約なしの基本的な noisy hypervolume improvement は

$$
q\operatorname{NEHVI}(X)
=
\mathbb E\left[
HV(\mathcal P(f(B)\cup f(X));r)
-
HV(\mathcal P(f(B));r)
\mid\mathcal D_n
\right]
$$

と表せます。ここでは $f(B)\cup f(X)$ は目的ベクトルの集合の和集合を表します。
候補と baseline の latent outputs は同じ joint posterior に従い、
baseline Pareto front 自体もサンプルごとに変化します。
この式は基本的な意思決定量を示すものであり、実際の数値評価では
partitioning、baseline pruning、制約などを別途考慮します。

## 7.10 Log formulation

hypervolume improvement が非常に小さい領域では、EI 系と同様に acquisition value や gradient の数値安定性が問題になります。

LogEHVI / LogNEHVI 系は、hypervolume improvement の意思決定原理を変えるものではなく、acquisition optimization
の数値安定性を改善する formulation として理解します。

## 7.11 ScalarizationはMulti-output reductionでもある

Multi-objective optimizationを扱うもう一つの方法は、目的ベクトルをscalar utilityへ変換することです。
この操作によりmulti-output posteriorからacquisitionが評価するscalar decision quantityを作れます。

weight vector $`w`$ に対して

$$
s_w(f(x))
$$

を定義し、scalarized objective に対して単目的 acquisition を適用できます。

単純な weighted sum だけでなく、Pareto front の non-convex region も扱いやすい Chebyshev 型などの scalarization が利用されます。

重み付き和で全ての非支配点を選べるとは限りません。
例えば可行な目的ベクトルが $(1,0),(0,1),(0.4,0.4)$ の三つだけなら、全て非支配ですが、
正の重みの和では中央点は最大になりません。中央の値は $0.4(w_1+w_2)$、
端点の最大値は少なくとも $0.5(w_1+w_2)$ となるためです。

最大化向けのChebyshev型の一例は、同じ尺度へそろえた目的 $\tilde y_j$ に対し
$s_w(y)=\min_j(w_j\tilde y_j)+\rho\sum_jw_j\tilde y_j$（$w_j>0,\rho>0$）です。
上の三点で等重み・小さい $\rho$ なら、各目的を同時にある程度達成する中央点も選べます。
目的の正規化、重み、補助項の符号まで含めて効用を定義します。

## 7.12 ParEGO / NParEGO

ParEGO は iteration ごとに scalarization weight を変えながら scalar BO を行い、Pareto front の異なる領域を探索する考え方です。

Noisy / MC setting では NParEGO 系の acquisition が利用されます。

Hypervolume-based methods と scalarization-based methods は同じ MOBO 問題を異なる utility design で扱います。

```text
EHVI / NEHVI
    hypervolume improvement を直接評価

ParEGO / NParEGO
    scalarization を変えながら単目的 utility を評価
```

## 7.13 Hypervolume Knowledge Gradient

Hypervolume Knowledge Gradient（HVKG）は、現在の candidate が直接作る hypervolume improvement ではなく、観測によって将来の
Pareto decision quality がどれだけ改善するかを value-of-information として扱います。

したがって HVKG は、

- multi-objective
- hypervolume
- lookahead / KG

が交差する手法です。

EHVI と HVKG の違いは、単目的の EI と KG の違いに対応して理解できます。

## 7.14 Objective dependenceとmodel contract

複数目的が同じ入力 $`x`$ に対して correlation を持つ場合、surrogate model がその依存構造を表現できるかは acquisition quality に影響します。

一方、目的ごとに独立した予測モデルを置く構成もあります。

Multi-objective acquisitionがmulti-output posteriorを受け取れることと、surrogateがobjective間の
dependenceをmodel化していることは別の話です。

さらにcandidate間のjoint dependenceとobjective間のdependenceも別の軸です。q-batch MOBOでは
両方がacquisition samplesのjoint semanticsへ影響し得ます。

## 7.15 Constraints を伴う MOBO

実問題では Pareto optimality だけでなく feasibility も必要です。

制約付き MOBO では、infeasible な objective improvement をそのまま hypervolume gain として扱わないよう constraint
semantics を組み込みます。

constraint modelingとobjective-space hypervolumeは異なる責務です。ここでいうblack-box constraintは
model outputからfeasibilityを評価する制約です。既知のcandidate/input-space constraintは
探索領域の制限として別に扱います。詳細は
[Constraints](08_constraints.md) で扱います。

## 7.16 目的数と計算量

hypervolume computation は目的数が増えるほど難しくなります。

目的数が多い many-objective setting では、

- Pareto set の増大
- hypervolume partitioning cost
- reference point sensitivity
- acquisition optimization difficulty

が顕著になります。

そのため目的数が増えた場合、hypervolume-based acquisition だけでなく scalarization-based methods も重要な選択肢になります。

## 7.18 この章で覚えておくこと

多出力の予測と多目的の意思決定は別の問題です。
EHVIは基準点と非支配な目的ベクトル集合から新しく覆う体積を評価し、
NEHVIは基準集合の潜在値も不確実として扱います。基準点と目的の尺度は効用の定義の一部です。

スカラー化では別の効用でPareto領域を探索でき、重み付き和だけでは選べない非支配点もあります。
候補間と目的間の依存を分け、未知制約の可行性と入力の可行集合も別に定義します。

## 7.19 まとめ

MOBO では単一の incumbent ではなく Pareto structure を扱います。

```text
EHVI / NEHVI
    Pareto front の hypervolume improvement

NParEGO
    scalarization を通じた Pareto exploration

HVKG
    将来の multi-objective decision value
```

を区別することが重要です。

reference point は hypervolume criterion の一部であり、noise がある場合は observed front と latent front
の違いも考慮する必要があります。

## References

- Daulton, S., Balandat, M., and Bakshy, E. (2020), *Differentiable Expected Hypervolume
Improvement for Parallel Multi-Objective Bayesian Optimization*. NeurIPS.
- Daulton, S., Balandat, M., and Bakshy, E. (2021), *Parallel Bayesian Optimization of Multiple
Noisy Objectives with Expected Hypervolume Improvement*. NeurIPS.
- Ament, S. et al. (2023), *Unexpected Improvements to Expected Improvement for Bayesian
  Optimization*. Advances in Neural Information Processing Systems 36.
- Knowles, J. (2006), *ParEGO: A Hybrid Algorithm With On-Line Landscape Approximation for
  Expensive Multiobjective Optimization Problems*. IEEE Transactions on Evolutionary Computation,
  10(1), 50–66.
