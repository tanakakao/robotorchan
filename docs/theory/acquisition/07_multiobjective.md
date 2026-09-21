# 7. Multi-objective Acquisition

## 7.1 単一の最良値から Pareto 集合へ

複数目的

\[
f(x)=\left(f_1(x),\ldots,f_m(x)\right)
\]

を同時に最大化する問題では、一般にすべての目的で唯一最良となる点は存在しません。

目的間に trade-off があるため、multi-objective Bayesian Optimization（MOBO）では単一の incumbent ではなく、Pareto dominance と Pareto front を扱います。

以下では各目的を最大化する convention を使います。最小化目的は符号反転などにより同じ convention へ変換できます。

## 7.2 Pareto dominance

目的ベクトル \(y,y'\in\mathbb R^m\) に対して、\(y\) が \(y'\) を Pareto dominate するとは、

\[
y_j\ge y'_j
\quad
\forall j
\]

かつ少なくとも一つの目的で strict inequality が成立することです。

他の feasible solution に dominate されない解を Pareto optimal solution と呼び、その目的空間上の集合が Pareto front です。

MOBO の目的は、通常この front を少ない評価回数で学習・改善することです。

## 7.3 Hypervolume

Pareto front の品質を測る代表的な指標が hypervolume（HV）です。

reference point \(r\) を Pareto front より悪い点として設定し、non-dominated objective vectors と \(r\) の間で支配される領域の体積を

\[
HV(\mathcal P;r)
\]

とします。

2目的なら面積、3目的なら体積、それ以上では高次元 volume です。

Hypervolume は、

- Pareto front が良い方向へ進むこと
- front が広い trade-off を覆うこと

の両方を一つの scalar measure に反映できます。

## 7.4 Reference point は尺度の一部

reference point は単なる実装引数ではありません。どの目的領域を hypervolume として評価するかを決める尺度の一部です。

最大化問題では通常、関心のある Pareto front より各目的で悪い点を使います。

reference point が不適切だと、

- 有用な Pareto 改善を評価できない
- 特定の目的領域を過度に重視する
- hypervolume improvement の比較自体が変わる

可能性があります。

したがって reference point の設計は EHVI / NEHVI の理論的問題設定に含まれます。

## 7.5 Hypervolume Improvement

現在の Pareto set を \(\mathcal P_n\) とし、新しい目的ベクトル \(y\) を追加したときの hypervolume improvement を

\[
HVI(y)
=
HV(\mathcal P_n\cup\{y\};r)
-
HV(\mathcal P_n;r)
\]

とします。

すでに Pareto front に dominate される点では improvement は 0 です。

## 7.6 Expected Hypervolume Improvement

候補 \(x\) の目的ベクトルは未知なので、posterior expectation を取ります。

\[
EHVI(x)
=
\mathbb E[
HVI(f(x))
\mid\mathcal D_n
]
\]

これが Expected Hypervolume Improvement（EHVI）です。

単目的 EI が scalar improvement の期待値を扱うのに対し、EHVI は Pareto front の hypervolume improvement の期待値を扱います。

## 7.7 qEHVI

batch candidate set

\[
X=(x_1,\ldots,x_q)
\]

では、q 点を同時に追加した場合の joint hypervolume improvement を評価します。

\[
qEHVI(X)
=
\mathbb E[
HV(\mathcal P_n\cup f(X);r)
-
HV(\mathcal P_n;r)
]
\]

候補間 posterior correlation と、候補同士が作る新しい non-dominated region が相互作用するため、pointwise EHVI の上位 q 点を選ぶこととは異なります。

## 7.8 Noisy Expected Hypervolume Improvement

observation noise がある場合、観測された objective vectors から作った Pareto front が latent Pareto front と一致するとは限りません。

Noisy Expected Hypervolume Improvement（NEHVI）は baseline points の latent objective values に関する uncertainty も考慮し、noise-aware な hypervolume improvement を評価します。

これは単目的の NEI と同じく、

```text
observed incumbent / front
```

と

```text
latent incumbent / front
```

を区別する考え方です。

batch / noisy setting では qNEHVI 系が重要になります。

## 7.9 Log formulation

hypervolume improvement が非常に小さい領域では、EI 系と同様に acquisition value や gradient の数値安定性が問題になります。

LogEHVI / LogNEHVI 系は、hypervolume improvement の意思決定原理を変えるものではなく、acquisition optimization の数値安定性を改善する formulation として理解します。

## 7.10 Scalarization

Multi-objective optimization を扱うもう一つの方法は、目的ベクトルを scalar utility へ変換することです。

weight vector \(w\) に対して

\[
s_w(f(x))
\]

を定義し、scalarized objective に対して単目的 acquisition を適用できます。

単純な weighted sum だけでなく、Pareto front の non-convex region も扱いやすい Chebyshev 型などの scalarization が利用されます。

## 7.11 ParEGO / NParEGO

ParEGO は iteration ごとに scalarization weight を変えながら scalar BO を行い、Pareto front の異なる領域を探索する考え方です。

Noisy / MC setting では NParEGO 系の acquisition が利用されます。

Hypervolume-based methods と scalarization-based methods は同じ MOBO 問題を異なる utility design で扱います。

```text
EHVI / NEHVI
    hypervolume improvement を直接評価

ParEGO / NParEGO
    scalarization を変えながら単目的 utility を評価
```

## 7.12 Hypervolume Knowledge Gradient

Hypervolume Knowledge Gradient（HVKG）は、現在の candidate が直接作る hypervolume improvement ではなく、観測によって将来の Pareto decision quality がどれだけ改善するかを value-of-information として扱います。

したがって HVKG は、

- multi-objective
- hypervolume
- lookahead / KG

が交差する手法です。

EHVI と HVKG の違いは、単目的の EI と KG の違いに対応して理解できます。

## 7.13 Objective correlation

複数目的が同じ入力 \(x\) に対して correlation を持つ場合、surrogate model がその依存構造を表現できるかは acquisition quality に影響します。

一方、ModelList のように目的ごとに独立 model を使う構成もあります。

Multi-objective acquisition が multi-output posterior を受け取れることと、surrogate が目的間 correlation をモデル化していることは別の話です。

## 7.14 Constraints を伴う MOBO

実問題では Pareto optimality だけでなく feasibility も必要です。

制約付き MOBO では、infeasible な objective improvement をそのまま hypervolume gain として扱わないよう constraint semantics を組み込みます。

constraint modeling と objective-space hypervolume は異なる責務です。詳細は [Constraints](08_constraints.md) で扱います。

## 7.15 目的数と計算量

hypervolume computation は目的数が増えるほど難しくなります。

目的数が多い many-objective setting では、

- Pareto set の増大
- hypervolume partitioning cost
- reference point sensitivity
- acquisition optimization difficulty

が顕著になります。

そのため目的数が増えた場合、hypervolume-based acquisition だけでなく scalarization-based methods も重要な選択肢になります。

## 7.16 BoTorch / robotorchan との対応

BoTorch は qLogEHVI、qLogNEHVI、qLogNParEGO、HVKG などの multi-objective acquisition を提供しています。

robotorchan は標準 MOBO acquisition をローカルに再実装せず、BoTorch native path を基本とします。

実際の利用方法は [Multi-objective optimization guide](../../optimization/multiobjective.md)、現在の統合状況は [Acquisition integration status](../../optimization/acquisition-status.md) を参照してください。

## 7.17 まとめ

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

reference point は hypervolume criterion の一部であり、noise がある場合は observed front と latent front の違いも考慮する必要があります。
