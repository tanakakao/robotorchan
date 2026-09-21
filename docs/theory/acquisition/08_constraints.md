# 8. Constrained Acquisition

## 8.1 最適値だけでなく feasibility を考える

実問題では、目的関数

\[
f(x)
\]

を最大化するだけでなく、制約

\[
c_j(x)\le 0,
\qquad j=1,\ldots,J
\]

を満たす必要があります。

このとき最適化対象は

\[
\max_{x\in\mathcal X} f(x)
\quad
\text{subject to}
\quad
c_j(x)\le0
\]

です。

Bayesian Optimization では objective と constraints の両方が高コスト・未知である場合があり、それぞれを surrogate posterior で表現します。

## 8.2 Feasibility probability

制約 \(c(x)\le0\) に対して posterior

\[
c(x)\mid\mathcal D_n
\sim
\mathcal N(\mu_c(x),\sigma_c^2(x))
\]

を仮定すると、feasibility probability は

\[
P(c(x)\le0\mid\mathcal D_n)
=
\Phi\left(
\frac{-\mu_c(x)}{\sigma_c(x)}
\right)
\]

です。

複数制約が posterior 上で独立なら、joint feasibility probability を各確率の積として書ける場合があります。ただし constraints が相関している場合、この積は一般には正しくありません。

## 8.3 Objective utility と feasibility

直感的な constrained acquisition は、

\[
\alpha_{\mathrm{constrained}}(x)
\approx
\alpha_{\mathrm{objective}}(x)
\times
P(\text{feasible at }x)
\]

と理解できます。

例えば Expected Improvement と feasibility probability を組み合わせれば、

- objective improvement が大きそう
- かつ feasible である可能性が高い

候補を優先できます。

ただし、すべての constrained acquisition がこの単純な積で定義されるわけではありません。MC formulation では posterior samples ごとに objective utility と constraint satisfaction を組み合わせられます。

## 8.4 Sample-wise constraint

posterior sample \(f^{(s)}(x)\) と constraint sample \(c^{(s)}(x)\) に対し、

\[
U^{(s)}(x)
=
I^{(s)}(x)
\mathbf 1[c^{(s)}(x)\le0]
\]

のように infeasible sample の utility を無効化できます。

複数 constraints なら

\[
\prod_{j=1}^{J}
\mathbf 1[c_j^{(s)}(x)\le0]
\]

を使えます。

実装では hard indicator の代わりに smooth approximation を使い、gradient-based acquisition optimization を安定させる場合があります。

## 8.5 Constraint の符号 convention

constraint API では

\[
c(x)\le0
\]

を feasible とする convention がよく使われますが、文献やコードによって符号規約は異なります。

```text
c(x) <= 0 が feasible
```

なのか

```text
c(x) >= 0 が feasible
```

なのかを必ず確認する必要があります。

constraint の符号を誤ると acquisition は feasible / infeasible を逆に解釈します。

## 8.6 Known constraints と unknown constraints

すべての constraints が surrogate modeling を必要とするわけではありません。

### Known input constraints

例えば

\[
Ax\le b
\]

のように入力から厳密に判定できる制約は、candidate optimization の feasible domain として扱えます。

### Unknown outcome constraints

実験しなければ分からない品質制約、安全制約、副生成物量などは posterior model を必要とします。

この二つは責務が異なります。

```text
known input constraint
    acquisition optimizer の探索領域を制限

unknown outcome constraint
    posterior uncertainty を含む constrained acquisition
```

## 8.7 Objective、Constraint、Posterior Transform

BoTorch 的な構成を理解するには、次を分離することが重要です。

### Model output

surrogate が予測する raw outputs。

### Posterior transform

posterior の output structure を acquisition が必要とする形へ変換する処理。

### Objective

posterior samples から optimization utility の対象量を作る処理。

### Constraint

sample が feasible か、または feasibility weight をどう与えるかを定義する処理。

### Acquisition function

objective utility と uncertainty / feasibility を組み合わせ、candidate value を返す処理。

この責務分離により、同じ surrogate model を異なる optimization problem に再利用できます。

## 8.8 Multi-output model と constraints

例えば model が

\[
(f(x),c_1(x),c_2(x))
\]

を同時に出力する場合、objective output と constraint outputs を同じ posterior sample から取り出せます。

一方、

- objective model
- constraint model 1
- constraint model 2

を別々に構成することもできます。

どちらを選ぶかは、output correlation をモデル化したいか、data availability が同じか、noise structure が共通かなどに依存します。

## 8.9 Feasible incumbent

Improvement-based constrained BO では、「現在の best」を feasible observations の中から定義する必要があります。

infeasible だが objective value が非常に高い観測を incumbent にすると、optimization target と整合しません。

さらに observation noise がある場合は feasibility 自体も不確実なので、単純な observed feasible best では不十分になる場合があります。

## 8.10 Feasible point がまだない場合

初期データに feasible observation が一つもない場合、通常の improvement-based logic は扱いにくくなります。

この段階では、

- feasibility probability を高める
- constraint violation を減らす
- feasibility と objective learning を両立する

などの初期探索戦略が必要になります。

したがって constrained BO は「EI に feasibility probability を掛ければ常に終わり」ではありません。

## 8.11 Constraint boundary の学習

制約付き最適化では、optimal solution が constraint boundary 付近に存在することが多くあります。

そのため constraint posterior の uncertainty、特に

\[
c(x)\approx0
\]

の領域を正しく学習することが重要です。

これは level-set estimation と数学的に近い側面がありますが、目的は異なります。

```text
Level-set
    boundary 自体を学ぶことが目的

Constrained BO
    feasible region 内で objective を最適化することが目的
```

## 8.12 Multi-objective constraints

MOBO でも constraints の考え方は同じです。

目的空間では Pareto / hypervolume utility を評価し、constraint outputs では feasibility を評価します。

```text
objective outputs
    → Pareto / hypervolume utility

constraint outputs
    → feasibility

combined acquisition
    → feasible Pareto improvement
```

hypervolume reference point と constraint threshold は異なる概念なので混同しません。

## 8.13 Safety-critical constraints

制約違反そのものが許容できない safe optimization では、通常の constrained BO より強い要件があります。

一般的な constrained acquisition は「infeasible observation を絶対に発生させない」保証を自動的に与えるものではありません。

safe BO では safe set expansion や high-probability safety guarantee など、別の理論枠組みが必要になります。

したがって constraint-aware と safe を同義として扱わないことが重要です。

## 8.14 BoTorch / robotorchan との対応

BoTorch の MC acquisition は objective と constraints を組み合わせるための仕組みを提供しています。また known input constraints は acquisition optimization 側で扱える場合があります。

robotorchan は BoTorch-first とし、標準的な constrained acquisition semantics を独自 wrapper で置き換えません。

現在の対応状況は [Acquisition integration status](../../optimization/acquisition-status.md) を参照してください。

## 8.15 まとめ

Constrained BO では、

```text
objective
    何を良くしたいか

constraint
    何が feasible か

posterior
    それらについて何が不確実か

acquisition
    不確実性を含めて次にどこを評価するか

optimizer
    known input constraints を満たして候補を探す
```

を分離することが重要です。

特に known input constraint と unknown outcome constraint、constraint-aware optimization と safe optimization を混同しないようにします。
