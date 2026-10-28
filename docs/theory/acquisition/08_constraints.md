# 8. Constrained Acquisition

[改善量章](02_improvement.md)と、正規分布の確率計算を前提とします。
未知制約を $c_j(x)\le0$、改善量を $I(x)\ge0$ とし、可行な結果の効用と可行確率の要件を区別します。

## 8.1 最初に2種類のconstraintを分ける

~~~text
unknown output / black-box constraint
    実験・評価するまで値が分からない
    -> surrogate + constrained acquisition

known candidate / input-space constraint
    candidate Xだけからfeasibilityを判定できる
    -> acquisition optimizer / search strategy
~~~

同じconstraintという語でも、uncertaintyを学習する問題とcandidate domainを制限する問題は
異なる数学的役割を持ちます。既知の入力制約は許容探索領域を定め、
未知の出力制約は事後分布を通じて実行可能性を評価します。

## 8.2 最適値だけでなくfeasibilityを考える

実問題では、目的関数

$$
f(x)
$$

を最大化するだけでなく、制約

$$
c_j(x)\le 0,
\qquad j=1,\ldots,J
$$

を満たす必要があります。

このとき最適化対象は

$$
\max_{x\in\mathcal X} f(x)
\quad
\text{subject to}
\quad
c_j(x)\le0
$$

です。

Bayesian Optimizationではobjectiveとconstraintsの両方が高コスト・未知である場合があり、
それぞれをsurrogate posteriorで表現します。

## 8.3 Feasibility probability

制約 $`c(x)\le0`$ に対して posterior

$$
c(x)\mid\mathcal D_n
\sim
\mathcal N(\mu_c(x),\sigma_c^2(x))
$$

を仮定すると、feasibility probability は

$$
P(c(x)\le0\mid\mathcal D_n)
=
\Phi\left(
\frac{-\mu_c(x)}{\sigma_c(x)}
\right)
$$

です（$\sigma_c(x)>0$ の場合）。$\sigma_c(x)=0$ なら
$P(c(x)\le0\mid\mathcal D_n)=\mathbf 1[\mu_c(x)\le0]$ です。

複数制約がposterior上で独立なら、joint feasibility probabilityを各確率の積として書ける場合が
あります。ただしconstraintsが相関している場合、この積は一般には正しくありません。

## 8.4 Objective utility と feasibility

直感的な constrained acquisition は、

$$
\alpha_{\mathrm{constrained}}(x)
\approx
\alpha_{\mathrm{objective}}(x)
\times
P(\text{feasible at }x)
$$

と理解できます。

例えば Expected Improvement と feasibility probability を組み合わせれば、

- objective improvement が大きそう
- かつ feasible である可能性が高い

候補を優先できます。

ただし、すべてのconstrained acquisitionがこの単純な積で定義されるわけではありません。
MC formulationではposterior samplesごとにobjective utilityとconstraint satisfactionを
組み合わせられます。

例えば improvement $I(x)$ と feasibility event $F(x)$ に対して、一般には

$$
\mathbb E[I(x)\mathbf 1_{F(x)}\mid\mathcal D_n]
\ne
\mathbb E[I(x)\mid\mathcal D_n]\,
P(F(x)\mid\mathcal D_n)
$$

です。右辺への分解が成立するのは、posterior の下で improvement と
feasibility indicator が独立である場合などに限られます。
目的と制約が相関する surrogate では joint samples による期待値が重要です。

例えば同じ確率の二つの状態で、一方だけが可行で改善量2、もう一方が不可行で改善量0とします。
真の可行改善の期待値は1ですが、改善量の期待値1に可行確率0.5を掛けると0.5になります。
目的と制約を独立に組み合わせる近似は、このような依存を失います。

## 8.5 Sample-wise constraint

posterior sample $`f^{(s)}(x)`$ と constraint sample $`c^{(s)}(x)`$ に対し、

$$
U^{(s)}(x)
=
I^{(s)}(x)
\mathbf 1[c^{(s)}(x)\le0]
$$

のように infeasible sample の utility を無効化できます。

複数 constraints なら

$$
\prod_{j=1}^{J}
\mathbf 1[c_j^{(s)}(x)\le0]
$$

を使えます。

数値最適化のため、指示関数を
$w_\eta(c)=1/(1+e^{c/\eta})$、$\eta>0$ で平滑化する方法があります。
$\eta$ は制約値と同じ単位を持ち、小さくすると硬い判定へ近づきますが、勾配が狭い境界へ集中します。
$c=0$ では常に $w_\eta=0.5$ なので、境界を含めた指示関数 $\mathbf1[c\le0]$ の値1と同じではありません。
連続な事後分布で境界の確率が零などの条件の下で、積分した効用の極限を考えます。

また、不可行標本の効用を零にする構成では、元の効用が非負であることが重要です。
可行な効用が負なら、零にした不可行標本の方が高い値になり得ます。改善量以外の目的には基準や罰則を別に定義します。

## 8.6 制約の符号規約を明示する

output / black-box constraintの説明では

$$
c(x)\le0
$$

を feasible とする convention がよく使われますが、数学的には逆符号で
$-c(x)\ge0$ と表現しても同じ実行可能領域を定義できます。

```text
c(x) <= 0 が feasible
```

なのか

```text
c(x) >= 0 が feasible
```

なのかを必ず確認する必要があります。

constraint の符号を誤ると acquisition は feasible / infeasible を逆に解釈します。

## 8.7 Known candidate constraintsとunknown output constraints

すべての constraints が surrogate modeling を必要とするわけではありません。

### Known candidate / input-space constraints

例えば

$$
Ax\le b
$$

のように入力から厳密に判定できる制約は、candidate optimization の feasible domain として扱えます。

### Unknown output / black-box constraints

実験しなければ分からない品質制約、安全制約、副生成物量などは posterior model を必要とします。

この二つは責務が異なります。

```text
known input constraint
    acquisition optimizer の探索領域を制限

unknown outcome constraint
    posterior uncertainty を含む constrained acquisition
```

## 8.8 予測から制約付き効用まで

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

## 8.9 Multi-output model と constraints

例えば model が

$$
(f(x),c_1(x),c_2(x))
$$

を同時に出力する場合、objective output と constraint outputs を同じ posterior sample から取り出せます。

一方、

- objective model
- constraint model 1
- constraint model 2

を別々に構成することもできます。

どちらを選ぶかは、output correlation をモデル化したいか、data availability が同じか、noise structure が共通かなどに依存します。

## 8.10 Feasible incumbentとnoise

Improvement-based constrained BO では、「現在の best」を feasible observations の中から定義する必要があります。

infeasible だが objective value が非常に高い観測を incumbent にすると、optimization target と整合しません。

さらに observation noise がある場合は feasibility 自体も不確実なので、単純な observed feasible best では不十分になる場合があります。

## 8.11 Feasible pointがまだない場合

初期データに feasible observation が一つもない場合、通常の improvement-based logic は扱いにくくなります。

この段階では、

- feasibility probability を高める
- constraint violation を減らす
- feasibility と objective learning を両立する

などの初期探索戦略が必要になります。

したがって constrained BO は「EI に feasibility probability を掛ければ常に終わり」ではありません。

## 8.12 Constraint boundaryの学習

制約付き最適化では、optimal solution が constraint boundary 付近に存在することが多くあります。

そのため constraint posterior の uncertainty、特に

$$
c(x)\approx0
$$

の領域を正しく学習することが重要です。

これは level-set estimation と数学的に近い側面がありますが、目的は異なります。

```text
Level-set
    boundary 自体を学ぶことが目的

Constrained BO
    feasible region 内で objective を最適化することが目的
```

## 8.13 Multi-objective constraints

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

可行確率を効用へ掛けることは、$P(c_j(x)\le0\ \forall j\mid D)\ge1-\delta$ という確率制約と異なります。
前者は大きな改善を見込んで低い可行確率を受け入れる場合もありますが、後者は候補に最低確率を課します。
どちらも事後分布の仮定に依存する判断であり、制約違反が一度も起きない保証を自動的に意味するものではありません。

## 8.14 Safety-critical constraints

制約違反そのものが許容できない safe optimization では、通常の constrained BO より強い要件があります。

一般的な constrained acquisition は「infeasible observation を絶対に発生させない」保証を自動的に与えるものではありません。

safe BO では safe set expansion や high-probability safety guarantee など、別の理論枠組みが必要になります。

したがって constraint-aware と safe を同義として扱わないことが重要です。

## 8.17 この章で覚えておくこと

既知の入力制約は候補の領域を、未知の出力制約は観測後の可行性を定めます。
改善量と可行確率の積への分解には独立性が必要であり、一般には同時標本で可行な効用を評価します。
平滑化を使う場合は、係数の単位、境界での値、効用が非負かどうかを確認します。

可行確率による重み付け、候補への確率制約、探索全体の安全性保証は異なる条件です。
多目的でも、ハイパーボリュームの基準点と制約閾値を混同せずに組み合わせます。

## 8.18 まとめ

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

特にknown candidate constraintとunknown output constraint、constraint-aware optimizationと
safe optimizationを混同しないようにします。

## References

- Gardner, J. R., Kusner, M. J., Xu, Z. E., Weinberger, K. Q., and Cunningham, J. P. (2014),
  *Bayesian Optimization with Inequality Constraints*. ICML.
- Gelbart, M. A., Snoek, J., and Adams, R. P. (2014), *Bayesian Optimization with Unknown
  Constraints*. UAI.
