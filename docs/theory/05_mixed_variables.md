# 5. Mixed Variables in Gaussian Process Models

## 5.1 Variable semantics

A mixed design space combines continuous, integer, ordinal, nominal,
and possibly conditional variables. Their statistical meaning—not
the numerical dtype used to store them—determines the covariance
model and the feasible candidate set.

- **Continuous:** distances and interpolation are meaningful.
- **Ordinal:** order is meaningful, but equal spacing need not be.
- **Integer:** numerical distances may be meaningful, subject to
  discrete feasibility.
- **Nominal:** labels identify categories; their numeric codes do
  not define Euclidean distances.
- **Conditional:** some variables exist or are active only under
  particular parent choices.

A task index or fidelity level can also be stored as an integer,
but it represents a distinct statistical structure.

## 5.2 Categorical covariance

For nominal labels \(c,c'\), a simple valid kernel is

$$
k_{\mathrm{cat}}(c,c')
=
\begin{cases}
\sigma_c^2, & c=c',\\
\sigma_c^2\rho, & c\ne c',
\end{cases}
\qquad
-\frac{1}{K-1}\le\rho\le1,
$$

where \(K>1\) is the number of categories. Its covariance
matrix is \(\sigma_c^2[(1-\rho)I+\rho\mathbf 1\mathbf 1^\top]\);
the displayed bounds on \(\rho\) guarantee positive semidefiniteness.

A common alternative is the Hamming-distance kernel

$$
k_{\mathrm{cat}}(c,c')
=
\sigma_c^2\exp\bigl(-\lambda\,\mathbf 1[c\ne c']\bigr),
\qquad \lambda\ge0.
$$

For multiple categorical coordinates, products of such valid
kernels are valid. Category-specific embeddings and learned
coregionalization structures provide more expressive alternatives.

Treating category codes as coordinates in an RBF kernel instead
introduces arbitrary ordering and spacing assumptions.

## 5.3 Mixed covariance constructions

Write \(x=(x_{\mathrm{cont}},x_{\mathrm{cat}})\).
Let \(k_c\) be a valid continuous kernel and \(k_d\) a valid
categorical kernel. Two basic mixed kernels are

$$
k_{\mathrm{add}}(x,x')
=
k_c(x_{\mathrm{cont}},x'_{\mathrm{cont}})
+
k_d(x_{\mathrm{cat}},x'_{\mathrm{cat}})
$$

and

$$
k_{\mathrm{prod}}(x,x')
=
k_c(x_{\mathrm{cont}},x'_{\mathrm{cont}})
\,k_d(x_{\mathrm{cat}},x'_{\mathrm{cat}}).
$$

The additive kernel represents independent contributions,
whereas the product represents interactions whose correlation
depends on both variable types. Positive weighted sums of
these terms are also valid covariance functions.

An additive-only model can miss strong category-dependent
continuous effects; a product-only model can constrain
cross-category sharing too strongly. Kernel choice is an
inductive assumption to validate, not a universally optimal rule.

## 5.4 One-hot encoding and integer variables

One-hot encoding is a possible representation, but a generic
continuous kernel on encoded vectors is not automatically the
same as a desired nominal covariance. Moreover, an optimizer
must enforce one-hot feasibility; fractional indicator values
do not correspond to valid categories.

For integer-valued inputs, the choice between an ordered
numerical kernel and a nominal categorical kernel depends on
whether numerical proximity carries scientific meaning.
An integer design variable can be modeled numerically while
still requiring discrete candidate generation.

## 5.5 Surrogate and search-space constraints

A mixed Bayesian optimization problem can be expressed as

$$
\max_{x\in\mathcal X_{\mathrm{feasible}}}
\alpha(x;\mathcal D),
\quad
\mathcal X_{\mathrm{feasible}}
\subseteq
\mathcal X_c\times\mathcal X_d,
$$

where \(\mathcal X_c\) is continuous and \(\mathcal X_d\)
contains discrete assignments.

Modeling categorical similarity does **not** enforce
candidate feasibility. Common optimization strategies include
enumerating discrete assignments and optimizing continuous
variables conditionally, searching a finite candidate set, or
using mixed-integer/combinatorial optimizers.

Full enumeration becomes expensive when the number of
combinations grows as \(\prod_j K_j\).
Batch selection adds joint diversity and feasibility
considerations; the batch objective generally cannot be
reduced to independently maximizing a single-point score.

Conditional variables require a feasible-space representation
that respects their activation rules. This is a different
problem from merely choosing a categorical covariance.

## 5.6 Constraints and transformations

Input normalization should preserve the intended meaning of
categorical identifiers and structural columns. A linear
transformation of nominal codes can introduce spurious
distances.

Outcome constraints restrict which outcomes are acceptable;
candidate constraints restrict which design points may be
proposed. They are different mathematical objects and should
not be interchanged.

A complete mixed-variable BO formulation requires agreement
among the data representation, the surrogate covariance,
and the feasible acquisition search space.

## 5.7 Limitations

- High-cardinality categorical variables may have too little
  data per level to estimate meaningful similarities.
- Unseen categories require an explicit generalization
  assumption; nominal labels alone offer none.
- Category-dependent interactions can increase model
  complexity and data requirements.
- Integer, ordinal, nominal, task, and fidelity coordinates
  should not be conflated solely because all are encoded
  as integers.


## 5.8 混合変数GPの理論的補足

### カテゴリ共分散の半正定値性

5.2節の等相関カテゴリカーネルについて、$K$水準の共分散行列を
$C=\sigma_c^2[(1-\rho)I+\rho\mathbf 1\mathbf 1^\top]$とおきます。
$\mathbf 1$方向の固有値は$\sigma_c^2[1+(K-1)\rho]$、
その直交補空間の固有値は$\sigma_c^2(1-\rho)$です。
したがって、$-1/(K-1)\le\rho\le1$が半正定値性の必要十分条件になります。
これはカテゴリ番号間の距離を仮定せずに共分散を定義できる例です。

### 加法カーネルと積カーネルの帰納バイアス

加法カーネルは$f(x_c,x_d)=f_c(x_c)+f_d(x_d)$という構造に対応し、
カテゴリごとに異なる連続変数の応答形状を十分に表せない場合があります。
一方、積カーネルはカテゴリと連続変数の相互作用を表現できますが、
カテゴリ間の相関が小さい場合には情報共有も弱くなります。
和と積を組み合わせることで主効果と相互作用の両方を扱えます。
ただし、これらの表現はカーネルによる事前分布の仮定であり、
観測データが少ない場合に相互作用を確実に識別できるわけではありません。

### 条件付き変数と非活性次元

親カテゴリの値によって子変数が無効になる場合、
非活性次元に任意の数値を代入して通常の距離を計算すると、
同じ実験条件を異なる点として扱うおそれがあります。
非活性を明示した共分散、階層構造に応じた表現、または
有効な変数のみを比較する設計が必要です。
これはカテゴリカーネルだけでは解決しない問題です。

### q-batchでの候補選択

$q>1$の獲得関数では、候補を独立に最大化するだけでは
同一カテゴリや近接点に候補が集中することがあります。
候補間の事後相関を考慮する同時獲得関数と、
混合整数・条件付き制約を満たす候補生成を区別する必要があります。

これらはモデルの数学的性質です。個別のモデルクラス、入力列指定、
最適化バックエンドの使い方は[モデルドキュメント](../models/README.md)と
[最適化ドキュメント](../optimization/README.md)を参照してください。

## References

1. Garrido-Merchán, E. C., & Hernández-Lobato, D. (2020).
   Dealing with categorical and integer-valued variables in
   Bayesian optimization with Gaussian processes.
   *Neurocomputing*, 380, 20–35.
2. Rasmussen, C. E., & Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## Related documents

- [Kernel](03_kernel.md)
- [Acquisition Function](04_acquisition_function.md)
- [Hierarchical / Contextual GP](12_hierarchical_contextual_gp.md)
- [Multi-task / Multi-output](07_multitask_multioutput.md)
- [Mixed-variable model documentation](../models/README.md)
