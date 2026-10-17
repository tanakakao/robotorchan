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

## 5.8 カテゴリごとに異なる連続応答

混合入力では、カテゴリが変わると連続変数に対する応答曲面そのものが変わることがあります。
たとえば材料の製法カテゴリ \(c\) と温度 \(t\) に対し、加法カーネルのみを使うと
\(f(t,c)=g(t)+h(c)\) という構造を仮定します。この場合、カテゴリによって
温度依存性の形が変化する相互作用を十分に表せません。

積カーネル \(k_t(t,t')k_c(c,c')\) はカテゴリ間の情報共有を保ちながら
連続変数との相互作用を表します。加法項と積項を併用すると、共有された主効果と
カテゴリ固有の変化を同時に扱えます。ただし、データが少ない場合には
過度に柔軟な構造を識別できないことがあります。

## 5.9 条件付き変数と階層的な探索空間

親カテゴリ \(c\) に応じて子変数 \(u\) の有効性が変わる場合、
単純な直積空間 \(\mathcal X_c\times\mathcal X_u\) の全点が有効とは限りません。
たとえば、ある製法でのみ圧力が制御可能な場合、別の製法に対して
圧力の数値を割り当てて距離を計算することには物理的意味がありません。

この場合は有効性を表すマスク、条件付きカーネル、または階層構造に適した
共分散を検討します。候補生成でも同じ有効性条件を満たす必要があります。
**モデルが有効性を理解することと、最適化器が有効な候補だけを生成することは別の要件**です。

## 5.10 q-batchとカテゴリの組合せ

複数候補 \(X=(x_1,\ldots,x_q)\) を同時に提案する場合、獲得関数は
通常、候補間の事後共分散にも依存します。カテゴリの異なる候補を含めることが
常に望ましいわけではなく、探索・活用のトレードオフとカテゴリ間相関によって
最適な組合せは変わります。

固定カテゴリごとに連続最適化する方法は実装しやすい一方、
カテゴリの組合せ数が多い場合やバッチ全体に制約がある場合には
組合せ探索そのものが計算上のボトルネックになります。

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
