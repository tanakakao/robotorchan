# Mixed and Discrete Acquisition Optimization

## 4.1 Search space

Mixed optimization では candidate を

\[
x=(x_c,x_i,x_k)
\]

のように continuous \(x_c\)、integer \(x_i\)、categorical \(x_k\) に分けます。

robotorchan の `MixedVariableSpace` は public candidate coordinates 上で
各dimensionにちょうど1つの variable kind を割り当てます。

## 4.2 Integer と categorical は異なる

integer variable は順序と距離を持ちます。

\[
1 < 2 < 3
\]

のような関係が意味を持つため、rounding や整数mutationを定義できます。

一方 categorical variable の code が \(0,1,2\) でも、

\[
|0-1| < |0-2|
\]

という数値距離に意味があるとは限りません。

したがって unordered category を連続値として最適化し、
最後に nearest code へ丸める方法は人工的な geometry を導入します。

## 4.3 Enumeration / fixed-feature decomposition

categorical combination が有限で小さい場合、
各 categorical assignment を fixed feature として固定し、
残りの continuous subproblem を最適化できます。

概念的には

\[
\max_{c\in\mathcal C}\;
\max_{x\in\mathcal X_c}\alpha(x,c)
\]

です。

BoTorch mixed optimization はこの種の fixed-feature decomposition を利用できます。

## 4.4 Integer repair

continuous optimizer や population representation を利用しつつ、
評価前に

\[
x_i \leftarrow
\operatorname{clip}\!\left(\operatorname{round}(x_i),\ell_i,u_i\right)
\]

として integer feasibility を回復する方法があります。

ただし continuous proposal の多くが同じ整数値へ写像されるため、
元の continuous landscape と実際に評価される acquisition landscape は一致しません。

## 4.5 Type-aware evolutionary operator

Mixed GA では variable kind ごとに operator を変えられます。

- continuous: real-valued crossover / perturbation
- integer: integer-preserving mutation / repair
- categorical: allowed category から mutation

この方式では categorical variable に不要な数値距離を仮定せずに探索できます。

## 4.6 Sampling

Random / Sobol 系でも structured variable を評価前に
legal value へ写像する、または legal set から直接生成する必要があります。

単に bounds 内の実数を生成することと、
mixed feasible candidate を生成することは同義ではありません。

## 4.7 現在の backend 境界

現在の実装では代表的に次の違いがあります。

| Backend | Integer | Categorical | 主な semantics |
| --- | --- | --- | --- |
| BoTorch mixed | 対象構成による | yes | fixed-feature enumeration |
| Mixed GA | yes | yes | type-aware evolutionary search |
| Random / Sobol | yes | yes | structured sampling |
| DE | yes | no | integer repair |
| PSO | yes | no | integer repair |
| CMA-ES | no | no | continuous search |

DE は unordered categorical variable を明示的に拒否します。
PSO も categorical variable を現在は受け付けません。

## 4.8 Fixed features

fixed feature が integer dimension なら integer value、
categorical dimension なら allowed category である必要があります。

これは単なる bounds validation ではなく variable semantics の validation です。

## 4.9 Surrogate と optimizer の Mixed 対応

Mixed surrogate が category を正しく model 化できることと、
optimizer が category を正しく探索できることは別条件です。

```text
surrogate mixed capability
    !=
optimizer mixed capability
```

さらに optimizer の `mixed=True` だけでは categorical support を意味しません。
integer と categorical capability を個別に確認します。

## 4.10 q-batch

joint q-batch では structured dimension が各 candidate に存在します。

GA などが内部的に \([q,d]\) を flatten して \([qd]\) を探索する場合でも、
integer / categorical ownership は q 個の candidate 全体へ正しく展開する必要があります。

## 参考文献

1. Garrido-Merchán, E. C., & Hernández-Lobato, D. (2020).
   *Dealing with Categorical and Integer-valued Variables in Bayesian Optimization
   with Gaussian Processes*. Neurocomputing, 380, 20-35.
2. Daulton, S., Wan, X., Eriksson, D., Balandat, M., Osborne, M. A.,
   & Bakshy, E. (2022).
   *Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic
   Reparameterization*. NeurIPS 35.
3. BoTorch documentation. *Mixed search spaces* and `optimize_acqf_mixed`.
