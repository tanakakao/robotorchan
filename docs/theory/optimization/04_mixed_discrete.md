# Mixed and Discrete Acquisition Optimization

[混合変数のモデル化](../05_mixed_variables.md)と[初期化章](03_initialization.md)を前提とします。
本章では、予測モデルの共分散とは別に、許される候補を探索演算でどう保つかを考えます。

## 4.1 Search space

Mixed optimization では candidate を

$$
x=(x_c,x_i,x_k)
$$

のように continuous $x_c$、integer $x_i$、categorical $x_k$ に分けます。

各入力次元の変数型と許容集合を明示し、変数型に適した探索演算を定義します。

## 4.2 Integer と categorical は異なる

数量を表す整数変数では、数値の順序と距離に意味を持たせられます。保存形式が整数でも、名義ラベルならカテゴリとして区別します。

$$
1 < 2 < 3
$$

のような関係が意味を持つため、rounding や整数mutationを定義できます。

一方 categorical variable の code が $0,1,2$ でも、

$$
|0-1| < |0-2|
$$

という数値距離に意味があるとは限りません。

したがって unordered category を連続値として最適化し、
最後に nearest code へ丸める方法は人工的な geometry を導入します。

## 4.3 Enumeration / fixed-feature decomposition

categorical combination が有限で小さい場合、
各 categorical assignment を fixed feature として固定し、
残りの continuous subproblem を最適化できます。

概念的には

$$
\max_{c\in\mathcal C}\;
\max_{x\in\mathcal X_c}\alpha(x,c)
$$

です。

有限な割当集合の最大化としては、この分解は厳密です。ただし内側の非凸問題を局所的に解けば、外側で比較しても大域最適性は得られません。
q候補のカテゴリが独立にC通りずつあるなら、同時割当は一般に $C^q$ 通りです。
全候補を同じカテゴリへ固定するC通りの探索は、別の制限された問題になります。

## 4.4 Integer repair

continuous optimizer や population representation を利用しつつ、
評価前に

$$
x_i \leftarrow
\operatorname{clip}\!\left(\operatorname{round}(x_i),\ell_i,u_i\right)
$$

として integer feasibility を回復する方法があります。

ただし continuous proposal の多くが同じ整数値へ写像されるため、
元の連続な探索面と実際に評価する効用は一致しません。
上の式の整数境界は、元の実数境界から必要に応じて切り上げ・切り下げた許容整数とします。
許容整数集合が空なら、修復できる点はありません。

また、連続解を丸めても整数問題の最適解には限りません。
例えば $\alpha(x)=-(x-0.4)^2(4-3x)$ を $[0,1]$ で最大化すると連続解は0.4ですが、
丸めた0の値は $-0.64$、整数1の値は $-0.36$ なので整数問題では1の方が良い解です。

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

## 4.7 探索方式と変数型の整合性

fixed-feature enumeration、type-aware evolutionary search、structured sampling、
integer repair は異なる探索方式です。
整数の順序構造と unordered categorical の集合構造を区別する必要があります。
ライブラリ固有の対応状況は [Optimization guides](../../optimization/README.md) を参照してください。

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

整数と categorical の探索可能性は個別に評価する必要があります。

## 4.10 q-batch

joint q-batch では structured dimension が各 candidate に存在します。

GA などが内部的に $[q,d]$ を flatten して $[qd]$ を探索する場合でも、
integer / categorical ownership は q 個の candidate 全体へ正しく展開する必要があります。

## この章で覚えておくこと

列挙、型に応じた演算、整数修復は別の探索方法です。
連続提案を丸めると効用面と制約が変わるため、実際に評価する離散候補で比較します。
予測モデルがカテゴリを正しく扱えることは、探索の可行性や解の品質を保証しません。

## 参考文献

1. Garrido-Merchán, E. C., & Hernández-Lobato, D. (2020).
   *Dealing with Categorical and Integer-valued Variables in Bayesian Optimization
   with Gaussian Processes*. Neurocomputing, 380, 20-35.
2. Daulton, S., Wan, X., Eriksson, D., Balandat, M., Osborne, M. A.,
   & Bakshy, E. (2022).
   *Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic
   Reparameterization*. NeurIPS 35.
3. BoTorch documentation. *Mixed search spaces* and `optimize_acqf_mixed`.
