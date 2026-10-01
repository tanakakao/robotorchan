# High-dimensional Search Strategies

## 直感

高次元 Bayesian optimization には少なくとも2つの異なる難しさがあります。

1. surrogate model が高次元入力から関数構造を学習する難しさ
2. acquisition function を高次元 candidate space で最適化する難しさ

例えば PCA で GP の内部入力を10次元へ落としても、public candidate が100次元で
`optimize_acqf` が100次元 box を探索するなら、candidate search の難しさは残ります。

robotorchan はこのため、**surrogate-side reduction** と **search strategy** を分離します。

## 1. Original-space search

通常の BoTorch optimization は

```text
X* = argmax_X alpha(X)
     subject to X in bounds
```

として public input space で acquisition を最適化します。

dimension が増えると、

- multistart initial condition の探索
- local optimization
- constraint handling
- batch candidate の joint optimization

が難しくなります。

高次元 search strategy は、この candidate geometry 自体を変える方法です。

## 2. Search embedding

embedding-based search は低次元 target coordinate `z` から original candidate `x` を作ります。

```text
z in R^d
 -> embedding / projection
 -> x in R^D
 -> alpha(x)

d << D
```

重要なのは、acquisition model が必ずしも embedded-space GP である必要はないことです。
search strategy は candidate generation の責務を持ち、surrogate model と分離できます。

## 3. REMBO

REMBO は低次元 random embedding を固定し、

```text
x = project(A z)
```

として target space を探索します。

original box から外れる projection は public bounds へ写像します。

robotorchan の `REMBOStrategy` は、

- random embedding の保持
- target-space acquisition wrapper
- original-space projection
- target-space optimization
- original-space candidate の返却

を担当します。

したがって `PCAGP` のような data-driven input reducer とは異なります。

REMBO の仮定は、目的関数が高次元 ambient space の中の低次元 effective subspace に主として
依存することです。

## 4. HeSBO

HeSBO は dense random matrix の代わりに sparse hashing-style embedding を使います。

各 ambient dimension を少数の target coordinates へ割り当てることで、projection cost と
embedding storage を小さくできます。

robotorchan の `HeSBOStrategy` も public contract は REMBO と同様で、

```text
target candidate
 -> sparse projection
 -> original-space candidate
 -> acquisition evaluation
```

です。

PCA / PLS のように training data から projection direction を学習する手法ではありません。

## 5. ALEBO

ALEBO は linear embedding を使いますが、単純な clipped random embedding とは異なります。

original-space box

```text
l <= A z <= u
```

は target space では linear inequalities に対応し、一般には axis-aligned box ではなく
polytope になります。

`ALEBOStrategy` は、

- embedding
- target-space linear constraints
- feasibility 判定
- feasible sampling
- constrained target-space optimization
- original-space projection

を管理します。

## 6. ALEBO surrogate geometry

robotorchan では search strategy に加えて `ALEBOGP` があります。

`ALEBOGP` は embedded coordinates 上で Euclidean RBF をそのまま使うのではなく、
Mahalanobis geometry を持つ covariance を提供します。

概念的には

```text
k(z, z')
  = sigma^2 exp(
      -1/2 (z - z')^T M (z - z')
    )
```

です。

実装は metric parameter の fitting、Laplace covariance estimation、metric parameter sampling、
metric-marginal posterior / acquisition model まで持ちます。

したがって

```text
ALEBOStrategy = candidate search geometry
ALEBOGP       = surrogate geometry
```

であり、責務は異なります。

## 7. TuRBO の基本

TuRBO は global box 全体で毎回 acquisition を最適化せず、現在の incumbent 周辺に
trust region を置きます。

```text
global bounds
 -> local trust region
 -> candidate search
 -> observation
 -> success / failure
 -> resize trust region
```

robotorchan の `TuRBOState` は少なくとも

- trust-region length
- minimum / maximum length
- success counter
- failure counter
- success / failure tolerance
- best value
- observed best value
- restart flag / count

を保持します。

## 8. TuRBO state update

candidate batch が改善を生んだかどうかで counter を更新します。

success が一定回数続けば

```text
length <- min(2 * length, length_max)
```

failure が一定回数続けば

```text
length <- length / 2
```

とします。

`length < length_min` になると restart が必要になります。

robotorchan は observed values と、任意の `state_values` を分けられます。
これにより raw observation とは別の denoised / robust utility で trust-region state を
更新する構成も可能です。

## 9. TuRBO restart

restart は単に state object を作り直すことではありません。

`restart_turbo_state` は global best を保持しつつ、

- trust-region length
- success / failure counters
- restart flag

を reset し、restart count を増やします。

新しい center が必要な場合、`generate_turbo_restart_center` は global bounds 内から
Sobol point を生成できます。

## 10. ARD trust-region geometry

TuRBO の local box は全 dimension を同じ物理幅で縮める必要はありません。

robotorchan は dimension weights を geometric mean 1 に正規化し、

```text
half_width_d
  = 0.5 * length
    * weight_d
    * global_width_d
```

として trust region を作ります。

`turbo_dimension_weights_from_model` は model の lengthscale から public-space weights を
導出できます。

ただし model が reduced input space で lengthscale を持つ場合、その latent lengthscale を
original-space TuRBO geometry として直接使うことはできません。

その場合は public-space dimension weights を明示的に与える必要があります。

これは surrogate reduction と search geometry を混同しないための重要な correctness boundary です。

## 11. TuRBO Thompson sampling

robotorchan は local acquisition optimization に加えて Thompson-sampling candidate generation を
提供します。

`generate_turbo_thompson_choices` は trust region 内に Sobol perturbations を作り、各 candidate
について一部 dimension だけを center から変更します。

default perturbation probability は概念的に

```text
p = min(20 / D, 1)
```

です。

どの candidate も少なくとも1 dimension は perturb されるよう保証します。

`TuRBOStrategy.thompson_sample()` はこの local candidate pool と posterior sampling を組み合わせます。

## 12. Mixed TuRBO

Mixed search space では continuous dimension と discrete dimension に同じ trust-region geometry を
適用しません。

`turbo_mixed_trust_region_bounds` は

- continuous design dimensions: local trust region へ縮小
- integer / categorical dimensions: global range を保持

します。

`TuRBOStrategy.optimize_mixed()` は mixed-variable optimization backend と組み合わせて candidate を
生成します。

これによりカテゴリコードを連続 ARD geometry として縮小することを避けます。

## 13. MultiFidelity TuRBO

MultiFidelity search でも fidelity coordinate と design coordinate は役割が異なります。

`turbo_multifidelity_trust_region_bounds` は design dimensions を localize しつつ、
fidelity dimensions は global range に保持します。

したがって trust region は

```text
design space    -> local
fidelity space  -> global
```

という構造を取ります。

`TuRBOStrategy.optimize_multifidelity()` はこの geometry を MultiFidelity candidate generation と
接続します。

## 14. Constraints と pending points

local search を使っても candidate feasibility を失ってはいけません。

TuRBO implementation は public-space candidate constraints と組み合わせられる経路を持ち、
生成候補の feasibility を確認します。

また asynchronous BO では unresolved candidate を `X_pending` として扱う必要があります。

pending points は public input coordinates で shape / finite validation されます。

この点は stateful trust-region strategy と batch / async BO を組み合わせる際に重要です。

## 15. BAxUS

BAxUS は fixed-dimensional embedding ではなく、探索中に target subspace dimension を拡張します。

```text
small target subspace
 -> search
 -> repeated failures
 -> split embedding
 -> larger target subspace
 -> continue
```

intrinsic dimension が未知でも、最初から大きな target dimension を固定せず探索できます。

## 16. BAxUS state

`BAxUSState` は target dimension と trust-region state を組み合わせます。

failure によって local search が十分に縮小すると subspace expansion が必要になり、
embedding の split budget に従って target dimension を増やします。

したがって BAxUS は

- embedding strategy
- adaptive subspace dimension
- trust-region adaptation

を組み合わせた stateful search strategy です。

## 17. BAxUS acquisition optimization と TS

`BAxUSStrategy` は embedded target space で acquisition を最適化し、original-space candidate を
返します。

加えて robotorchan には Thompson-sampling candidate generation の経路があり、
target-space candidate pool を original space へ投影した上で posterior sampling を行います。

このため「BAxUS = 1種類の optimizer」ではなく、adaptive subspace geometry と candidate
selection を分けて理解する方が正確です。

## 18. Stateful strategy contract

TuRBO / BAxUS の state は surrogate model の parameter ではありません。

典型的な BO loop は

```text
fit surrogate
 -> build acquisition
 -> strategy.optimize(...)
 -> evaluate objective
 -> strategy.update_state(...)
 -> append data
 -> refit surrogate
```

です。

surrogate model は objective evaluation を所有せず、search strategy も surrogate training を
暗黙に所有しません。

この責務分離により、同じ surrogate / acquisition と異なる search strategy を比較できます。

## 19. Surrogate reduction との組合せ

surrogate-side reduction と search-side embedding を組み合わせること自体は可能ですが、
coordinate system を明示する必要があります。

```text
search target z_s
 -> original candidate x
 -> surrogate reducer r(x)
 -> GP latent z_m
 -> acquisition
```

ここには2種類の latent coordinate が存在します。

- `z_s`: search strategy の target coordinate
- `z_m`: surrogate model の reduced coordinate

両者は同一とは限りません。

二重 embedding を使う場合は、

- bounds
- candidate constraints
- pending points
- trust-region weights
- gradient
- feasibility
- returned candidate

がどの空間に属するかを明示する必要があります。

## 20. Search strategy の使い分け

| Strategy | Search geometry | Stateful | Target dim |
| --- | --- | --- | --- |
| REMBO | dense random embedding | No | fixed |
| HeSBO | sparse hash embedding | No | fixed |
| ALEBO | feasible linear subspace | No | fixed |
| TuRBO | local trust region | Yes | original |
| BAxUS | adaptive sparse subspace + TR | Yes | expanding |

これは優劣の順位ではなく、探索 geometry の違いです。

## 21. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- REMBO / HeSBO は search embedding であり fitted input reducer ではない
- embedding strategy は original-space candidate を返す
- ALEBO は target-space feasible polytope を明示的に扱う
- `ALEBOStrategy` と `ALEBOGP` は別責務
- TuRBO は success / failure により trust-region length を適応
- TuRBO は restart state と restart center generation を持つ
- TuRBO は public-space ARD dimension weights を扱う
- reduced-model latent lengthscale を public-space TuRBO weight として誤用しない
- TuRBO は acquisition optimization と Thompson sampling の両経路を持つ
- Mixed TuRBO は discrete dimensions を continuous TR から分離
- MultiFidelity TuRBO は fidelity dimensions を design TR から分離
- constraints / pending points は public-space semantics を維持
- BAxUS は target dimension を statefully expansion
- BAxUS は acquisition optimization と Thompson-sampling candidate generation を持つ
- state update は objective evaluation / surrogate fitting と分離
- surrogate latent space と search target space は別座標系

詳細は
[high-dimensional search strategies](../high_dimensional_search_strategies.md)
を参照してください。

## 参考文献

1. Wang, Z., Zoghi, M., Hutter, F., Matheson, D., and de Freitas, N. (2013).
   Bayesian optimization in high dimensions via random embeddings.
   *International Joint Conference on Artificial Intelligence*.
2. Nayebi, A., Munteanu, A., and Poloczek, M. (2019).
   A framework for Bayesian optimization in embedded subspaces.
   *International Conference on Machine Learning*.
3. Letham, B., Calandra, R., Rai, A., and Bakshy, E. (2020).
   Re-examining linear embeddings for high-dimensional Bayesian optimization.
   *Advances in Neural Information Processing Systems*.
4. Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., and Poloczek, M. (2019).
   Scalable global optimization via local Bayesian optimization.
   *Advances in Neural Information Processing Systems*.
5. Papenmeier, L., Nardi, L., and Poloczek, M. (2022).
   Increasing the scope as you learn: Adaptive Bayesian optimization in nested subspaces.
   *Advances in Neural Information Processing Systems*.
6. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [Dimensionality Reduction GP](18_dimensionality_reduction_gp.md)
- [Neural Representation GP](19_neural_reduction_gp.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
- [Advanced High-dimensional Models](21_advanced_high_dimensional_models.md)
