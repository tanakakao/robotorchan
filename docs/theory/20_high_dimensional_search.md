# High-dimensional Search Strategies

## 直感

高次元 Bayesian optimization には少なくとも2つの異なる難しさがあります。

1. surrogate model が高次元入力から関数構造を学習する難しさ
2. acquisition function を高次元 candidate space で最適化する難しさ

例えば PCA で GP の内部入力を10次元へ落としても、public candidate が100次元で
`optimize_acqf` が100次元 box を探索するなら、candidate search の難しさは残ります。

### 最初に3つの座標系を区別する

高次元BOでは、同じ「低次元」という言葉でも座標系が異なることがあります。

| 座標 | 意味 |
| --- | --- |
| original/public X | objectiveを実際に評価する入力 |
| model latent Z_m | surrogate内部のPCA / neural representation |
| search target Z_s | REMBO / ALEBO / BAxUS等がcandidate探索に使う座標 |

`Z_m` と `Z_s` は同じとは限りません。candidate constraint、pending point、
trust-region weightがどの座標で定義されるかを常に確認します。

## 1. Original-space search


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

search strategyはcandidate generationの責務を持ち、surrogate modelと分離できます。

## 3. REMBO

REMBO は低次元 random embedding を固定し、


```text
x = project(A z)
```

として target space を探索します。

original boxから外れるprojectionはnormalized boxでclampしてpublic boundsへ写像します。

そのため異なるtarget coordinateが同じboundary pointへ写ることがあり、clippingはsearch geometryを
非線形に歪めます。ALEBOがfeasible polytopeを明示する理由の1つがこの違いです。

- random embedding の保持
- original-space projection
- target-space optimization
- original-space candidate の返却

を担当します。

したがって PCA による GP のような data-driven input reducer とは異なります。

REMBO の仮定は、目的関数が高次元 ambient space の中の低次元 effective subspace に主として
依存することです。

## 4. HeSBO

HeSBO は dense random matrix の代わりに sparse hashing-style embedding を使います。

各 ambient dimension を少数の target coordinates へ割り当てることで、projection cost と
embedding storage を小さくできます。


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

ALEBO 探索 のpolytope constraintsは**public process constraintではなく**、
projection後のcandidateをbox内へ保つためのinternal constraintsです。

ALEBO 探索 は、

- embedding
- target-space linear constraints
- feasibility 判定
- feasible sampling
- constrained target-space optimization
- original-space projection

を管理します。

## 6. ALEBO surrogate geometry

ALEBO の共分散モデル は embedded coordinates 上で Euclidean RBF をそのまま使うのではなく、
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
ALEBO 探索 = candidate search geometry
ALEBO の共分散モデル       = surrogate geometry
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

`generate_turbo_thompson_choices` は trust region 内に Sobol perturbations を作り、各 candidate
について一部 dimension だけを center から変更します。


```text
p = min(20 / D, 1)
```

です。

どの candidate も少なくとも1 dimension は perturb されるよう保証します。

## 12. 混合領域の局所探索

連続変数とカテゴリ変数に同じ距離を課すことは一般には適切ではありません。一つの設計は、連続座標のみ局所領域へ縮小し、整数・カテゴリの許容集合を保つことです。カテゴリ符号を連続 ARD 距離で縮める操作とは区別します。

## 13. Fidelity と設計座標

fidelity と物理設計座標は役割が異なります。設計座標に局所探索を行い、fidelity の選択可能範囲は維持する構成を考えられます。最終意思決定を評価する目標 fidelity と、次に観測する fidelity も区別します。

## 14. 座標変換と制約

元空間で定義した制約を探索座標へ移すには、射影との合成で同じ許容集合を表す必要があります。同じ係数を別の座標へそのまま渡しても同じ制約にはなりません。未完了観測の扱いと、候補が領域内にあるかの判定は別の問題です。

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

BAxUS の状態 は target dimension と trust-region state を組み合わせます。

failure によって local search が十分に縮小すると subspace expansion が必要になり、
embedding の split budget に従って target dimension を増やします。

したがって BAxUS は

- embedding strategy
- adaptive subspace dimension
- trust-region adaptation

を組み合わせた stateful search strategy です。

ARD の情報を探索側で使う場合、元座標と部分空間の幾何の関係を明示する必要があります。

## 17. BAxUS acquisition optimization と TS

BAxUS 探索 は embedded target space で acquisition を最適化し、original-space candidate を
返します。

このため「BAxUS = 1種類の optimizer」ではなく、adaptive subspace geometry と candidate
selection を分けて理解する方が正確です。

### State updateのutilityはscalar

TuRBO / BAxUSのsuccess / failure判定は、multi-objective outcome vectorそのものを比較する操作では
ありません。state updateへはcandidateごとのscalar utilityが必要です。

観測値と状態判定の効用は必ずしも同じではありません。ノイズ除去後や頑健性を考慮した効用を使う場合は、その定義を明示します。

## 18. 逐次的な状態更新

観測から予測モデルを更新し、獲得効用を構成、候補を選び、目的を観測してから探索状態を更新します。未観測の pending 点を完了観測として成功・失敗に数えないよう区別します。探索の状態と確率モデルのパラメータは別の量です。

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

## 22. この章で覚えておくこと

- surrogate-side dimension reductionとcandidate-search reductionは別問題である
- original X、model latent Z_m、search target Z_sを区別する
- REMBO / HeSBOはfixed embedding、ALEBOはfeasible linear subspaceを扱う
- REMBOのclippingとALEBOのpolytope feasibilityは異なるgeometryである
- TuRBOはoriginal/public spaceのlocal trust regionをstatefully適応する
- Mixed TuRBOではdiscrete dimensions、MultiFidelityではfidelity dimensionsをlocalizeしない
- BAxUSはsparse target subspace自体を探索中に拡張する
- TuRBO / BAxUSのstateはsurrogate parameterではなくBO loop側で更新する
- candidate constraintは座標変換後も同じ式とは限らず、未対応strategyでは明示的に拒否する
- model reductionとsearch embeddingを併用する場合、2つのlatent coordinateを混同しない

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
