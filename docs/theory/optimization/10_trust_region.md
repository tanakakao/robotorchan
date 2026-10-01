# Trust-region Acquisition Optimization

## 10.1 目的

Trust-region Bayesian Optimization は、毎 iteration で global domain 全体を探索する代わりに、
現在の incumbent 周辺へ局所探索領域を置き、その領域を観測結果に応じて適応させます。

acquisition function を `alpha(X)`、global bounds を `X_global` とすると、通常の探索は

    X* = argmax_X alpha(X),  X in X_global

ですが、trust-region search では iteration t ごとに

    X* = argmax_X alpha(X),  X in X_TR,t

を解きます。

重要なのは、trust region が surrogate parameter ではなく **candidate search state** だという点です。

## 10.2 TuRBO の基本構造

TuRBO (Trust Region Bayesian Optimization) は局所 BO と adaptive trust region を組み合わせます。

    center_t
      -> local trust region
      -> candidate generation
      -> objective evaluation
      -> success / failure
      -> resize / restart

robotorchan では `TuRBOState` と `TuRBOStrategy` がこの責務を分担します。

- `TuRBOState`: length、counter、best value、restart state
- `TuRBOStrategy`: center、local bounds、candidate generation、state update
- surrogate / acquisition: objective の確率モデルと utility

この分離により、同じ surrogate / acquisition に異なる探索戦略を適用できます。

## 10.3 Persistent state

`TuRBOState` は少なくとも次を保持します。

- `dim`
- `batch_size`
- `length`, `length_min`, `length_max`
- `success_counter`, `failure_counter`
- `success_tolerance`, `failure_tolerance`
- `best_value`
- `observed_best_value`
- `restart_triggered`, `restart_count`

default failure tolerance は

    ceil(max(4 / batch_size, dim / batch_size))

です。

したがって batch size と dimension は state transition の速度にも影響します。

## 10.4 Success / failure

completed batch の scalar utility の最大値を `y_new`、state best を `y_best` とします。

robotorchan は単なる `y_new > y_best` ではなく、

    y_new > y_best + eta * max(1, abs(y_best))

を meaningful improvement とみなします。

`eta` は `relative_improvement` で、default は `1e-3` です。

success なら success counter を増やして failure counter を0へ戻し、
failure なら逆の操作を行います。

## 10.5 Trust-region length update

success counter が tolerance に達すると、

    length <- min(2 * length, length_max)

failure counter が tolerance に達すると、

    length <- length / 2

です。

counter は resize 後に0へ戻ります。

縮小後に

    length < length_min

となると `restart_triggered=True` になります。

この状態では robotorchan は candidate generation を継続せず、明示的な restart を要求します。

## 10.6 Observed value と state utility

ノイズ、robust utility、denoised estimate を使う場合、raw observation と
trust-region decision に使う値を分けたいことがあります。

`update_turbo_state(..., values, state_values=...)` では、

- `values`: `observed_best_value` を更新
- `state_values`: success / failure と `best_value` を更新

という分離が可能です。

`state_values=None` なら両方に `values` を使います。

これは観測値を保存しつつ、別の caller-owned scalar utility で局所探索状態を更新するための
runtime contract です。

## 10.7 Multi-objective outcome

TuRBO state update は1 candidateにつき1つの scalar utility を要求します。

したがって multi-objective outcome

    y in R^m,  m > 1

をそのまま `update_state()` へ渡すことはできません。

hypervolume contribution、scalarization、robust utility など、どの値を探索状態へ使うかは
caller が明示的に決めて scalar へ還元します。

これは acquisition が multi-objective であることと、trust-region state update の utility を
混同しないための境界です。

## 10.8 Center update

`TuRBOStrategy.update_state(..., candidates=X)` に candidate を渡した場合、
meaningful improvement を生んだ batch 内 candidate が新しい center になります。

candidate を渡さない場合、state value は更新されても center は暗黙には変更されません。

したがって

    state adaptation
    center movement

は関連しますが、同一操作ではありません。

## 10.9 Batch-size contract

TuRBO state は `batch_size` を persistent state として持ちます。

robotorchan では candidate generation と state update の双方で

    q == state.batch_size

を要求します。

たとえば `batch_size=4` の state に対して `q=1` で最適化したり、
4点中2点だけで state update したりすることは許可されません。

batch の意味を iteration 間で変えないための correctness contract です。

## 10.10 Trust-region geometry

global bounds を `[l_d, u_d]`、center を `c_d`、dimension weight を `w_d` とすると、
local half width は

    h_d = 0.5 * length * w_d * (u_d - l_d)

です。

したがって

    lower_d = max(c_d - h_d, l_d)
    upper_d = min(c_d + h_d, u_d)

として global domain に clip します。

`length` は絶対座標幅ではなく、各 global input range に対する相対 scale です。

## 10.11 ARD dimension weights

dimension weights は geometric mean が1になるよう正規化されます。

GP が public input space の ARD lengthscale `ell_d` を持つ場合、
robotorchan はその情報から anisotropic trust-region geometry を構成できます。

bounds が与えられる場合は input range で lengthscale を正規化してから weight を作ります。

これにより単位や physical range が異なる入力でも、raw lengthscale をそのまま比較することを
避けます。

## 10.12 Reduced-space surrogate との境界

PCA、PLS、AE などの reduced-space model が持つ latent lengthscale は、
original/public input dimension の TuRBO geometry ではありません。

したがって

    latent ARD lengthscale
      != public-space TuRBO dimension weight

です。

robotorchan は model が input reduction を明示している場合、
その lengthscale から public-space weight を自動生成することを拒否します。

必要なら caller が public-space `dimension_weights` を明示します。

## 10.13 Acquisition optimization path

`TuRBOStrategy.optimize()` は現在の trust-region bounds を作り、
通常の robotorchan acquisition optimizer dispatcher へ渡します。

したがって TuRBO は acquisition optimizer を置き換えるのではなく、

    global bounds
      -> local bounds
      -> selected optimizer

という geometry layer として働きます。

optimizer backend の gradient-based / derivative-free 等の性質は引き続き適用されます。

## 10.14 Thompson-sampling path

TuRBO は acquisition maximization 以外に posterior sampling path も持ちます。

`generate_turbo_thompson_choices()` は local bounds 内に Sobol perturbation pool を生成します。

default perturbation probability は

    p = min(20 / dim, 1)

です。

各 pool candidate は center の一部 dimension だけを変更し、
何も変更されない row には最低1 dimension の perturbation を強制します。

その pool に対して posterior sampling を行い、replacementなしで `q` 点を選択します。

## 10.15 Pending points

Thompson-sampling path は `X_pending` を public input coordinates で受け取ります。

shape は

    [n_pending, input_dim]

で、finite value が要求されます。

local candidate pool と pending point が一致する候補は除外されます。

したがって pending points は trust-region state の一部ではなく、
現在の candidate generation で再選択を避けるための async context です。

## 10.16 Candidate constraints

通常の `optimize()`、`optimize_mixed()`、`optimize_multifidelity()` は
public-space `CandidateConstraints` を各 optimizer path へ渡します。

一方、TuRBO Thompson sampling は有限 candidate pool を point-wise に filter するため、

- intra-point linear constraints: 対応
- intra-point nonlinear constraints: 対応
- inter-point linear constraints: 非対応
- inter-point nonlinear constraints: 非対応

です。

joint q-batch feasibility が必要な inter-point constraint を、独立 pool filtering として
誤って処理しないための制限です。

## 10.17 Mixed-variable TuRBO

continuous、integer、categorical variable を同じ Euclidean trust region で縮小すると、
離散変数へ人工的な局所距離を導入します。

`turbo_mixed_trust_region_bounds()` は

- continuous dimensions: localize
- integer dimensions: global range を保持
- categorical dimensions: global range を保持

とします。

`optimize_mixed()` は legal integer / categorical assignments を fixed-feature combinations として
列挙し、continuous local region と組み合わせて BoTorch mixed optimization を行います。

## 10.18 MultiFidelity TuRBO

fidelity coordinate も design coordinate と同じ意味ではありません。

`turbo_multifidelity_trust_region_bounds()` は

- design dimensions: localize
- fidelity dimensions: global range を保持

とします。

これにより incumbent 周辺の design を探索しつつ、fidelity 自体は global に選択できます。

全 dimension が fidelity の場合は localize する design dimension がないため拒否されます。

## 10.19 Restart

`length < length_min` になると restart が必要です。

restart 前に `optimize()`、mixed / multifidelity optimization、
Thompson sampling を呼ぶと runtime error になります。

`restart_turbo_state()` は

- success / failure counters を reset
- trust-region length を reset
- `restart_triggered=False`
- `restart_count += 1`

とし、global best information は保持します。

## 10.20 Restart center

`TuRBOStrategy.restart()` は明示的な center を受け取るか、
指定がなければ global bounds 内の scrambled Sobol point を新しい center として生成します。

seed を指定すれば再現可能です。

したがって restart は単なる length reset ではなく、

    local state reset + new search center

として扱われます。

## 10.21 Stateful BO loop

典型的な利用順序は次です。

    fit surrogate
      -> build acquisition
      -> strategy.optimize(...) or strategy.thompson_sample(...)
      -> evaluate objective
      -> strategy.update_state(...)
      -> append observations
      -> refit surrogate

restart が trigger された場合は、次の candidate generation 前に

    strategy.restart(...)

を実行します。

search strategy は objective evaluation や surrogate fitting を暗黙には所有しません。

## 10.22 TuRBO と BAxUS

TuRBO は original/public input space の局所 trust region を適応します。

BAxUS は trust-region adaptation に加えて、低次元 target subspace 自体を段階的に拡張します。

| Method | Local region | Search dimension |
| --- | --- | --- |
| TuRBO | adaptive | original dimension |
| BAxUS | adaptive | adaptive target subspace |

したがって BAxUS は単なる TuRBO backend ではなく、
adaptive embedding と trust-region search を組み合わせた高次元探索法です。

## 10.23 実装上の contract

現在の実装と理論を対応させる際の主要 contract は次です。

- trust-region state は surrogate parameter ではない
- `q` は `state.batch_size` と一致する
- state update は completed batch ごとに1回行う
- state update utility は scalar per candidate
- `values` と `state_values` を分離できる
- center は public/original input space に保持する
- trust-region length は global input width に対する相対 scale
- public-space ARD と reduced-space ARD を混同しない
- restart trigger 後は明示的 restart 前に candidate を生成しない
- Mixed では discrete dimensions を continuous trust region から分離する
- MultiFidelity では fidelity dimensions を design trust region から分離する
- TS の pending points は public coordinates
- TS の constraint filtering は inter-point constraints を扱わない

## 10.24 関連章

高次元 embedding と BAxUS は
[High-dimensional Acquisition Search](09_high_dimensional.md) を参照してください。

batch / sequential / one-shot optimization は
[Batch and One-shot Optimization](08_batch_one_shot.md) を参照してください。

制約の一般 contract は [Candidate Constraints](05_constraints.md) を参照してください。

より広い高次元 search strategy の整理は
[High-dimensional Search Strategies](../20_high_dimensional_search.md) を参照してください。

## 参考文献

1. Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. (2019).
   *Scalable Global Optimization via Local Bayesian Optimization*.
   Advances in Neural Information Processing Systems 32.
2. Papenmeier, L., Nardi, L., & Poloczek, M. (2022).
   *Increasing the Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces*.
   Advances in Neural Information Processing Systems 35.
3. Eriksson, D., & Jankowiak, M. (2021).
   *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces*.
   Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence.
4. BoTorch documentation and tutorials for TuRBO and high-dimensional Bayesian optimization.
