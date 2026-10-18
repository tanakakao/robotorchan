# Acquisition Optimization Theory

このディレクトリでは、獲得関数を構築した後に、
candidate をどの座標系・探索領域・numerical optimizer で探索するかを扱います。

獲得関数そのものの理論は
[Acquisition Function Theory](../acquisition/README.md)、
API・runtime contract は
[Optimization guides](../../optimization/README.md)
を参照してください。

## 責務境界

獲得関数最適化では複数の責務を区別し、
次の責務を分離します。

    posterior
      -> objective / posterior transform
      -> acquisition function
      -> search strategy / search geometry
      -> initialization
      -> numerical optimizer backend
      -> candidate

この subtree は後半の search / initialization / numerical optimization を中心に扱います。

特に、

- acquisition が何を最大化するか
- search strategy がどこを探索するか
- optimizer backend がどう数値最適化するか

を混同しないことが重要です。

## 章構成

1. [Foundations](01_foundations.md)
2. [Gradient-based optimization](02_gradient_based.md)
3. [Initialization](03_initialization.md)
4. [Mixed and discrete optimization](04_mixed_discrete.md)
5. [Candidate constraints](05_constraints.md)
6. [Derivative-free optimization](06_derivative_free.md)
7. [Evolutionary and population methods](07_evolutionary.md)
8. [Batch and one-shot optimization](08_batch_one_shot.md)
9. [High-dimensional search](09_high_dimensional.md)
10. [Trust-region optimization](10_trust_region.md)
11. [Hybrid optimization](11_hybrid.md)
12. [Selection guide](12_selection_guide.md)

初めて optimizer を選ぶ場合は
[Selection guide](12_selection_guide.md) から読むと、
問題構造から関連章へ移動できます。

## Search strategy と optimizer backend

高次元探索や Trust Region は、numerical optimizer と同じレイヤーではありません。

    Search strategy
      LatentSpace / REMBO / HeSBO / ALEBO / BAxUS / TuRBO

    Numerical backend
      BoTorch / Torch / Random / Sobol / DE / CMA-ES / GA / PSO

という責務分離があります。

たとえば TuRBO は trust-region state と local search geometry を管理し、
「BoTorchの代わりになる1個の optimizer backend」という位置付けではありません。

同様に REMBO や BAxUS は search coordinates を定義するため、
任意の backend と自動的に合成可能とは限りません。

## Orchestration

Hybrid は search strategy とは別に、複数の optimizer stage を接続します。

現在の基本構成は

    derivative-free global search
      -> explicit batch initial condition
      -> BoTorch local refinement

です。

global / local stage では constraint handling や device semantics が異なる場合があります。

詳細は [Hybrid optimization](11_hybrid.md) を参照してください。

## Batch semantics

`q_batch=True` は「q個の候補を返せる」という意味だけではありません。

joint q acquisition では

    X in R^(q x d)

を1つの decision variable として最適化します。

sequential optimization は別の capability であり、
one-shot acquisition はさらに auxiliary variables を含む augmented optimization を必要とします。

詳細は
[Batch and one-shot optimization](08_batch_one_shot.md)
を参照してください。

## Candidate constraints

candidate constraints は acquisition-level output constraint と異なります。

ここで扱うのは input candidate に対する既知の

- linear inequality / equality
- nonlinear inequality
- inter-point nonlinear

などです。

backend によって native / feasibility-first / penalty など
enforcement mechanism が異なります。

high-dimensional embedding では public-space constraint を
reduced coordinates に黙って読み替えないことも重要です。

詳細は [Candidate constraints](05_constraints.md) を参照してください。

## High-dimensional search

高次元 optimization では surrogate model の高次元対応と
acquisition search の高次元対応を分けます。

    model-side representation
      X -> representation -> surrogate

と

    search-side representation
      Z -> X -> acquisition(model, X)

は異なる責務です。

REMBO、HeSBO、ALEBO、BAxUS は後者の search strategy に位置付けられます。

詳細は
[High-dimensional search](09_high_dimensional.md)
を参照してください。

## Tree surrogate

Random Forest / Extra Trees などでは candidate coordinate に関する
smooth gradient を前提にできません。

`TreeEnsembleSearchStrategy` はこのための独立した最適化原理ではなく、
現在は generic random-search strategy を tree-surrogate 用に明示した specialization です。

したがって理論上は
[Derivative-free optimization](06_derivative_free.md)
に位置付けます。


## 実装情報の参照先

optimizer API、バックエンドの対応状況、制約処理の実装上の契約は
[Optimization guides](../../optimization/README.md) を参照してください。
