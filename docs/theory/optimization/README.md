# Acquisition Optimization Theory

このディレクトリでは、獲得関数を構築した後に、
candidate をどの座標系・探索領域・numerical optimizer で探索するかを扱います。

獲得関数そのものの理論は
[Acquisition Function Theory](../acquisition/README.md)、
API・runtime contract は
[Optimization guides](../../optimization/README.md)
を参照してください。

## 責務境界

robotorchan では optimization を単一の `optimize()` 処理として扱わず、
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

## 理論 family と現在の実装

| 理論 family | 現在の主な実装 |
| --- | --- |
| Gradient-based | BoTorch backend, Adam, AdamW, SGD |
| Sampling | Random, Sobol |
| Evolutionary / population | DE, CMA-ES, GA, Mixed GA, PSO, NSGA-II |
| Mixed / discrete | BoTorch mixed, Mixed GA, Random/Sobol, DE / PSO の対応範囲 |
| Hybrid | global optimizer -> BoTorch local refinement |
| High-dimensional | LatentSpace, REMBO, HeSBO, ALEBO, BAxUS |
| Trust region | TuRBO |
| Tree-surrogate search | TreeEnsembleSearchStrategy |

この表は family の索引です。
詳細な制約、q、Mixed、fixed-feature、device capability は
[Selection guide](12_selection_guide.md) と runtime-facing docs を正とします。

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

## Named API と specialist API

`robotorchan.optim.optimize_acqf()` の named optimizer registry と、
specialist / direct API の対応範囲は完全には同一ではありません。

例として、

- BoTorch mixed optimization
- Mixed GA
- NSGA-II
- direct Hybrid backend
- high-dimensional search strategies
- TuRBO strategy

には named scalar dispatcher とは異なる入口や contract があります。

したがって registry にないことだけを理由に「未実装」と判断せず、
各 specialist API と capability descriptor を確認します。

## Capability metadata

`OptimizerCapabilities` は documentation label だけでなく、
named dispatcher の runtime validation に使われます。

主な軸は

- continuous / integer / categorical / mixed
- gradient requirement
- linear / nonlinear candidate constraints
- inter-point nonlinear constraints
- q-batch / sequential
- fixed features
- GPU / batch evaluation

です。

constraint support については boolean だけでなく、

- native
- penalty
- feasibility-first
- repair
- projection
- rejection
- unsupported

という enforcement semantics も区別します。

最新の runtime-facing contract は
[Constraint handling metadata](../../optimization/constraint-handling-metadata.md)
を参照してください。

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

## Runtime-facing source of truth

Theory は数式、仮定、探索原理、手法間の関係を説明します。

現在の public API、runtime validation、backend capability は
[Optimization guides](../../optimization/README.md)
および実装を source of truth とします。

特に確認すべき runtime docs は

- [Public optimizer API](../../optimization/public-optimizer-api.md)
- [Optimizer architecture](../../optimization/optimizer-architecture.md)
- [Constraint handling metadata](../../optimization/constraint-handling-metadata.md)
- [Acquisition initialization](../../optimization/initialization.md)
- [Initialization compatibility](../../optimization/initialization-compatibility.md)
- [Optimizer runtime contract](../../optimization/runtime-contract.md)
- [High-dimensional search](../../optimization/high_dimensional_search.md)
- [TuRBO](../../optimization/turbo.md)

です。

## Theory audit status

この optimization subtree は、現在の実装拡張後に以下を横断監査しています。

- 01-12 の全章に参考文献を明記
- gradient-based / derivative-free / population backend を反映
- Mixed / discrete semantics を反映
- candidate constraint と enforcement semantics を反映
- joint q / sequential / one-shot の違いを反映
- high-dimensional search と model-side reduction を分離
- TuRBO の state / restart / structured-dimension contract を反映
- Hybrid の stage-wise contract を反映
- named dispatcher と specialist API の差を反映
- runtime-facing docs と capability metadata の差分を監査

新しい optimizer が追加された場合、class 名ごとに章を増やすのではなく、
探索原理が既存 family に属するなら対応する章へ追加します。

一方、新しい search geometry や orchestration が既存 family では説明できない場合は、
独立章として分離します。
