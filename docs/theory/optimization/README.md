# Acquisition Optimization Theory

このディレクトリでは、獲得関数を構築した後に、
候補をどのように探索するかを扱います。

獲得関数そのものの理論は
[Acquisition Function Theory](../acquisition/README.md)、
API・runtime contract は
[Optimization guides](../../optimization/README.md)
を参照してください。

## 責務境界

```text
posterior
 -> objective / posterior transform
 -> acquisition function
 -> initialization
 -> acquisition optimization
 -> candidate
```

ここでは最後の2段階を中心に扱います。

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

## 現在の実装との対応

| 理論 family | 現在の主な実装 |
| --- | --- |
| Gradient-based | BoTorch backend, Adam, AdamW, SGD |
| Sampling | random, Sobol |
| Evolutionary / population | DE, CMA-ES, GA, Mixed GA, PSO, NSGA-II |
| Mixed / discrete | BoTorch mixed, Mixed GA, sampling, DE / PSO の対応範囲 |
| Hybrid | hybrid backend |
| High-dimensional | REMBO, HeSBO, ALEBO, BAxUS |
| Trust region | TuRBO |
| Specialized search | TreeEnsembleSearchStrategy |

新しい optimizer が追加された場合、class 名ごとに章を増やすのではなく、
探索原理が既存 family に属するなら対応する章へ追加します。
