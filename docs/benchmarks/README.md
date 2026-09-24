# Benchmarks

robotorchan のモデル・探索戦略を、再現可能な条件で比較するためのベンチマーク文書です。

## Expressive surrogate models

- [Predictive benchmark](expressive_predictive.md)

## Non-GP surrogates

`benchmarks/non_gp_surrogates.py` provides a deterministic predictive diagnostic for the
implemented empirical non-GP surrogates. It is an executable benchmark rather than a permanent
performance ranking: raw ensemble spread is not treated as calibrated uncertainty without a
separate calibration study.

## High-dimensional Bayesian optimization

- [Sequential BO](high_dimensional/sequential_bo.md)
- [Batch BO](high_dimensional/batch_bo.md)

ベンチマークは利用方法や理論説明の代替ではありません。モデル選択は [Model guides](../models/README.md)、探索戦略は [Optimization guides](../optimization/README.md) を参照してください。

## 文書の責務

`docs/benchmarks/` は実験条件と結果の解釈契約を置く場所です。モデル理論、APIリファレンス、開発時の Phase 記録はここへ混在させません。benchmark script の一時的な実装メモではなく、再現に必要な恒久条件を記録します。

## Capability benchmark

`robotorchan.benchmarks.capability` is a library-level structural benchmark for registry filtering
and recommendation consistency. It is intentionally separate from predictive or BO-performance
benchmarks: counts of compatible models or recommendations are not measures of model quality.
