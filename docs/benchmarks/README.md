# Benchmarks

robotorchan のモデル・探索戦略を、再現可能な条件で比較するためのベンチマーク文書です。

## Expressive surrogate models

- [Predictive benchmark](expressive_predictive.md)

## High-dimensional Bayesian optimization

- [Sequential BO](high_dimensional/sequential_bo.md)
- [Batch BO](high_dimensional/batch_bo.md)

ベンチマークは利用方法や理論説明の代替ではありません。モデル選択は [Model guides](../models/README.md)、探索戦略は [Optimization guides](../optimization/README.md) を参照してください。

## 文書の責務

`docs/benchmarks/` は実験条件と結果の解釈契約を置く場所です。モデル理論、APIリファレンス、開発時の Phase 記録はここへ混在させません。benchmark script の一時的な実装メモではなく、再現に必要な恒久条件を記録します。
