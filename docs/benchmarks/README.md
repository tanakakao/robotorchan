# Benchmarks

robotorchan のモデル・探索戦略を、再現可能な条件で比較するためのベンチマーク文書です。

## Comparative optimization benchmark navigation

比較実験を再現・解釈する際は、次の順に確認してください。

1. **問題定義**: [Problem contract](problem-contract.md)、[Problem registry](problem-registry.md)、[Standard problems](standard-problems.md)、[Heterogeneous problems](heterogeneous-problems.md)
2. **実行条件**: [Experiment configuration](experiment-configuration.md)、[Execution profiles](execution_profiles.md)、[Environment provenance](environment_provenance.md)、[Fairness protocol](comparative-fairness-protocol.md)
3. **評価・集計**: [Metrics](metrics.md)、[Multi-seed statistics](multiseed_statistics.md)、[Comparative experiment matrix](comparative-experiment-matrix.md)、[Empirical evaluation protocol](empirical-evaluation-protocol.md)
4. **結果と証拠**: [Reporting](reporting.md)、[Empirical measured report](empirical-measured-report.md)、[Empirical evaluation status](empirical-evaluation.md)

実行可能なベンチマークやレポート生成手順が存在しても、実測比較が完了したことや最適化性能の優位性が認定されたことは意味しません。各レポートの実行状態・評価条件・証拠を確認してください。

## Heterogeneous end-to-end benchmark

- [Synthetic problem and reference feasibility](heterogeneous-e2e-benchmark.md)
- [Reproducible benchmark report generator](heterogeneous-e2e-benchmark-report.md)
- [Random, Sobol, and qEI baseline evaluation](heterogeneous-e2e-baselines.md)
- [Sequential closed-loop smoke evaluation](heterogeneous-e2e-closed-loop.md)
- [Paired multi-seed statistical evaluation](heterogeneous-e2e-multiseed.md)

These documents define a synthetic problem and an executable report protocol; neither constitutes a certified measured performance result.

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
