# Benchmarks

robotorchan のモデル・探索戦略を、再現可能な条件で比較するためのベンチマーク文書です。

## Comparative optimization benchmark navigation

比較実験を再現・解釈する際は、次の順に確認してください。

1. **問題定義**: [Problem contract](../../docs-internal/contracts/benchmarks/problem-contract.md)、[Problem registry](problem-registry.md)、[Standard problems](single_objective.md)、[Heterogeneous problems](heterogeneous-problems.md)
2. **実行条件**: [Experiment configuration](experiment-configuration.md)、[Execution profiles](execution_profiles.md)、[Environment provenance](environment_provenance.md)、[Fairness protocol](../../docs-internal/contracts/benchmarks/comparative-fairness-protocol.md)
3. **評価・集計**: [Metrics](metrics.md)、[Multi-seed statistics](multiseed_statistics.md)、[Comparative experiment matrix](comparative-experiment-matrix.md)、[Empirical evaluation protocol](empirical-evaluation-protocol.md)
4. **結果と証拠**: [Reporting](reporting.md)、[Empirical measured report](../../docs-internal/contracts/benchmarks/empirical-measured-report.md)、[Empirical evaluation status](../../.github/project-records/benchmarks/empirical-evaluation.md)

実行可能なベンチマークやレポート生成手順が存在しても、実測比較が完了したことや最適化性能の優位性が認定されたことは意味しません。各レポートの実行状態・評価条件・証拠を確認してください。

## Heterogeneous end-to-end benchmark

- [Synthetic problem and reference feasibility](heterogeneous-e2e-benchmark.md)
- [Reproducible benchmark report generator](../../.github/project-records/benchmarks/heterogeneous-e2e-benchmark-report.md)
- [Sequential closed-loop smoke evaluation](heterogeneous-e2e-closed-loop.md)
- [Random, Sobol, qEI baselines and paired multi-seed evaluation](heterogeneous-e2e-multiseed.md)

合成問題と実行可能な報告手順を定めていますが、それだけで認定済みの実測性能結果になるわけではありません。

## Historical benchmark audit records

日付付き監査は履歴として保存し、現行実装・release 状態・実測性能の根拠とは区別します。Phase 番号は異なる開発サイクルに属し、一続きの roadmap ではありません。欠落機能、報告形式、CI、将来計画の記述は後の変更で古くなり得るため、最新の契約・実装・実測 artifact を確認してください。

- [Comparative capability audit (2026-10-09)](../../.github/project-records/benchmarks/comparative-capability-matrix.md)
- [Comparative benchmark validation and release criteria](../../.github/project-records/benchmarks/comparative-benchmark-validation.md)
- [Benchmark Phase 30 release-readiness review](../../.github/project-records/benchmarks/release_readiness.md)
- [Empirical performance coverage and execution gates (2026-10-11)](../../.github/project-records/benchmarks/empirical-evaluation.md)

## Expressive surrogate models

- [Predictive benchmark](expressive_predictive.md)

## Non-GP surrogates

`benchmarks/non_gp_surrogates.py` は経験 non-GP surrogate の決定的な予測診断です。実行可能な benchmark であり恒久的な性能順位ではありません。較正実験なしに生の ensemble spread を較正済み不確実性とみなしません。

## High-dimensional Bayesian optimization

- [Sequential BO](high_dimensional/sequential_bo.md)
- [Batch BO](high_dimensional/batch_bo.md)

ベンチマークは利用方法や理論説明の代替ではありません。モデル選択は [Model guides](../models/README.md)、探索戦略は [Optimization guides](../optimization/README.md) を参照してください。

## 文書の責務

`docs/benchmarks/` は実験条件と結果の解釈契約を置く場所です。モデル理論、APIリファレンス、開発時の Phase 記録はここへ混在させません。benchmark script の一時的な実装メモではなく、再現に必要な恒久条件を記録します。

## Capability benchmark

`robotorchan.benchmarks.capability` は registry 絞り込みと推奨整合性の構造的 benchmark です。予測・BO 性能とは独立で、互換モデル数・推奨数は品質指標ではありません。