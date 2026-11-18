# Benchmarks

robotorchan のモデル・探索戦略を、再現可能な条件で比較するためのベンチマーク文書です。

初めて利用する場合は [Benchmark Quickstart](quickstart.md) から、実行・保存・実測レポートの再生成・結果解釈まで順に確認できます。

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

## Causal BO: known-DAG intervention benchmark

`run_causal_benchmark` evaluates a deterministic four-node SCM
`X → M → Y ← Z`, where `M = 0.25 + 0.75 X` and

```text
Y = -(M - 0.8)^2 - 0.5 (Z - 0.65)^2
```

The **known optimum** is `Y=0` for the joint hard intervention
`do(X=0.73333..., Z=0.65)`. The user supplies the DAG and structural
equations; no causal structure inference or observational causal
identification is performed.

Strategies:

- **CBO**: sequential `CausalBOStudy.ask/tell`, choosing `{X}`, `{Z}`
  or `{X,Z}`, using native BoTorch LogEI and optional per-set cost ranking
- **GP-BO (fixed-set)**: a regular `SingleTaskGP` with the same LogEI and
  `optimize_acqf`, choosing continuous values only for `do(X,Z)`
- **Random Search**: uniform choice among the same three intervention sets,
  then uniformly random continuous values

For a given seed, every strategy receives **exactly the same initial
evaluations**, including `n_initial_per_set` experiments from each set.
The fixed-set GP trains **only on full-intervention observations**, not
the partial-intervention data, because these are different mechanisms.
CBO trains one GP per set. Every strategy receives the same number of
additional evaluations and same acquisition optimizer configuration for
the two GP-based methods; cost spent is **not** constrained to be equal.

The default per-evaluation costs are `c({X})=1`, `c({Z})=1.5`, and
`c({X,Z})=3`. Report the best true intervention outcome, nonnegative
simple regret against `Y=0`, and cumulative intervention cost after
**every** evaluation, including the common initialization.
Comparison by number of evaluations does *not* imply equal-cost comparison.
Likewise, the random strategy has access to different intervention sets
than the fixed-set GP-BO baseline; do not describe this as an isolated
test of causal reasoning.

Run manually (the full multi-seed performance experiment is **not**
executed as part of CI):

```bash
python benchmarks/causal_bo.py --seeds 0,1,2 --n-initial-per-set 2 --iterations 5
```

The script emits per-evaluation CSV records and a JSON file of
seed-aggregated best-value, simple-regret and cumulative-cost curves
using the existing `summarize_curves` utility. Fixed seeds isolate
strategy-side PyTorch RNG; the deterministic SCM has no observation
noise. Tests verify mechanics and reproducibility, but **do not**
establish statistical superiority of CBO; use additional scenarios,
seeds and cost budgets before making performance claims.
