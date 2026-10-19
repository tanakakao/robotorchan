# Optimization guides

surrogate model とは独立した candidate search / optimization の利用者向けガイドです。

- [Unified API workflow](unified-api-workflow.md): GP学習からBoTorch獲得関数・Optimizerまでの実行可能な接続例
- [Standard acquisition functions](standard_acquisition.md): single-objective BO の推奨獲得関数とBoTorch直接利用方針
- [Thompson sampling](thompson_sampling.md): finite candidate set向けposterior samplingと連続探索との責務分離
- [Regression active learning](regression_active_learning.md): posterior uncertaintyとintegrated variance reduction
- [Level-set and boundary learning](level_set_learning.md): Straddleと境界推定
- [Information-theoretic acquisitions](information_theoretic.md): MES、GIBBON、Randomized Straddle
- [Lookahead acquisitions](lookahead.md): Knowledge Gradientとmulti-step lookahead
- [Multi-objective acquisitions](multiobjective.md): qLogEHVI、qLogNEHVI、qLogNParEGO
- [Multi-fidelity and cost-aware acquisitions](multifidelity-cost-aware.md): MF-KGとcost-aware utility
- [Expected Predictive Information Gain](epig.md): 予測指向の回帰Active Learning
- [BoundaryVariance](boundary-variance.md): robotorchan固有の境界探索ヒューリスティック
- [Acquisition integration](acquisition-integration.md): 対応範囲と制約
- [Mixed one-shot optimization](mixed-one-shot-optimization.md): qKG / qMFKG のmixed候補最適化境界
- [High-dimensional search](high_dimensional_search.md): Original space、Random Search、latent search、REMBO、HeSBO、ALEBO、TuRBO、BAxUS
- [TuRBO](turbo.md): stateful trust-region strategy の詳細
- [Optimizer architecture](optimizer-architecture.md): search strategy、backend、dispatch の責務境界
- [Public optimizer API](public-optimizer-api.md): named backend と structured variable space の利用方法
- [Constraint handling metadata](constraint-handling-metadata.md): backend ごとの制約処理 semantics
- [Nonlinear candidate constraints](nonlinear-candidate-constraints.md): X-space nonlinear制約、intra/inter、初期条件、対応範囲
- [Acquisition initialization](initialization.md): restart、q、constraints、one-shot、
  embedded search の初期条件契約
- [Initialization compatibility](initialization-compatibility.md): Acquisition × Optimizer ×
  Initialization の最終互換表

高次元 surrogate の選択は [High-dimensional models](../models/high_dimensional.md) を参照してください。

## Heterogeneous end-to-end validation

回帰・分類・制約・多目的を組み合わせた検証資料は、用途別に次の文書を参照してください。

- **問題と意味付け**: [Synthetic benchmark](../benchmarks/heterogeneous-e2e-benchmark.md)、[Semantic and acquisition composition validation](heterogeneous-e2e-acquisition-composition.md)
- **単目的・回帰**: [Single-objective validation](heterogeneous-e2e-acquisition-composition.md)、[Regression](../development/heterogeneous-e2e-regression.md)、[Regression variations](../development/heterogeneous-e2e-regression-variations.md)
- **分類・制約**: [Classification](../development/heterogeneous-e2e-classification.md)、[Classification and mixed outcome constraints](heterogeneous-e2e-classification-constrained.md)
- **多目的・確率目的**: [Multi-objective and classification-constrained multi-objective](heterogeneous-e2e-multiobjective.md)、[Probability objective benchmark and E2E validation](../benchmarks/probability_objectives.md)
- **実行・基準手法**: [Baselines](../benchmarks/heterogeneous-e2e-baselines.md)、[Closed-loop](../benchmarks/heterogeneous-e2e-closed-loop.md)、[Multi-seed](../benchmarks/heterogeneous-e2e-multiseed.md)、[Runtime compatibility](../development/heterogeneous-e2e-runtime-compatibility.md)
- **検証と制限**: [Edge failures](../development/heterogeneous-e2e-edge-failures.md)、[Verification](../development/heterogeneous-e2e-verification.md)、[Benchmark report generator](../benchmarks/heterogeneous-e2e-benchmark-report.md)

これらは機能・契約・検証範囲の記録です。レポート生成コードや限定的なテストの存在を、包括的な性能実証またはリリース認定と解釈しないでください。比較実験の共通条件は [Benchmark guides](../benchmarks/README.md) を参照してください。

## Theory / status navigation

獲得関数の数式・仮定・手法間の関係は [Acquisition Function Theory](../theory/acquisition/README.md) に分離しています。問題設定から手法を選ぶ場合は [Acquisition Selection Guide](../theory/acquisition/12_selection_guide.md) を入口にしてください。

現在の robotorchan / BoTorch integration と明示的な制約は [Acquisition integration](acquisition-integration.md) を正とします。Theory に掲載されていること自体は robotorchan 固有実装を意味しません。
