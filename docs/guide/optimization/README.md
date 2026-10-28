# 候補最適化の実装ガイド

数式と仮定は [Theory](../../theory/optimization/README.md)、使い方は [Optimization](../../optimization/README.md) を参照してください。各章は対応する理論、実装、近似と検証範囲を説明します。

- [Inner optimization：獲得関数最適化の実装ガイド](01_foundations.md)
- [Gradient-based：獲得関数最適化の実装ガイド](02_gradient_based.md)
- [Initialization：獲得関数最適化の実装ガイド](03_initialization.md)
- [Mixed / Discrete：獲得関数最適化の実装ガイド](04_mixed_discrete.md)
- [Candidate constraints：座標・数理契約・backendの制限](05_constraints.md)
- [Derivative-free / Sampling：獲得関数最適化の実装ガイド](06_derivative_free.md)
- [Evolutionary Optimization：スカラー獲得関数と多目的探索を区別する](07_evolutionary.md)
- [Batch / One-shot：実候補とfantasy変数を分離する](08_batch_one_shot.md)
- [High-dimensional search：モデル座標と候補探索座標を分離する](09_high_dimensional.md)
- [TuRBO Trust Region：状態遷移と局所探索](10_trust_region.md)
- [Hybrid optimization：獲得関数最適化の実装ガイド](11_hybrid.md)
- [Optimizer selection：獲得関数最適化の実装ガイド](12_selection_guide.md)
