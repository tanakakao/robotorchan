# Acquisition：数式参照とposterior依存を分ける

数学的定義は[Acquisition理論](../../../../theory/acquisition/README.md)、実装の対象・委譲は[Acquisitionガイド](../acquisition/01_foundations.md)、独立検証の詳細・tol・テスト追加履歴は[Verification Mapping](../../verification.md)が正本。

## 独立参照が記録された主なグループ

| 対象 | テスト | 何を比較しているか |
| --- | --- | --- |
| 回帰AL | `tests/acquisition/test_phase8_regression_al_formula_reference.py` | PosteriorVariance、PosteriorStd、Straddle、BoundaryVarianceの固定float64 posterior moment式（atol/rtol=1e-12） |
| 分類AL | `tests/acquisition/test_phase8_classification_al_reference.py` | PredictiveEntropy、MarginUncertainty、LatentStraddleの固定class probability/latent式（1e-10） |
| BALD / ProbabilityVariance | `tests/acquisition/test_phase8_bald_probability_variance_reference.py` | 固定された64サンプルからのmutual informationとpopulation variance（1e-10） |
| Gaussian analytic BO | `tests/acquisition/test_phase8_analytic_bo_gaussian_reference.py` | LogEI、LogPI、UCBとGaussian closed formの照合 |

最初の三参照は**獲得関数の算術**を検証し、GP posteriorの統計的正しさやclassifierの校正を証明しない。MCの固定sample momentの一致は、有限MC estimatorの収束保証ではない。

## Validation boundary

情報理論型、lookahead、EHVI/NEHVI、constrained/heterogeneous composition、cost-awareは、適用条件や既存の検証がfamilyごとに異なる。個別に[旧Acquisition Mapping](../../acquisition.md)と[Verification Mapping](../../verification.md)の該当節を確認する。上表にないfamilyを`Unverified`と断定するのではなく、**この索引では詳細照合未了**と扱う。
