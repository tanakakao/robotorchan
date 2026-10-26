# Acquisition：数式参照とposterior依存を分ける

数学的定義は[Acquisition理論](../../../../theory/acquisition/README.md)、実装の対象・委譲は[Acquisitionガイド](../acquisition/01_foundations.md)、独立検証の詳細・tol・テスト追加履歴は[Verification Mapping](../../verification.md)が正本。

## 独立参照が記録された主なグループ

| 対象 | テスト | 何を比較しているか |
| --- | --- | --- |
| 回帰AL | `tests/acquisition/test_phase8_regression_al_formula_reference.py` | PosteriorVariance、PosteriorStd、Straddle、BoundaryVarianceの固定float64 posterior moment式（atol/rtol=1e-12） |
| 分類AL | `tests/acquisition/test_phase8_classification_al_reference.py` | PredictiveEntropy、MarginUncertainty、LatentStraddleの固定class probability/latent式（1e-10） |
| BALD / ProbabilityVariance | `tests/acquisition/test_phase8_bald_probability_variance_reference.py` | 固定された64サンプルからのmutual informationとpopulation variance（1e-10） |
| Gaussian analytic BO | `tests/acquisition/test_phase8_analytic_bo_gaussian_reference.py` | LogEI、LogPI、UCBとGaussian closed formの照合（float64、q=1、atol=rtol=1e-8） |
| MC qSimpleRegret | `tests/acquisition/test_phase8_mc_simple_regret_reference.py` | q=2を各sample内でmaxし64 sampleで平均。2 batch、float64、seed=17、atol=rtol=1e-8 |
| MC qUCB | `tests/acquisition/test_phase8_mc_qucb_reference.py` | MC sample平均と絶対偏差を用いたUCB式のsample-wise q=2最大化・64 sample平均。2 batch、float64、beta=0.3、seed=23、atol=rtol=1e-8 |

最初の三参照は**獲得関数の算術**を検証し、GP posteriorの統計的正しさやclassifierの校正を証明しない。MCの固定sample momentの一致は、有限MC estimatorの収束保証ではない。

## Validation boundary

情報理論型、lookahead、EHVI/NEHVI、constrained/heterogeneous composition、cost-awareは、適用条件や既存の検証がfamilyごとに異なる。個別に[旧Acquisition Mapping](../../acquisition.md)と[Verification Mapping](../../verification.md)の該当節を確認する。上表にないfamilyを`Unverified`と断定するのではなく、**この索引では詳細照合未了**と扱う。

## MC参照の計算順序と証拠境界

二つのMCテストはBoTorch-native獲得関数の**sample aggregation**を検証しており、GP自体のposteriorを独立に計算したテストではない。

- **qSimpleRegret**: candidate batch `X`のposterior sampleを`f^(s)(X)`とすると、`mean_s max_{i=1..q} f_i^(s)(X)`を参照式とする。sampleに対してq軸を先に最大化し、その後にsample軸を平均する。テストは`SobolQMCNormalSampler(sample_shape=[64], seed=17)`で、`[64, 2, 2]`の実sample、`[2]`のacquisition出力を確認する。
- **qUCB**: `mean_i = mean_s f_i^(s)`をMC sampleごとでなく**sample軸平均**として求め、各sampleと候補に`u_i^(s) = mean_i + sqrt(beta*pi/2)*abs(f_i^(s)-mean_i)`を適用する。その後`mean_s max_i u_i^(s)`を求める。テストの`beta=0.3`、`seed=23`で、同じshapeと`1e-8`の比較を行う。

両テストは、独立の乱数系列を単純比較するのではなく、**acquisition側と参照式で同じSobol samplerのbase samplesを再利用**してsamplingノイズを除外する。このため検証できるのはMCの演算順序・集約の一致であり、独立サンプリングの分布一致、有限sampleからの収束、posterior較正、最適化候補の性能までは確証しない。

本節の数値は[Verification Mapping](../../verification.md)のPhase 8 MC記述と、各テストのソース本文を直接照合した。原文にある他の情報理論型・制約付き・多目的MC familyが検証済みだとは、この二件から推定しない。
