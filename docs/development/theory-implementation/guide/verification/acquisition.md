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


## Phase 8の独立参照：Regression / Classification ALの算術

`tests/acquisition/test_phase8_regression_al_formula_reference.py`は、学習の再実行を伴わない固定`SingleTaskGP`、float64、q=1の3候補`[0.15,0.45,0.85]`について、posteriorのmean `mu`とvariance `v`から次を直接算出する。

```text
PosteriorVariance = v
PosteriorStd = sqrt(max(v, 0))
Straddle = beta * PosteriorStd - abs(mu - target)
BoundaryVariance = max(v, 0) * exp(-0.5*((mu-target)/max(PosteriorStd,eps))**2)
target = 0.35, beta = 1.7
```

4つの公開acquisition値はいずれもshape `[3]`、`atol=rtol=1e-12`で参照値と一致する。これはposteriorに与えられた**momentから獲得関数値への算術**の検証であって、posterior moment自体を独立に計算した結果ではない。

`tests/acquisition/test_phase8_classification_al_reference.py`は、`BinarySingleTaskGPClassifier`の0/1ラベル4学習点とq=1の3候補で、固定の`predict_proba`と`latent_posterior`から以下を計算する。

```text
PredictiveEntropy = -sum_c(p_c * log(p_c))   # xlogyでp=0も扱う
MarginUncertainty = 1 - p_largest + p_second_largest
LatentStraddle = beta*sqrt(max(var_latent,0)) - abs(mean_latent)
beta = 1.5
```

3つの公開scoreがshape `[3]`で`atol=rtol=1e-10`に一致し、entropyが`[0,log(2)]`、marginが`[0,1]`内であることをassertする。label uncertaintyとlatent uncertaintyを**異なる量**として扱っており、分類器の学習収束・calibration・multiclass適合性の独立保証ではない。

## Phase 8の独立参照：Gaussian analytic BO

`tests/acquisition/test_phase8_analytic_bo_gaussian_reference.py`は、float64の固定単出力GPから得たposteriorの`mu`、`sigma`に対して、最大化の`z=(mu-best_f)/sigma`を使った次のclosed formを手計算する。

```text
EI = (mu-best_f)*Phi(z) + sigma*phi(z)
LogEI = log(EI)
LogPI = log(Phi(z))
UCB = mu + sqrt(beta)*sigma
best_f = 0, beta = 0.4
```

BoTorch-nativeの`LogExpectedImprovement`、`LogProbabilityOfImprovement`、`UpperConfidenceBound`を3個のq=1候補で比較し、各出力shape `[3]`および`atol=rtol=1e-8`を確認する。**BoTorchへのdelegateされた解析型acquisitionの値**を対象とし、極端なGaussian tailでの安定性、GPの独立posterior計算、q>1やnoisy/MC acquisitionへの保証にはならない。

以上3テストの本文を旧[Verification Mapping](../../verification.md)のPhase 8節に対応付けた。別familyに関する数値テストの存在や未検証の境界を、この3件から推測しない。


## Phase 8独立参照：BALDとProbabilityVarianceの固定sample

`tests/acquisition/test_phase8_bald_probability_variance_reference.py`は、`BinarySingleTaskGPClassifier`の`sample_class_probabilities`をテスト内で固定の64 drawへ差し替え、2候補（q=1）、2クラスのprobability sampleを共有する。`samples`の形状は`[64,2,1,2]`、acquisition scoreはそれぞれ`[2]`。固定4 drawを16回繰り返した数値tensorなので、samplerの乱数系列やモデルの学習品質は検査していない。

```text
p_bar = mean_samples(p_sample)
ProbabilityVariance = mean_classes(mean_samples((p_sample - p_bar)^2))
BALD = max(0, H(p_bar) - mean_samples(H(p_sample)))
H(p) = -sum_classes(xlogy(p,p))
```

ProbabilityVarianceは**不偏標本分散（N-1で除算）ではなく母分散型のsample平均（Nで除算）**を使い、さらにclass軸で平均する。BALDは予測entropyとsampleごとの条件付きentropyとの差を計算し、下限0へclampする。両者とも公開acquisition値と独立に組み立てたsample-moment式を`atol=rtol=1e-10`で比較し、非負性も確認する。

同じ固定sampleを実装側と参照式に渡すため、検証範囲は**集約演算と軸の意味**である。実際のposterior samplingの分布、64 drawの収束、相互情報量の推定誤差、分類器のcalibrationやactive learning性能は確認していない。

旧[Verification Mapping](../../verification.md)のPhase 8 final reconciliationは**代表propertyの検証範囲についてのexit recommendation**であり、qNEI/qLogNEI、multi-objective hypervolume、constrained/heterogeneous composition、情報利得lookaheadの独立数値参照まで認定していない。既存の[未検証事項](limitations.md)と併読する。
