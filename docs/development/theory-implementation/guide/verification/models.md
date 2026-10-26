# モデル：数値参照をproperty別に追う

モデル検証の正本：[Verification Mapping](../../verification.md)のモデル検証行列・final exit reconciliation。数式を知る入口は[GP](../02_gaussian_process.md)、[Mixed](../05_mixed_variables.md)、[MultiTask](../07_multitask_multioutput.md)、[Reduction](../18_dimensionality_reduction_gp.md)、[Noise](../15_heteroskedastic_noise.md)、[Classification](../24_classification.md)。

| 検証対象 | 独立参照テスト | 検証されるproperty | 検証されない範囲 |
| --- | --- | --- | --- |
| Kronecker block-design prior | `tests/models/standard/test_phase7_kronecker_covariance_reference.py` | 固定float64の`K_input⊗K_task`、対称性・PSD・task置換。atol=rtol=1e-10 | posterior conditioning、学習、全variant |
| Data-fidelity kernel | `tests/models/standard/test_phase7_multifidelity_kernel_reference.py` | offset boundary、残差rank-one、PSD。atol=rtol=1e-10 | full MF posterior、linear-truncated |
| PCA reducer | `tests/models/high_dimensional/reduced/test_phase7_pca_projection_reference.py` | 固定データのeigenspectrum、orthogonality、再構成 | PLS・neural・予測精度 |
| Replicate mean noise | `tests/models/robust/test_phase7_replicate_noise_reference.py` | 不偏標本分散`s²`とmean variance`s²/n` | 異分散posterior較正 |
| Binary probit | `tests/models/classification/test_phase7_binary_probit_reference.py` | `E[Phi(f)] = Phi(mu/sqrt(1+sigma²))`と確率正規化 | classifier fitting、calibration |
| Gaussian adapter | `tests/models/non_gp/test_phase7_gaussian_posterior_reference.py` | affine reparameterization、batch/sample軸、zero varianceとmean gradient | 外部estimatorの予測精度 |
| PCA public/latent coordinates | `tests/models/high_dimensional/reduced/test_phase7_posterior_coordinate_reference.py` | 固定reducer経路のposterior mean/variance同値 | reducerの学習品質、他reducer |
| Mixed-PCA categorical column | `tests/models/high_dimensional/reduced/test_phase7_mixed_reduced_categorical_reference.py` | 元カテゴリID保存とoriginal/reduced posterior同値 | すべてのMixed reducer |
| Heteroskedastic noise | `tests/models/robust/test_phase7_heteroskedastic_uncertainty_reference.py` | 制御されたlog-noise、latentとの分離、手動の`Var[y]=Var[f]+v` | `posterior(observation_noise=True)`の場所依存noise注入 |
| Forest adapter | `tests/models/non_gp/test_phase7_forest_estimator_reference.py` | sklearn各treeの予測とBoTorch ensemble momentsの一致 | epistemic calibration、他ensemble |

この表は原文に明記されたpropertyを分類したものであり、今回新しい数値テストを実行したという意味ではない。Exact GP conditioning referenceやcategorical relabelingの参照は原文に記述があるが、ここでは確認可能なテストパスのない行を推測で補っていない。

**明示的に未検証の境界：** Heteroskedastic wrapperの候補点ごとの`posterior(observation_noise=True)`が正しい観測分散を自動合成する性質は保証されない。手動の恒等式の検証と、公開posterior実装の検証を分ける。[未検証事項](limitations.md)へ。
