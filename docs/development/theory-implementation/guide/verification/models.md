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


## ソースassertionとの直接照合：Kronecker block-design

`tests/models/standard/test_phase7_kronecker_covariance_reference.py`は3入力・2タスクの`KroneckerMultiTaskGP`から**学習前のprior covariance**を取り、入力優先のflatten順で全36要素を確認する。添字`(i,a)`と`(j,b)`に対し、

```text
K[i*m + a, j*m + b] = K_input[i,j] * K_task[a,b]
(n, m) = (3, 2)
```

を4重ループのscalar referenceで構築し、`torch.testing.assert_close`の`atol=rtol=1e-10`で検査する。さらに対称性、最小固有値`>=-1e-10`、タスクの入替えに対する対応する行列の並べ替えを検査する。乱数seedの固定はなく、テストが確認するのは**prior covarianceの構造と順序**だけである。posteriorの条件付き共分散、noise、最適化されたhyperparameter、Kronecker型の全variantの正しさはここからは言えない。

## ソースassertionとの直接照合：PCA reducer

`tests/models/high_dimensional/reduced/test_phase7_pca_projection_reference.py`は6サンプル・3特徴の固定float64データに`PCAInputReducer(n_components=2)`を当て、独立に`cov = centered.T @ centered / (n-1)`から固有値を求める。

検証するのは、(1) projectedの`[6,2]`形状、(2) retained basisの直交性`components.T @ components = I`、(3) `projected = centered @ components`、(4)説明分散と上位固有値、(5)射影の標本共分散が対応する対角固有値に一致すること、(6)復元後の残差がretained basisへ直交することである。独立したcovariance参照には`atol=rtol=1e-10`を指定し、**主成分の符号そのものは固定しない**。

これらは入力削減の線形代数であり、PCAを内部に持つGPの`posterior(X)`、分類特徴の保持、PLSやAE/VAEの数値契約の独立検証ではない。対応するモデル・reducerの境界は[Reductionの実装ガイド](../18_dimensionality_reduction_gp.md)で扱う。

以上二件は旧[Verification Mapping](../../verification.md)のPhase 7 reference記録を**実際のテストassertionと再照合した範囲**である。表に挙げた他のfamilyについて同じ程度のassertion-level照合が済んだとは扱わない。


## ソースassertionとの直接照合：Multi-fidelity downsampling kernel

`tests/models/standard/test_phase7_multifidelity_kernel_reference.py`は、設計値・カテゴリ値・fidelity値を連結した4行3列の固定float64入力で`MixedSingleTaskMultiFidelityGP`を作る。カテゴリ軸は`cat_dims=[1]`、fidelity軸は`data_fidelities=[2]`である。モデルの`covar_module.kernels[1]`から取り出すdownsampling kernelに対して、次の形の構造を確認する。

```text
K_f(s,t) = offset + c * (1-s)^p * (1-t)^p
s=1 または t=1 の境界では K_f = offset
K_residual = K_f - offset
```

最後のfidelityが`1.0`であるため、最終行・最終列の全値が`offset=K_f(1,1)`となることを`atol=1e-10, rtol=0`で検査する。残差のrank-one構造は`K_residual[:,0:1] @ K_residual[0:1,:] / K_residual[0,0]`との一致で確かめ、対称性・半正定値性（最小固有値`>=-1e-10`）も検証する。行列shapeは`[4,4]`、dtypeはfloat64、行列比較は原則`atol=rtol=1e-10`である。

**検証対象はfidelity-only kernel成分**であり、Mixed GPの全covariance積、条件付きposterior、他のfidelity kernel変種やhyperparameter fittingの正しさを示すものではない。

## ソースassertionとの直接照合：Heteroskedastic uncertainty

`tests/models/robust/test_phase7_heteroskedastic_uncertainty_reference.py`は`HeteroskedasticSingleTaskGP`に固定log-noiseのtest doubleを接続し、入力`x=0.2`で`0.01`、`x=0.8`で`0.09`の**観測雑音分散**が`predicted_noise(X)`から返ることを検査する。固定されたlog-varianceを指数変換した値との比較は`atol=rtol=1e-12`である。

`posterior(X, observation_noise=False).variance`は潜在関数の不確実性であり、入力依存の観測雑音分散と同じではないことを確かめる。さらに、独立な零平均観測誤差を仮定した場合の手動の恒等式

```text
Var[y(x) | D] = Var[f(x) | D] + v(x)
```

を固定2点の候補で数値検査する。ここでは**手動合成した分散**を比較するだけで、`posterior(X, observation_noise=True)`が位置依存雑音を自動的に反映することをテストしていない。雑音GP自体の学習、雑音較正、未知地点での不確実性の精度も検証範囲外である。

この2件は旧[Verification Mapping](../../verification.md)に記載された性質のうち、実際のテスト本文を再照合した範囲だけを説明している。


## 実装assertionの照合：Binary probit classification

`tests/models/classification/test_phase7_binary_probit_reference.py`は`BinarySingleTaskGPClassifier`の0/1ラベル（8学習点）を用い、3候補点`[0.15,0.5,0.85]`について`latent_posterior`と`predict_proba`を分けて取得する。潜在分布をGaussianとするprobitリンクの恒等式

```text
f ~ Normal(mu, sigma²)
P(y=1 | x, D) = E[Phi(f)] = Phi(mu / sqrt(1 + sigma²))
```

を`torch.erfc`から独立に構築し、`probabilities[:,1]`と`atol=rtol=1e-10`で比較する。加えて`[3,2]`の確率shape、陰性確率が`1-p`であること、行和が1、全確率が`[0,1]`内に収まることをassertする。これは**予測確率変換の恒等式**の検査であり、分類モデルのELBO最適化、予測確率の較正、データ上の分類精度を証明しない。

## 実装assertionの照合：Gaussian non-GP adapter

`tests/models/non_gp/test_phase7_gaussian_posterior_reference.py`は`GaussianDistributionPosterior`を固定float64のmean/varianceとbase samplesから構成する。BoTorchと接続するサンプル契約

```text
sample = mean + sqrt(variance) * base_sample
```

を`rsample_from_base_samples(sample_shape=[2],...)`と比較し、`atol=rtol=1e-12`で確認する。期待サンプルshapeは`[2,2,2,1]`。ゼロ分散成分はbase sampleに依らずmeanへ一致し、2サンプルの和から逆伝播した`mean.grad`が全成分で2になることも確かめる。これは**宣言したGaussian posterior adapterのaffine samplingとgradientの契約**であり、外部non-GP estimator自体の予測精度・uncertainty較正や全adapterの適合性を保証しない。

以上2件は実際のテスト本文を照合した結果であり、この更新で新たな独立数値実験を実行したという意味ではない。


## 実装assertionの照合：PCA reduced GPの公開座標と内部座標

`tests/models/high_dimensional/reduced/test_phase7_posterior_coordinate_reference.py`は、float64の6×4学習データで`PCAGP(n_components=2)`を構築する。候補は`[batch=2,q=2,d=4]`で、`input_reducer.transform(candidates)`による2次元表現を明示的に得る。`model.posterior(candidates)`と`model.posterior(latent)`の平均・分散をそれぞれ`atol=rtol=1e-10`で比較し、両者のshapeが`[2,2,1]`であることを確認する。さらに`components`と`mean`がposterior呼び出しによって変化しないことを検証する。**固定済みreducerを経由した座標契約**の確認であり、PCA学習の選択やGPの推定精度そのものの検証ではない。

## 実装assertionの照合：Mixed-PCAでカテゴリ列を保持

`tests/models/high_dimensional/reduced/test_phase7_mixed_reduced_categorical_reference.py`は3連続変数と1カテゴリ列からなる6×4学習データを使い、`MixedPCAGP(n_components=2, cat_dims=[1])`を構築する。縮約後のカテゴリ座標が`reduced_cat_dims == [2]`であること、縮約後のカテゴリ値が元の`candidates[...,1]`と一致すること、内部学習入力でも元カテゴリIDを保持することを検査する。

2 batch×2候補のfloat64入力について、元空間の`posterior(candidates)`と縮約空間の`posterior(reduced)`の平均・分散を`atol=rtol=1e-10`で照合する。カテゴリIDを**連続成分としてPCAで圧縮しない**という対象構成の実装契約を確認するテストであり、全てのカテゴリkernelや他の次元削減手法に一般化しない。

## 実装assertionの照合：Random Forestのensemble posterior

`tests/models/non_gp/test_phase7_forest_estimator_reference.py`は`sklearn`が利用可能な環境で、float64の12学習点から`RandomForestSurrogate(n_estimators=7,random_state=17)`を学習する。候補shapeは`[2,2,1]`。7個のtreeそれぞれの`predict`を独立に集め、`posterior.values`、`posterior.mean`、`posterior.variance`を参照する。

```text
mean = mean(tree_predictions)
variance = var(tree_predictions, unbiased=True)
```

3つの比較はいずれも`atol=rtol=1e-12`を採用し、`posterior(observation_noise=True)`が`NotImplementedError`を返すことも明示的にassertしている。これは**7つのtree予測をensemble momentsとして表現する契約**であり、epistemic uncertaintyの較正やforest以外のnon-GPモデルを検証するものではない。
