# Expressive GP：表現力の源と推論の違い

## 1. どの構造を柔軟にするか

Expressive GPは単一のアルゴリズムではない。[理論第22章](../theory/22_expressive_gp.md)では、deterministic neural features、stochastic GP hierarchy、neural-network-derived kernel prior、spectral covarianceの違いを説明している。この章では、それぞれの**posteriorの意味**と学習APIへの接続を扱う。

| Family | 数理構造 | robotorchanでの公開入口 | 学習・推論の要点 |
| --- | --- | --- | --- |
| Deep Kernel Learning (DKL) | `k(g_θ(x),g_θ(x'))` | `JointEncoderGP` | encoderとExact GPをjoint trainingする |
| DeepGP | 階層的な確率GP写像 | `SingleTaskDeepGP` | variational / MC推論でありExact MLLではない |
| Infinite-width BNN | 無限幅NNに対応するNNGP kernel | `InfiniteWidthBNNGP` | kernel prior上のExact GP |
| Spectral Mixture | 複数周波数成分による定常kernel | `SpectralMixtureGP` | Exact GP、mixture数と初期化に敏感 |

## 2. アルゴリズムから実装へ

**Extended / Delegated：** DKLに関連する`src/robotorchan/models/high_dimensional/reduced/joint_neural.py`の`JointEncoderGP`は、元入力をencoderへ通してGPに渡す。BoTorch/GPyTorchが基本GPの推論を扱う一方、robotorchanはfeature extractorとGP objectiveを結合する`training_loss`経路を担う。これはfrozen AEの「先に表現を学習してから固定する」経路と区別する。

**Implemented / Delegated：** DeepGP系は確率的な複数GP layerを通すため、出力のサンプリングとvariational学習を要求する。posterior samplesの意味がDKLのdeterministic latent featuresと異なる。NNGPおよびSpectral Mixtureはkernel側の選択なので、DeepGPの層構造を持たない。

**Mixed / MultiTask契約：** `MixedJointEncoderGP`ではカテゴリ列を連続encoderへ無条件に渡さない。`JointEncoderMultiTaskGP`のlong-format task featureと`JointEncoderKroneckerMultiTaskGP`のblock-design task軸は別であり、予測時に与える元入力形状も異なる。実際の対応モデルは[Expressiveモデルガイド](../models/expressive.md)で区別する。

## 3. 最小の入口例

```python
import torch
from robotorchan.models import JointEncoderGP

train_X = torch.rand(16, 4, dtype=torch.double)
train_Y = (train_X[:, :1] - 0.4).square()
model = JointEncoderGP(
    train_X=train_X,
    train_Y=train_Y,
    latent_dim=2,
    hidden_dims=(16, 8),
)
print(model.latent_dim)
```

モデル構築例であり、学習完了やencoder勾配検証を示すものではない。Joint系の実用では、`training_loss`に基づくoptimizer step、encoder gradient、保存・再読込後の入力変換、posteriorサンプルshapeを確認する。

## 4. 互換性と検証の境界

MC acquisitionがposteriorsを消費できても、全Expressive familyで任意の`q`、constraint、`optimize_acqf`が成功するとは限らない。DeepGPのstochastic posteriorを単一Gaussian posteriorとみなさず、NNGPのpriorを有限NNのweight posteriorとみなさない。Spectral Mixtureでは入力スケール・初期化・mixture数を比較条件として固定する。

参照：[Neural Representation](19_neural_reduction_gp.md)、[Expressive Notebook](../../examples/notebooks/24_expressive_surrogate_gp.ipynb)、[旧Models Mapping](../../.github/project-records/theory-guide/mapping/models.md)、[Verification](../../.github/project-records/theory-guide/mapping/verification.md)。完了した検証の範囲は数値propertyと実行環境により限定する。

## 5. 確率的深層GPと決定論的特徴写像の違い

Deep Kernel Learningでは、決定論的encoder g_theta(x)を通した特徴の上でGP kernel k(g_theta(x), g_theta(x'))を評価する。一方、DeepGPは確率的なGP layerを合成し、各中間層の不確実性を積分・サンプリングして推論する。後者を単一のExact GP kernelへ置き換えることは一般にできない。

src/robotorchan/models/expressive/deep_gp.pyのSingleTaskDeepGPはGPyTorch DeepGPを継承し、hidden_layersとoutput_layerの各層にVariationalStrategy、CholeskyVariationalDistribution、RBFKernelを構成する。make_mll(num_data)はDeepApproximateMLL(VariationalELBO(...))を返し、training_loss()はその負値を評価する。Exact GPのExactMarginalLogLikelihoodを流用しない。

DeepGPのposterior(X)はnum_samples個のサンプルからDeepGPPosteriorを返す。Tensor-valued observation_noiseはNotImplementedErrorで拒否し、num_samplesは2以上を要求する。GaussianLikelihoodを介したobservation_noise=Trueの経路と、latent samplesの経路を区別する。単一のGaussian posteriorとみなして解析的獲得関数へ無条件に渡さない。

## 6. NNGP kernelとSpectral Mixtureの推論契約

| Family | covarianceの構成 | 訓練の意味と制約 |
| --- | --- | --- |
| InfiniteWidthBNNGP | infinite-width ReLU network由来のanalytic kernel recursion | src/robotorchan/models/expressive/infinite_width_bnn.pyでInfiniteWidthReLUKernelを実装し、SingleTaskGPのExact GP学習へ接続。有限幅NNのweight posteriorではない |
| SpectralMixtureGP | GPyTorch SpectralMixtureKernelをScaleKernelで包む | src/robotorchan/models/expressive/spectral_mixture.pyがnum_mixturesと初期化方式を検証する。num_mixtures>0、initializationはdata/empspect |
| MultiTask variants | task列の除去またはblock-design output軸の扱い | long-formatではtask_featureを通常の設計kernelに入れず、Kroneckerではtrain_Yのtask軸を保持 |
| Mixed variants | カテゴリ用kernelと連続側のexpressive kernelを合成 | 全モデルでone-hotするという一般化は誤り。連続設計次元を必要とするvariantがある |

InfiniteWidthReLUKernelの再帰では、初期線形共分散から各ReLU層の解析的共分散写像を反復する。これは無限幅における**prior covariance**を構成する処理で、ニューラルネットワークのパラメータを直接学習する方法ではない。GP posteriorとMLL計算はSingleTaskGP経由でBoTorch/GPyTorchへ委譲される。

SpectralMixtureGPでは学習前にkernelのspectral parameterをtrain_X/Yから初期化し、その後Exact GPとして学習する。初期化はdataまたはempspectのみで、収束・周期性・予測性能を保証する処理ではない。SpectralMixtureMultiTaskGPはtask featureをdesign kernelの入力から除外し、Kronecker版は複数出力を使った初期化用ターゲットを構成する。

## 7. 何を数値検証する必要があるか

DeepGPではMC sample数・ランダムseed・training_loss勾配・posteriorの形状とサンプリングの再現性を検証する。Infinite-width BNNではkernelの対称性・PSD・深さとパラメータの感度、Spectral Mixtureでは初期化方式・mixture数・入力スケールに対する安定性を検証する。DKLはencoder gradientとGP推論の接続を別途確かめる。今回は上記ソース契約の確認であり、独立数値検証や学習実行を行っていない。

## 実装補足：理論本文から分離した対応情報

### robotorchan の関連ドキュメント

- [Expressive models](../models/expressive.md)
- [Joint neural GP](../models/joint_neural_training.md)
- [DeepGP](../models/deep_gp.md)
- [Infinite-width BNN GP](../models/infinite_width_bnn.md)
- [Spectral Mixture GP](../models/spectral_mixture.md)
- [BoTorch integration](../models/expressive_botorch_integration.md)
- [Predictive benchmark](../benchmarks/expressive_predictive.md)

## 理論章から分離した実装上の条件

### 23. 理論と実装の境界

Expressive GPの理論では、決定論的な特徴写像、確率的なGP階層、無限幅極限から導かれるカーネル、定常スペクトル混合を区別します。また、Exact GPとDeep GPの変分推論を同一視しないことが重要です。

モデル別のコンストラクタ、学習API、Posterior契約、初期化オプションについては[Expressive models](../models/expressive.md)と[BoTorch integration](../models/expressive_botorch_integration.md)を参照してください。

#### 2. Deep Kernel Learning / Joint neural GP：実装上の補足

robotorchan の `JointEncoderGP` は encoder と GP hyperparameters を同じ predictive objective
から学習します。

#### 8. MultiTask DeepGP：実装上の補足

robotorchan は long-format `MultiTaskDeepGP` を提供します。

#### 10. Mixed × MultiTask DeepGP：実装上の補足

`task_feature` を `cat_dims` に含めることはできません。また現行APIではtask/category indexは
non-negative integerかつtraining data内でcontiguous zero-basedであることを要求します。

#### 11. Infinite-width neural network GP：実装上の補足

robotorchan の `InfiniteWidthReLUKernel` は fully-connected ReLU network に対応する NNGP
covariance recursion を解析的に計算します。

#### 13. Infinite-width BNN MultiTask / Kronecker：実装上の補足

現在の実装は SingleTask に限定されません。

#### 14. Mixed Infinite-width BNN：実装上の補足

したがって「Infinite-width BNN GP = single-task neural kernel」という説明では現在の実装範囲を
十分に表しません。

#### 18. Spectral Mixture MultiTask / Kronecker：実装上の補足

現在の実装には

#### 21. Bayesian optimization との接続：実装上の補足

Joint neural GP、Infinite-width BNN GP、Spectral Mixture GP は Exact GP-compatible posterior を
提供し、通常の BoTorch acquisition と接続できます。

#### 24. この章で覚えておくこと：実装上の補足

- expressive GPは「どこを柔軟にするか」で分類する
- Joint neural GPはdeterministic learned feature mapとExact GPをjoint trainingする
- DeepGPはstochastic GP hierarchyであり、variational ELBOとsample-based posteriorを使う
- `supports_mll=True` でもDeepGPのMLLはExact MLLではない
- MixedMultiTaskDeepGPはcategory embeddingとtask embeddingを別構造として扱う
- Infinite-width BNN GPは有限BNNを学習するmodelではなくanalytic NNGP kernelを使うExact GPである
- Infinite-width ReLU kernelはExact GPでも一般にstationaryとは限らない
- Spectral Mixture GPはstationary spectral structureを柔軟化する
- Spectral mixture数の増加は表現力を増やすが、性能改善を保証しない
- expressive surrogateとcandidate search strategyは別レイヤーである

## 数値 API と利用上の補足

現在は `混合入力 マルチタスク Deep GP` も実装されています。

これは「MixedとMultiTaskを名前だけ組み合わせたwrapper」ではなく、category identityとtask
identityを別のstructural variablesとして扱うためのcontractです。

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。


frozen `AutoEncoderGP` のような二段階 reduction と違い、GP marginal likelihood の勾配が
encoder へ伝わります。

```text
JointEncoderGP
 -> encoder + GP joint optimization
```


このため `JointEncoderGP` は neural-reduction章と expressive-model章の両方に関係しますが、
同じ説明を重複させるのではなく役割を分けます。

`MixedJointEncoderGP` は continuous features だけを encoder に渡し、categorical features を
native mixed covariance 側へ保持します。


このため standard `MultiTaskGP` の exact task covariance と同一の構造ではありません。

現在は `MixedMultiTaskDeepGP` も実装されています。

- `InfiniteWidthBNNMultiTaskGP`
- `InfiniteWidthBNNKroneckerMultiTaskGP`

- `MixedInfiniteWidthBNNGP`
- `MixedInfiniteWidthBNNMultiTaskGP`
- `MixedInfiniteWidthBNNKroneckerMultiTaskGP`

- `SpectralMixtureMultiTaskGP`
- `SpectralMixtureKroneckerMultiTaskGP`

- `MixedSpectralMixtureGP`
- `MixedSpectralMixtureMultiTaskGP`
- `MixedSpectralMixtureKroneckerMultiTaskGP`

- SAAS: sparse relevance prior
- PCA GP: explicit dimension reduction
- JointEncoderGP: learned nonlinear kernel geometry
- InfiniteWidthBNNGP: neural prior without explicit reduction
- SpectralMixtureGP: spectral stationary structure
