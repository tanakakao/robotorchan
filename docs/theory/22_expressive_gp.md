# Expressive Gaussian Process Models

## 直感

標準 Gaussian Process は kernel によって関数の smoothness や correlation structure を表します。

RBF / Matérn のような標準 kernel は強力ですが、

- learned nonlinear representation
- hierarchical stochastic representation
- neural-network-derived prior
- periodic / quasi-periodic / multi-frequency structure

を明示的に表現したい場合があります。

robotorchan では expressive GP を、表現力の源が異なる複数 family として扱います。

### 最初に「何を柔軟にしているか」を分ける

expressive GPでは、複雑さを追加する場所が異なります。

| Family | 柔軟にする場所 | latent representation |
| --- | --- | --- |
| Joint neural GP | kernelへ入る決定論的feature map | deterministic |
| Deep GP | GP mappingそのものを階層化 | stochastic |
| Infinite-width BNN GP | kernel priorの関数形 | explicit latentなし |
| Spectral Mixture GP | stationary kernelのspectral density | explicit latentなし |

したがって「neural」という語が付くモデルでも、有限networkのweightを学習するのか、
GP layerを確率的に積むのか、無限幅極限のanalytic kernelを使うのかで意味が異なります。

## 1. Expressiveness は1種類ではない

この章で扱う主な構造は次です。

| Family | 表現力の源 | 推論 |
| --- | --- | --- |
| Joint neural GP / DKL | learned deterministic features | Exact GP |
| Deep GP | stochastic GP hierarchy | Variational / MC |
| Infinite-width BNN GP | neural-network kernel prior | Exact GP |
| Spectral Mixture GP | spectral density mixture | Exact GP |

これらはすべて「複雑な関数を表せる」という点では共通しますが、posterior semantics と
training contract は大きく異なります。

## 2. Deep Kernel Learning / Joint neural GP

決定論的 neural feature extractor `g_theta` を使うと、

```text
z = g_theta(x)

k_DKL(x, x')
  = k_GP(
      g_theta(x),
      g_theta(x')
    )
```

と考えられます。

robotorchan の `JointEncoderGP` は encoder と GP hyperparameters を同じ predictive objective
から学習します。

```text
x
 -> neural encoder
 -> latent z
 -> Exact GP
```

frozen `AutoEncoderGP` のような二段階 reduction と違い、GP marginal likelihood の勾配が
encoder へ伝わります。

詳細は [Neural Representation GP](19_neural_reduction_gp.md) を参照してください。

## 3. DKL と neural reduction の境界

同じ neural encoder を使っていても、

```text
frozen AE
 -> representation learning
 -> freeze
 -> GP fitting
```

と

```text
JointEncoderGP
 -> encoder + GP joint optimization
```

は別の statistical model です。

前者は dimensionality reduction infrastructure、後者は learned kernel geometry として
理解できます。

このため `JointEncoderGP` は neural-reduction章と expressive-model章の両方に関係しますが、
同じ説明を重複させるのではなく役割を分けます。

## 4. Mixed Joint neural GP

`MixedJointEncoderGP` は continuous features だけを encoder に渡し、categorical features を
native mixed covariance 側へ保持します。

```text
continuous X
 -> encoder
 -> latent continuous Z

categorical X
 -> native categorical covariance
```

カテゴリコードを連続 neural geometry へ暗黙に混ぜないことが重要です。

## 5. Deep Gaussian Process

Deep GP は GP mapping 自体を階層化します。

```text
x
 -> GP_1
 -> h_1
 -> GP_2
 -> ...
 -> y
```

中間表現 `h_l` が stochastic variable である点が DKL と本質的に異なります。

階層を周辺化すると一般に単純な Gaussian posterior にはならないため、variational inference と
Monte Carlo approximation が必要になります。

## 6. SingleTaskDeepGP

`SingleTaskDeepGP` は hidden GP layers と output GP layer を持つ variational DeepGP です。

主な構成 parameter は

- `hidden_dims`
- `num_inducing`
- `posterior_samples`
- input standardization

です。

各 hidden layer は inducing-point variational GP として構成されます。

public posteriorはlatent hierarchyから複数sampleを生成した
`DeepGPPosterior` を返します。これは単一のanalytic Gaussian posteriorへ潰したものではなく、
sample-based posterior contractです。

Tensor-valued `observation_noise` など、Exact GP と同一でない posterior contract もあるため、
「BoTorch model interfaceを持つ」ことと「Exact GPと全機能が同じ」ことは区別します。

## 7. DeepGP training

DeepGP は exact marginal likelihood を解析的に計算する model ではありません。

variational objectiveによりinducing distributionsとkernel parametersを学習します。

現行 `make_mll()` は共通training APIを維持するため存在しますが、返すのはExact
`ExactMarginalLogLikelihood` ではなく `DeepApproximateMLL(VariationalELBO)` です。
したがって `supports_mll = True` は「Exact GPである」という意味ではありません。

`training_loss()` も同じELBOを負号付きlossとして評価し、likelihood samplesを使います。
Exact GPのtraining semanticsをそのまま当てはめず、DeepGPのvariational contractに従います。

posterior も latent-function samples を通じて近似されます。

## 8. MultiTask DeepGP

robotorchan は long-format `MultiTaskDeepGP` を提供します。

task identity は単なる continuous scalar input として扱うのではなく、学習可能な task
representation として stochastic hierarchy へ結合します。

```text
data features
 + task representation
 -> DeepGP hierarchy
```

このため standard `MultiTaskGP` の exact task covariance と同一の構造ではありません。

## 9. Mixed DeepGP

`MixedSingleTaskDeepGP` は categorical feature を learnable embedding として扱います。

これは `MixedJointEncoderGP` の native categorical covariance とも、Mixed SAAS の one-hot
encoding とも異なる categorical treatment です。

つまり robotorchan の Mixed model は underlying mathematics に応じて、

- native categorical kernel
- one-hot transform
- learnable categorical embedding

を使い分けます。

## 10. Mixed × MultiTask DeepGP

現在は `MixedMultiTaskDeepGP` も実装されています。

このmodelでは、

- continuous featuresはcontinuous branchとして標準化
- categorical featuresはlearnable embeddings
- task identityはcategorical featureとは別のlearnable task embedding

として分離した後、DeepGP hierarchyへ結合します。

`task_feature` を `cat_dims` に含めることはできません。また現行APIではtask/category indexは
non-negative integerかつtraining data内でcontiguous zero-basedであることを要求します。

これは「MixedとMultiTaskを名前だけ組み合わせたwrapper」ではなく、category identityとtask
identityを別のstructural variablesとして扱うためのcontractです。

## 11. Infinite-width neural network GP

有限幅 neural network の width を無限大へ取ると、適切な weight prior の下で function prior が
Gaussian Process に収束する場合があります。

robotorchan の `InfiniteWidthReLUKernel` は fully-connected ReLU network に対応する NNGP
covariance recursion を解析的に計算します。

初期 covariance は概念的に

```text
K^0(x, x')
  = sigma_b^2
    + sigma_w^2
      <x, x'> / D
```

で、各 hidden layer で ReLU covariance map を再帰適用します。

## 12. InfiniteWidthBNNGP

`InfiniteWidthBNNGP` は

```text
ScaleKernel(
  InfiniteWidthReLUKernel
)
```

を使う Exact GP です。

主な構造 parameter は

- `depth`
- `weight_variance`
- `bias_variance`
- `ard`

です。

有限neural networkのweightを学習するDKLとは違い、neural-network-derived inductive biasを
kernel priorとして使います。

現行実装では `depth`、`weight_variance`、`bias_variance` はconstructorで与える構造値で、
GP fitting中に通常の `torch.nn.Parameter` として最適化される値ではありません。一方、
kernelのlengthscaleはGP hyperparameterとして学習できます。

したがってinferenceはExact GPのままですが、「有限BNNのweight posteriorを近似している」
という意味ではありません。

## 13. Infinite-width BNN MultiTask / Kronecker

現在の実装は SingleTask に限定されません。

- `InfiniteWidthBNNMultiTaskGP`
- `InfiniteWidthBNNKroneckerMultiTaskGP`

があります。

long-format MultiTask では task feature を NNGP data kernel から除外し、task covariance 側で
扱います。

Kronecker model では NNGP kernel が data covariance を担当し、task covariance と Kronecker
structure を構成します。

## 14. Mixed Infinite-width BNN

さらに、

- `MixedInfiniteWidthBNNGP`
- `MixedInfiniteWidthBNNMultiTaskGP`
- `MixedInfiniteWidthBNNKroneckerMultiTaskGP`

があります。

Mixed model では continuous NNGP covariance と categorical covariance の役割を分離します。

したがって「Infinite-width BNN GP = single-task neural kernel」という説明では現在の実装範囲を
十分に表しません。

### NNGP kernelはstationary kernelとは限らない

Infinite-width ReLU kernelはinner productや各入力自身のnormに依存するため、一般に
RBFのようなtranslation-invariant stationary covarianceとは異なります。

したがって「Exact GPだからstationary」という対応はありません。Exact / variationalは
inferenceの分類、stationary / nonstationaryはcovariance structureの分類です。

## 15. Spectral representation

Bochner の定理により、continuous stationary kernel は spectral density と対応づけられます。

Spectral Mixture kernel は spectral density を Gaussian mixture で近似します。

1次元の概念形は

```text
k(tau)
  = sum_q
      w_q
      exp(
        -2 pi^2 tau^2 v_q
      )
      cos(
        2 pi tau mu_q
      )
```

です。

`mu_q` は frequency、`v_q` は spectral width、`w_q` は mixture weight に対応します。

## 16. SpectralMixtureGP

`SpectralMixtureGP` は GPyTorch `SpectralMixtureKernel` を使う Exact GP です。

`num_mixtures` が spectral components の数を制御します。

initialization は現在、

- `"data"`
- `"empspect"`

を選択できます。

mixture数を増やすと表現可能なspectral componentsは増えますが、常にpredictive performanceが
改善するとは限りません。mixture componentsにはlabel symmetryもあり、parameterそのものの
一意な解釈より、得られるcovariance / predictionを重視します。

また `initialization="data"` と `"empspect"` は初期parameterの作り方であり、
別のposterior familyを意味しません。

## 17. Stationary であること

Spectral Mixture kernel は柔軟ですが、基本的には stationary covariance family です。

複数の frequency を表現できることと、

```text
ell = ell(x)
```

のように location-dependent covariance を持つことは別です。

nonstationarity が必要なら
[Nonstationary Gaussian Process](17_nonstationary_gp.md)
と区別します。

## 18. Spectral Mixture MultiTask / Kronecker

現在の実装には

- `SpectralMixtureMultiTaskGP`
- `SpectralMixtureKroneckerMultiTaskGP`

があります。

long-format MultiTask では task feature を spectral data kernel から除外します。

Kronecker model では spectral kernel を data covariance に使います。
初期化時には multi-output `Y` の task平均を spectral initialization target として使います。

## 19. Mixed Spectral Mixture

さらに、

- `MixedSpectralMixtureGP`
- `MixedSpectralMixtureMultiTaskGP`
- `MixedSpectralMixtureKroneckerMultiTaskGP`

があります。

Mixed model では spectral structure を continuous dimensions に適用し、categorical structure を
別 covariance branch で扱います。

したがって categorical code を frequency axis として解釈しません。

## 20. 4 family の比較

| Model family | Representation | Posterior | 主な構造 |
| --- | --- | --- | --- |
| Joint neural GP | deterministic learned feature | Exact GP | nonlinear learned geometry |
| DeepGP | stochastic GP hierarchy | MC approximation | hierarchical nonlinear mapping |
| Infinite-width BNN GP | analytic NNGP kernel | Exact GP | neural prior |
| Spectral Mixture GP | spectral mixture kernel | Exact GP | stationary multi-frequency |

表現力の高さだけで優劣は決まりません。

data size、optimization stability、posterior calibration、acquisition cost、構造仮定との整合性を
比較する必要があります。

## 21. Bayesian optimization との接続

Joint neural GP、Infinite-width BNN GP、Spectral Mixture GP は Exact GP-compatible posterior を
提供し、通常の BoTorch acquisition と接続できます。

DeepGP は Monte Carlo posterior を提供するため、MC acquisition との組合せが自然です。

いずれも public candidate は original design-variable space を基本とします。

```text
original candidate X
 -> model-internal representation
 -> posterior
 -> acquisition
```

利用者が candidate search のために手動で latent representation へ変換する必要はありません。

これは REMBO / ALEBO / BAxUS のような search-space strategy とは別の責務です。

## 22. High-dimensional model との違い

expressive model と high-dimensional model は重なる場合がありますが同義ではありません。

例えば、

- SAAS: sparse relevance prior
- PCA GP: explicit dimension reduction
- JointEncoderGP: learned nonlinear kernel geometry
- InfiniteWidthBNNGP: neural prior without explicit reduction
- SpectralMixtureGP: spectral stationary structure

です。

したがって model selection は「高次元だから expressive model」という単純な対応ではなく、
どの structural assumption が必要かで判断します。

## 23. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- Joint neural GP は deterministic learned feature + Exact GP
- frozen AE/VAE reduction と joint neural GP を区別する
- Mixed Joint neural GP は continuous encoding と categorical covariance を分離
- DeepGP は stochastic hierarchy + variational inference
- DeepGP posterior は Monte Carlo approximation
- MultiTaskDeepGPはtask representationをhierarchyに組み込む
- MixedSingleTaskDeepGPはcategorical embeddingを使う
- MixedMultiTaskDeepGPはcontinuous / category / task representationを分離してDeepGPへ結合する
- DeepGPの `make_mll()` はDeepApproximateMLL / VariationalELBOでありExact MLLではない
- Infinite-width BNNはanalytic ReLU NNGP kernel + Exact GP
- NNGPのdepth / weight variance / bias varianceは現行ではconstructor設定値
- Infinite-width ReLU kernelは一般にstationary covarianceとは限らない
- Infinite-width BNN は SingleTask / MultiTask / Kronecker を持つ
- Infinite-width BNN は Mixed variants も持つ
- Spectral Mixture は stationary spectral-density mixture
- Spectral Mixtureはdata / empirical-spectrum initializationを選択可能
- initialization方式は初期化の違いでありposterior familyの違いではない
- Spectral Mixture は SingleTask / MultiTask / Kronecker を持つ
- Spectral Mixture は Mixed variants も持つ
- expressive surrogate と search-space strategy を分離する

## 24. この章で覚えておくこと

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

## robotorchan の関連ドキュメント

- [Expressive models](../models/expressive.md)
- [Joint neural GP](../models/joint_neural_training.md)
- [DeepGP](../models/deep_gp.md)
- [Infinite-width BNN GP](../models/infinite_width_bnn.md)
- [Spectral Mixture GP](../models/spectral_mixture.md)
- [BoTorch integration](../models/expressive_botorch_integration.md)
- [Predictive benchmark](../benchmarks/expressive_predictive.md)

## 参考文献

1. Wilson, A. G., Hu, Z., Salakhutdinov, R., and Xing, E. P. (2016).
   Deep kernel learning.
   *Artificial Intelligence and Statistics*.
2. Damianou, A. and Lawrence, N. D. (2013).
   Deep Gaussian processes.
   *Artificial Intelligence and Statistics*.
3. Salimbeni, H. and Deisenroth, M. (2017).
   Doubly stochastic variational deep Gaussian processes.
   *Advances in Neural Information Processing Systems*.
4. Neal, R. M. (1996).
   *Bayesian Learning for Neural Networks*. Springer.
5. Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., and Sohl-Dickstein, J.
   (2018).
   Deep neural networks as Gaussian processes.
   *International Conference on Learning Representations*.
6. Wilson, A. G. and Adams, R. P. (2013).
   Gaussian process kernels for pattern discovery and extrapolation.
   *International Conference on Machine Learning*.
7. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [Neural Representation GP](19_neural_reduction_gp.md)
- [Advanced High-dimensional Models](21_advanced_high_dimensional_models.md)
- [Nonstationary GP](17_nonstationary_gp.md)
- [High-dimensional Search](20_high_dimensional_search.md)
