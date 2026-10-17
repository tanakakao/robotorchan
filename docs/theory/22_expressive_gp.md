# Expressive Gaussian Process Models

## 直感

標準 Gaussian Process は kernel によって関数の smoothness や correlation structure を表します。

RBF / Matérn のような標準 kernel は強力ですが、

- learned nonlinear representation
- hierarchical stochastic representation
- neural-network-derived prior
- periodic / quasi-periodic / multi-frequency structure

を明示的に表現したい場合があります。

Expressive GPは、表現力の源が異なる複数のモデル群として整理できます。

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

## 6. Deep GPの変分表現

Deep GPでは複数の確率的GP写像を合成します。中間層の関数値を積分した予測分布は一般に単一の解析的Gaussian分布にならず、サンプリングにより近似します。

Inducing-point変分分布を各層に導入することで計算を近似できますが、層数・誘導点数・サンプル数が近似精度と計算量に影響します。

## 7. Deep GPの変分推論

Deep GPの周辺尤度は中間層の確率的写像により一般に解析的に計算できません。変分推論では、誘導変数の近似事後分布を導入し、ELBOを最大化します。

$
\mathcal L_{\mathrm{ELBO}}=\mathbb E_{q(F)}[\log p(Y\mid F)]-\mathrm{KL}(q(U)\|p(U))
$

この期待値はモンテカルロ近似される場合があります。Exact GPの周辺尤度最大化とは推論対象も近似誤差も異なります。

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

## 9. Mixed入力を扱うDeep GP

カテゴリ変数を確率的階層に組み込む際には、カテゴリ固有の共分散、適切な符号化、学習可能な埋め込みなどの異なる構造仮定があります。カテゴリ番号を無条件に連続値として扱うと、存在しない順序・距離を導入するため注意が必要です。

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

## 12. 無限幅ニューラルネットワーク由来のGP

適切な重み事前分布と幅の極限の下で、ニューラルネットワークの関数事前分布はGPに収束します。得られるNNGP共分散を通常のGPに用いる場合、有限幅ネットワークの重みを直接学習するわけではありません。

深さ、重み分散、バイアス分散は共分散関数の形状を決めます。これらを固定するか推定するかは推論設計の選択であり、無限幅極限そのものの性質とは区別します。

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

## 16. Spectral Mixture Kernelのモデル化

Spectral Mixture Kernelは、定常共分散関数のスペクトル密度をGaussian混合などで近似することで、複数の周波数成分や準周期性を表現します。混合成分を増やすと表現力は増しますが、過適合や推定不安定性にも注意が必要です。

スペクトルパラメータの初期化方法は最適化の出発点を変えるものであり、異なる事後分布族を定義するものではありません。混合成分の置換対称性のため、個々の成分の識別性にも注意が必要です。

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

## 23. 理論と実装の境界

Expressive GPの理論では、決定論的な特徴写像、確率的なGP階層、無限幅極限から導かれるカーネル、定常スペクトル混合を区別します。また、Exact GPとDeep GPの変分推論を同一視しないことが重要です。

モデル別のコンストラクタ、学習API、Posterior契約、初期化オプションについては[Expressive models](../models/expressive.md)と[BoTorch integration](../models/expressive_botorch_integration.md)を参照してください。

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
