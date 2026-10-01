# Nonstationary Gaussian Process

## 直感

stationary kernel は、入力空間のどこにいても「同じ距離」が同じ covariance structure を持つと
仮定します。例えば stationary RBF kernel では、global lengthscale が関数全体の smoothness を
支配します。

しかし実際の材料・製造データでは、ある領域では緩やかに変化し、相境界、閾値、遷移領域では
急激に変化することがあります。nonstationary GP は latent function の局所 geometry を
入力位置に依存させ、このような spatially varying smoothness を表現します。

### 最初に「何が場所で変わるのか」を分ける

見かけ上「領域によって挙動が違う」データでも、原因は複数あります。

~~~text
latent functionの相関距離が変わる
  → nonstationarity

observation scatterが変わる
  → heteroskedasticity

入力位置そのものが不確か
  → uncertain-input problem

既知のbranch / regimeで関数が分かれる
  → hierarchy / context / piecewise structureも検討
~~~

この章のnonstationarityは、主に**latent covariance geometryがXに依存する**問題です。

## 1. Stationarity

stationary covarianceでは、入力を同じ量だけ平行移動してもcovarianceが変わらず、概念的に

```text
k(x, x') = k(x - x')
```

と書けます。RBF kernel なら

```text
k(x, x')
  = sigma_f^2 exp(
      -1/2 sum_d (x_d - x'_d)^2 / ell_d^2
    )
```

であり、ARD を使っても各 `ell_d` は入力空間全体で共有されます。

これは「入力次元ごとにsmoothnessが違う」ことは表現できますが、
「同じ入力次元でも場所によってsmoothnessが違う」ことは表現しません。

なおstationaryとisotropicは別概念です。ARD RBFは方向ごとに異なるlengthscaleを持つため
anisotropicですが、lengthscaleが場所によらずglobalならstationaryです。

## 2. Input-dependent local lengthscale

nonstationary covariance の代表的な考え方は

```text
ell = ell(x)
```

として local lengthscale を入力依存にすることです。

robotorchan は Gibbs 型 covariance を使います。1次元では概念的に

```text
k(x, x')
  = sqrt(
      2 ell(x) ell(x')
      / (ell(x)^2 + ell(x')^2)
    )
    * exp(
        -(x - x')^2
        / (ell(x)^2 + ell(x')^2)
      )
```

です。

多次元では dimension ごとの prefactor の積と、dimension ごとの距離項の和を使います。
`ell_d(x)` が一定ならstationary RBF型covarianceに対応する形へ戻ります。

このGibbs kernelで変化させているのはlocal lengthscaleです。outputscaleやobservation noiseまで
同じ仕組みで入力依存にしているわけではありません。

## 3. robotorchan の local-lengthscale parameterization

`GibbsKernel` では local lengthscale を別の GP で推論するのではなく、入力の affine function
を softplus へ通す軽量な parameterization を使います。

```text
a(x) = b + W x

ell(x) = softplus(a(x)) + ell_floor
```

ここで

- `b`: `lengthscale_intercept`
- `W`: `lengthscale_slope`
- `ell_floor`: `lengthscale_floor`

です。

`softplus` と正の floor により、すべての入力位置で lengthscale を正に保ちます。

この設計は一般的な「任意の nonstationary GP」ではなく、**affine local-lengthscale Gibbs
kernel** です。理論上可能な nonstationary kernel 全般と、robotorchan の実装範囲を
区別する必要があります。

### 現行parameterizationが表現できる範囲

`lengthscale_slope` はd × d matrixなので、各local lengthscaleは自分自身の座標だけでなく、
他のcontinuous coordinateにも依存できます。

一方、softplusへ入る前はaffine functionです。そのため任意に複雑なlengthscale fieldを
表現するnonstationary GPではありません。

~~~text
柔軟性
  global stationary lengthscale
    <
  affine + softplus local lengthscale
    <
  より一般的なlatent lengthscale process等
~~~

この制約は、少標本で自由度を増やしすぎないという側面もあります。

## 4. Local lengthscale の解釈

`ell_d(x)` が小さい領域では、入力の小さな変化でも covariance が急速に低下します。
したがって latent function は局所的に速く変化できるようになります。

逆に `ell_d(x)` が大きい領域では、より離れた点同士も強く相関し、滑らかな関数を
表現しやすくなります。

robotorchan の各 nonstationary model は `local_lengthscale(X)` を公開しており、
学習された局所 smoothness を診断できます。

これは posterior uncertainty そのものではなく、kernel geometry の診断量です。

### local lengthscaleは物理的な「相境界検出器」ではない

小さい `ell(x)` が得られた領域は、modelが短い相関距離を必要としていることを示します。
しかし、それだけから相転移・故障regime・因果的境界が存在すると断定はできません。

同じ現象は、未説明のcategorical factor、急なmean structure、外れ値、data sparsityなどでも
生じ得ます。`local_lengthscale(X)` は**kernel diagnostic**として解釈し、domain knowledgeや
predictive validationと組み合わせます。

## 5. Single-task model

`NonstationarySingleTaskGP` は

```text
ScaleKernel(GibbsKernel)
```

を covariance module とする exact GP です。

`train_Yvar` を指定できるため、既知のobservation noise varianceを持つ場合にも
nonstationary latent covarianceとobservation noiseを分離できます。

ただし `train_Yvar` を渡せることと、未知のheteroskedastic noise surfaceを学習することは
同じではありません。

ここで変化するのは latent covariance の smoothness であり、noise variance を
入力依存にしているわけではありません。

## 6. Mixed variables

`MixedNonstationarySingleTaskGP` は nonstationarity を continuous dimensions に限定し、
categorical dimensions は native categorical covariance で扱います。

continuous と categorical の両方が存在する場合、mixed covariance には continuous-only
branch と continuous-categorical interaction branch があり、両方の continuous branch に
Gibbs kernel が入ります。

そのため `local_lengthscale(X)` は

```text
(additive_lengthscale, interaction_lengthscale)
```

の2つを返します。

カテゴリコード自体を local-lengthscale function の連続座標として扱うわけではありません。

## 7. Long-format MultiTask

`NonstationaryMultiTaskGP` は long-format data を対象とします。

```text
X = [data features, task feature]
Y = scalar observation
```

task identity は local smoothness を学ぶ data feature ではありません。
nonstationary data covariance と task covariance を分離し、`local_lengthscale(X)` の
公開値からも task feature を除外します。

したがって、

```text
nonstationarity in data space
!= continuous interpolation over task IDs
```

です。

### Long-format実装の注意点

現行 `NonstationaryMultiTaskGP` は、BoTorchのMultiTaskGPが期待するfull long-format inputを
data covarianceへ渡しつつ、Gibbs kernelの診断値ではtask featureを除外します。
task covariance自体はMultiTaskGP側の専用kernelが担います。

したがってpublic semanticsとしては

~~~text
continuous data geometry
  → Gibbs nonstationarity

task identity
  → task covariance
~~~

と理解します。task IDの数値差を「連続距離」として解釈するモデルではありません。

## 8. Mixed MultiTask

`MixedNonstationaryMultiTaskGP` では3種類の feature role を区別します。

- continuous data features
- categorical data features
- task feature

Gibbs nonstationarity は continuous data features のみに適用し、categorical feature と
task identity はそれぞれ専用の covariance structure で扱います。

`cat_dims` と `task_feature` の重複は許されません。

## 9. Kronecker MultiTask

`NonstationaryKroneckerMultiTaskGP` は block-design の multi-output / multitask data を
対象とし、data covariance に Gibbs kernel を使います。

概念的には

```text
K = K_task ⊗ K_data
```

という Kronecker structure の `K_data` を nonstationary にします。

task identity は `Y` の output/task axis にあるため、long-format model のような
`task_feature` 列はありません。

この違いは API 上も理論上も重要です。

## 10. Heteroskedasticity との違い

nonstationarity と heteroskedasticity は別の構造です。

| 構造 | 変化するもの | 代表量 |
| --- | --- | --- |
| Nonstationary GP | latent covariance | `ell(x)` |
| Heteroskedastic GP | observation variance | `sigma_noise^2(x)` |

見かけ上どちらも「場所によってばらつきが違う」ように見えることがあります。

しかし nonstationary GP では latent function の相関距離が変化し、heteroskedastic GP では
同じ latent function の周りの observation scatter が変化します。

詳細は [Heteroskedastic / Replicate Noise](15_heteroskedastic_noise.md) を参照してください。

## 11. Input uncertainty との違い

uncertain-input GP は観測した入力位置自体に uncertainty がある問題です。

```text
x_true ~ p(x | x_obs)
```

nonstationary GP は入力位置を deterministic としたまま、kernel geometry が場所によって
変わる問題です。

両者とも covariance を変化させますが、統計的な原因は異なります。

詳細は [Uncertain-input GP](16_uncertain_input_gp.md) を参照してください。

### Nonstationaryと「既知regime」は別問題

例えばprocess mode A/Bが既知で、それぞれで挙動が違うなら、そのmodeを無視して
nonstationary kernelだけで吸収させる必要はありません。既知category、task、context、
hierarchyとして構造をmodelへ渡せる場合があります。

逆にregime labelがなく、同じinput axis上でsmoothnessが連続的に変化するなら、
local-lengthscale modelが自然な表現になり得ます。

## 12. Bayesian optimization との関係

stationary GP が急変領域を過度に平滑化すると、

- posterior mean が局所構造を見逃す
- posterior variance が実際のモデル不確実性を適切に表さない
- acquisition が急変領域を過小評価する

可能性があります。

nonstationary surrogate は局所 lengthscale を変えることでこの misspecification を緩和できます。

一方、robotorchan の parameterization でも stationary GP より自由度が増えます。
特に小標本 BO では、局所構造を識別できるだけのデータがないと hyperparameter estimation が
不安定になる可能性があります。

したがって nonstationary model は常に stationary model より優れるというものではなく、
posterior predictive check、local lengthscale、held-out error なども確認します。

## 13. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- covarianceはinput-dependent Gibbs kernel
- stationarityとisotropyは別概念
- 現行Gibbs modelはlocal lengthscaleを変化させ、noise/outputscale全般を入力依存にはしない
- local lengthscale は affine function + softplus + positive floor
- `lengthscale_floor > 0`
- SingleTask は exact GP
- `train_Yvar` を指定可能
- Mixed 版では Gibbs kernel は continuous dimensions のみ
- Mixed SingleTask は2つの continuous Gibbs branch を持つ
- long-format MultiTask では task feature を local geometry から分離
- Mixed MultiTask では continuous / categorical / task role を分離
- Kronecker 版では nonstationarity は data covariance 側に入る
- `local_lengthscale(X)` はkernel geometryの診断APIで、regimeや物理境界の確定器ではない
- known heteroskedastic noiseを渡せても、unknown noise surfaceを自動学習するわけではない

詳細は [Nonstationary GP](../models/nonstationary_gp.md) を参照してください。

## 14. この章で覚えておくこと

- stationary kernelはcovarianceがabsolute locationではなく相対的な配置に依存する
- ARDでdimension別lengthscaleを持っても、globalならstationaryである
- nonstationarityとheteroskedasticityはlatent covarianceとobservation noiseの違いである
- robotorchanのGibbs kernelはaffine + softplusでlocal lengthscaleを表す限定的なモデルである
- local lengthscaleは他のcontinuous coordinatesにも依存できる
- Mixed / MultiTaskではcategoryやtask identityをcontinuous local geometryと分離する
- known regimeがあるならcategory / task / context / hierarchyとして明示する選択肢もある
- nonstationary modelはstationary modelを常に上回るものではなくpredictive validationが必要である

## 参考文献

1. Gibbs, M. N. (1997).
   *Bayesian Gaussian Processes for Regression and Classification*.
   PhD thesis, University of Cambridge.
2. Paciorek, C. J. and Schervish, M. J. (2004).
   Nonstationary covariance functions for Gaussian process regression.
   *Advances in Neural Information Processing Systems 16*.
3. Paciorek, C. J. and Schervish, M. J. (2006).
   Spatial modelling using a new class of nonstationary covariance functions.
   *Environmetrics*, 17(5), 483-506.
4. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [Gaussian Process](02_gaussian_process.md)
- [Kernel](03_kernel.md)
- [Heteroskedastic / Replicate Noise](15_heteroskedastic_noise.md)
- [Uncertain-input Gaussian Process](16_uncertain_input_gp.md)
- [MultiTask / MultiOutput](07_multitask_multioutput.md)
