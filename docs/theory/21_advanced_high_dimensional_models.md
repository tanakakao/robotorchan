# Advanced High-dimensional Gaussian Process Models

## 直感

高次元問題への対応は projection だけではありません。

```text
high-dimensional X
 -> projection
```

以外にも、

- 元特徴のごく一部だけが効く
- 関数が低次元 component の和で表せる
- 特定の low-dimensional linear subspace に目的関数が存在する
- 観測の一部だけが異常である

といった構造仮定を prior / kernel / likelihood に組み込めます。

この章では「高次元」という共通ラベルの下にある異なる statistical assumptions を分離します。

### 最初に「高次元へ効く理由」を分ける

同じhigh-dimensional modelでも、次元の難しさを減らす仮定は同じではありません。

| 構造仮定 | 何を簡単にするか | 代表例 |
| --- | --- | --- |
| axis-aligned sparsity | 多くの元特徴を弱くする | SAAS / MAP-SAAS |
| additive structure | 高次相互作用を制限する | OAK |
| linear subspace | ambient space内の低次元方向を使う | ALEBO |
| learned representation | 観測Xからlatent表現を作る | PCA / PLS / AE / VAE |
| local search geometry | candidate探索範囲を局所化する | TuRBO |
| adaptive search subspace | 探索部分空間を段階的に拡張する | BAxUS |

これらは「高次元」という症状に対する異なるstatistical / search assumptionsです。
複数を同じ意味のdimension reductionとして扱いません。

## 1. ARD と dimension relevance

ARD kernel は dimension ごとの lengthscale を持ちます。

```text
k(x, x')
  = sigma^2 exp(
      -1/2 sum_j
      (x_j - x'_j)^2 / ell_j^2
    )
```

大きな `ell_j` は、そのdimensionの変化に対して、**そのkernelと他のhyperparameterを固定した条件では**
関数が比較的鈍感であることを意味します。lengthscaleだけから因果的重要度を結論づけることは
できません。

しかし通常の ARD では高次元・小標本になるほど多数の lengthscale を安定に推定しにくくなります。

そこで lengthscale / inverse lengthscale に sparsity-inducing prior を置く考え方が SAAS です。

## 2. SAAS

SAAS は多くの input dimension の effect を強く shrink し、少数の active dimensions を
残すことを狙います。

概念的には inverse lengthscale

```text
rho_j = 1 / ell_j
```

の多くを0付近へ縮めます。

これは PCA のように

```text
x -> W^T x
```

と latent coordinate を作る方法ではありません。

元のinput dimensionsを保持したまま、そのrelevanceにsparse priorを置きます。

したがってSAASが想定する低有効次元性は主として**元座標軸に沿ったsparsity**です。
任意に回転したdense linear subspaceを直接学習するALEBOやprojection modelとは仮定が異なります。

## 3. Fully Bayesian SAAS

robotorchan の

- `SaasFullyBayesianSingleTaskGP`
- `SaasFullyBayesianMultiTaskGP`

は BoTorch の fully Bayesian SAAS implementation をラップします。

hyperparameter uncertaintyはpoint estimateへ潰さずMCMC samplesとして保持されます。

これは「重要dimensionを1組に確定する」操作ではなく、SAAS priorの下で複数のhyperparameter
samplesをposterior predictionへ反映する構成です。

そのため通常の exact GP の

```text
fit_gpytorch_mll(model.make_mll())
```

という fitting contract ではありません。

robotorchan でも `supports_mll = False` とし、fitting は BoTorch の
`fit_fully_bayesian_model_nuts` に委譲します。

これは単なる実装上の違いではなく、posterior が hyperparameter posterior を積分する
fully Bayesian model であることに対応します。

## 4. Fully Bayesian MultiTask SAAS

`SaasFullyBayesianMultiTaskGP` は long-format task-feature representation を使います。

```text
X = [data features, task feature]
```

task structure と data-feature sparsity は別の役割です。

MultiTask SAAS は「task ID 自体を sparse input feature として選択する」モデルではなく、
BoTorch の multitask SAAS structure を通じて task covariance と data covariance を扱います。

## 5. Mixed Fully Bayesian SAAS

robotorchan には

- `MixedSaasFullyBayesianSingleTaskGP`
- `MixedSaasFullyBayesianMultiTaskGP`

があります。

これらは raw mixed-space API を維持しながら、モデル内部で categorical features を one-hot
encoding して SAAS model へ渡します。

重要なのは、この family が robotorchan の一般的な native categorical kernel route と
同一ではないことです。

```text
raw mixed X
 -> model-owned categorical one-hot
 -> SAAS model
```

です。

input warping を使う場合も categorical features は warp 対象から除外します。

したがって SAAS relevance の解釈は encoded dimensions を意識する必要があります。

## 6. MAP-SAAS

fully Bayesian SAAS は MCMC cost が大きいため、MAP approximation を使う選択肢があります。

robotorchan:

- `AdditiveMapSaasSingleTaskGP`
- `EnsembleMapSaasSingleTaskGP`

BoTorch の MAP-SAAS model を raw-data retention contract とともに提供します。

`num_taus` や `taus` により複数のshrinkage scalesを扱います。

ここでクラス名の `AdditiveMapSaasSingleTaskGP` にある「Additive」を、後述する
Orthogonal Additive Kernelの「入力componentの加法分解」と同一視しません。
MAP-SAAS familyとOAK familyは別のmodeling assumptionsです。

fully Bayesian SAASと異なりNUTSによるhyperparameter posterior integrationは行いません。
`AdditiveMapSaasSingleTaskGP` と `EnsembleMapSaasSingleTaskGP` のposterior semanticsも
「fully Bayesian SAASを単に高速化した同一posterior」とは扱いません。

### fitting contractも分ける

現行robotorchanではFully Bayesian SAASは `supports_mll = False` で、
BoTorchのNUTS fittingを使います。

一方MAP-SAAS wrappersは `ExactGPModelMixin` を持つため、通常のMLL fitting contractへ
接続できます。したがって「SAAS」という共通名だけから同じtraining APIだと考えません。

## 7. Mixed MAP-SAAS

robotorchan には

- `MixedAdditiveMapSaasSingleTaskGP`
- `MixedEnsembleMapSaasSingleTaskGP`

があります。

これらも categorical feature を model-owned one-hot transform で内部 encoding します。

したがって

```text
Mixed SAAS / MAP-SAAS
!= native categorical-kernel mixed GP
```

です。

raw API では category values をそのまま渡せますが、内部 sparsity geometry は encoded numeric
space 上に構成されます。

## 8. Additive structure

高次元関数が少数 active dimensions に依存するという仮定と、関数が component の和であるという
仮定は別です。

additive GP は概念的に

```text
f(x)
  = sum_k f_k(x_{S_k})
```

とします。

各 component が低次元なら、ambient dimension が大きくても statistical complexity を抑えられます。

## 9. Orthogonal Additive GP

`OrthogonalAdditiveGP` は BoTorch の Orthogonal Additive Kernel を利用します。

first-order additive componentsに加えて、`second_order=True` ではsecond-order interactionも
扱います。これは任意次数のinteractionを自由に表現するkernelとは異なり、許すinteraction構造を
明示的に制限する仮定です。

orthogonalization により component decomposition の解釈性を改善することが目的の1つです。

この model は SAAS のような active-dimension shrinkage model ではありません。

```text
SAAS:
  sparse relevance across original dimensions

OAK:
  additive decomposition across components
```

という違いがあります。

## 10. Mixed Orthogonal Additive GP

`MixedOrthogonalAdditiveGP` では categorical feature を内部 one-hot encoding します。

これは単なる実装都合ではありません。

upstream `OrthogonalAdditiveKernel` は continuous interval `[0, 1]` 上の
Gauss-Legendre quadrature による orthogonalization を前提とします。

標準 categorical kernel へ単純に差し替えると、この orthogonalization measure と整合しません。

そのため現在は

```text
categorical raw feature
 -> one-hot columns
 -> original OAK
```

という fallback を使います。

結果として component-level interpretation は raw categorical feature 単位ではなく、
**encoded column 単位**になります。

native categorical OAK を実装するなら discrete orthogonalization measure 自体を設計する必要が
あります。

また continuous raw inputs は upstream OAK の要件に従い `[0, 1]` にある必要があります。

## 11. Relevance pursuit は別の sparsity

`RobustRelevancePursuitSingleTaskGP` も sparse structure を使いますが、SAASとは対象が違います。

```text
SAAS
 -> sparse input relevance

Robust relevance pursuit
 -> sparse observation-side correction / robustness
```

です。

どちらも sparsity という語を使えますが、統計的意味は同一ではありません。

詳細は [Robust Gaussian Process](14_robust_gaussian_process.md) を参照してください。

## 12. ALEBOGP

ALEBO は objective が ambient space 内の low-dimensional linear subspace に依存するという
仮定を利用します。

`ALEBOGP` は ALEBO search embedding と整合する Mahalanobis geometry を持つ専用 surrogate です。

```text
k(z, z')
  = sigma^2 exp(
      -1/2 (z - z')^T M (z - z')
    )
```

ここで `M` が embedded coordinate の metric を表します。

これは training data から PCA reducer を fit する `ReducedGP` とは別の考え方です。

## 13. ALEBO metric uncertainty

現在の `ALEBOGP` は単なるMahalanobis RBF kernelに留まりません。

また `train_Yvar` は必須で、現行ALEBOGPはfixed-noise single-output modelとして構成されます。

実装には、

- metric parameter vector
- metric log posterior
- Hessian diagonal
- Laplace covariance estimation
- metric parameter sampling
- metric-sample predictions
- metric-marginal posterior
- acquisition model

が含まれます。

したがって metric point estimate だけでなく、Laplace approximation を用いて metric uncertainty を
acquisition model へ反映する経路があります。

metric uncertaintyのmarginalizationでは、metric sampleごとのGaussian predictionをそのまま
mixtureとして返すのではなく、meanとfull predictive covarianceを使ってGaussianへ
moment matchingします。したがってmetric mixtureの高次momentまで保持するposteriorではありません。

## 14. ALEBOGP と ALEBOStrategy

```text
ALEBOGP
 -> embedded-space surrogate geometry

ALEBOStrategy
 -> feasible target-space candidate search
```

です。

両者は協調できますが同一オブジェクトではありません。

search strategy 側は original box に対応する feasible polytope を扱い、model 側は embedded
coordinates の covariance geometry を扱います。

詳細は [High-dimensional Search](20_high_dimensional_search.md) を参照してください。

## 15. Projection model との違い

PCA / PLS / AE / VAE は observed training inputs から representation を構築します。

SAAS / additive GP / ALEBO はそれぞれ異なる構造仮定です。

| Family | 主な仮定 | 元特徴をlatentへ変換 |
| --- | --- | --- |
| SAAS | 少数の元特徴がactive | No |
| MAP-SAAS | sparse relevanceのMAP近似 | No |
| OAK | additive component structure | No |
| PCA / PLS | low-dimensional linear representation | Yes |
| AE / VAE | nonlinear representation | Yes |
| ALEBO | low-dimensional linear search subspace | Search側 |
| TuRBO | optimumをlocal regionで探索可能 | No |
| BAxUS | useful subspaceを段階的に拡張可能 | Search側 |

「高次元対応」という理由だけで同じ family とみなさないことが重要です。

## 16. Mixed model の注意

高次元 model の Mixed 対応には複数方式があります。

- native categorical covariance
- model-owned one-hot encoding
- continuous-only reduction + categorical passthrough
- search strategy 側で discrete dimensions を構造化

どの方式を使うかは underlying model の数学的構造によります。

したがって `MixedXxxGP` という名前だけから、すべてが同じ categorical kernel implementation を
持つと仮定してはいけません。

## 17. 問題構造との対応

| 観測・仮定 | model / strategy側の考え方 |
| --- | --- |
| 少数のoriginal axesが主に効く | SAAS / MAP-SAAS |
| low-order additive decompositionが妥当 | OAK |
| observed Xからlinear representationを学ぶ | PCA / PLS |
| observed Xからnonlinear representationを学ぶ | AE / VAE |
| low-dimensional linear search subspaceを仮定 | ALEBO |
| original spaceの局所探索へ集中 | TuRBO |
| search subspace dimensionを探索中に拡張 | BAxUS |

これは性能順位ではありません。高次元だからといって複数の強い仮定を無条件に重ねると、
model misspecificationを増やす可能性があります。

predictive performance、posterior calibration、計算量、BO trajectoryを同じ条件で比較します。

## 18. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- fully Bayesian SAAS は BoTorch NUTS fitting を利用し MLL-style fitting ではない
- SAAS SingleTask / MultiTask が存在する
- Mixed SAAS は model-owned one-hot encoding を使う
- categorical dimensions は input warping 対象から除外する
- MAP-SAASはAdditive / Ensemble variantsを持つ
- MAP-SAAS wrappersはExactGP MLL fitting contractを持ち、Fully Bayesian SAASとはtraining APIが異なる
- AdditiveMapSaasの名称をOAKのadditive input decompositionと同一視しない
- Mixed MAP-SAAS も model-owned one-hot encoding を使う
- OAK は additive decomposition であり SAAS sparsity とは別
- Mixed OAK は continuous orthogonalizationを維持するため one-hot fallback を使う
- Mixed OAK component interpretation は encoded-column space 上
- relevance pursuit の sparsity は observation-side robustness
- ALEBOGP は Mahalanobis embedded-space surrogate
- ALEBOGPはfixed-noise single-output modelで `train_Yvar` を必要とする
- ALEBOGPはmetric uncertaintyのLaplace / sampling経路を持つ
- metric-marginal posteriorはsample mixtureをGaussianへmoment matchingする
- ALEBOGP と ALEBOStrategy は別責務
- model reduction と search-space reduction を区別する

## 19. この章で覚えておくこと

- high-dimensional対応は1種類ではなく、sparsity・additivity・subspace・representation・local searchを分ける
- SAASは主としてoriginal axesに沿ったsparse relevanceを仮定する
- Fully Bayesian SAASはNUTS、MAP-SAASはExact GP MLL系でtraining contractが異なる
- Mixed SAAS / MAP-SAASはmodel-owned one-hotを使い、native categorical kernel routeとは異なる
- OAKはlow-order additive structureを仮定し、SAASのfeature sparsityとは別概念である
- Mixed OAKのcomponent interpretationはraw categoryではなくencoded columns上になる
- ALEBOGPはembedded-space Mahalanobis geometryを持つfixed-noise surrogateである
- ALEBOのmetric-marginal posteriorはmetric uncertaintyをmoment-matched Gaussianへ近似する
- surrogate-side high-dimensional modelとsearch-side strategyを同じレイヤーとして扱わない

## 参考文献

1. Eriksson, D. and Jankowiak, M. (2021).
   High-dimensional Bayesian optimization with sparse axis-aligned subspaces.
   *Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence*.
2. Hvarfner, C., Hellsten, E. O., and Nardi, L. (2024).
   Vanilla Bayesian optimization performs great in high dimensions.
   *International Conference on Machine Learning*.
3. Lu, X., Boukouvalas, A., and Hensman, J. (2022).
   Additive Gaussian processes revisited.
   *International Conference on Machine Learning*.
4. Letham, B., Calandra, R., Rai, A., and Bakshy, E. (2020).
   Re-examining linear embeddings for high-dimensional Bayesian optimization.
   *Advances in Neural Information Processing Systems*.
5. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [High-dimensional GP](08_high_dimensional_gp.md)
- [Dimensionality Reduction GP](18_dimensionality_reduction_gp.md)
- [Neural Representation GP](19_neural_reduction_gp.md)
- [High-dimensional Search](20_high_dimensional_search.md)
- [Robust Gaussian Process](14_robust_gaussian_process.md)
