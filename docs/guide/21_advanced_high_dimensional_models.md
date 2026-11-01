# Advanced High-dimensional GP：統計的仮定と実装を区別する

## この章の到達点

高次元問題に使える手法を名前で並べるのではなく、**何の構造を仮定するか**と**どの計算を委譲するか**で比較する。数理的な正本は
[理論第21章](../theory/21_advanced_high_dimensional_models.md)とする。

- **ARD / SAAS**
  構造仮定：元特徴軸ごとのrelevance、sparse inverse lengthscale
  実装と学習上の重要点：`high_dimensional/fully_bayesian.py`、NUTS・MCMC posterior

- **Additive MAP-SAAS**
  構造仮定：sparse priorとadditive surrogate構造
  実装と学習上の重要点：`high_dimensional/map_saas.py`、BoTorchのMAP-SAAS実装に委譲

- **Ensemble MAP-SAAS**
  構造仮定：複数のMAP状態を使用
  実装と学習上の重要点：単一GP posteriorと同一視せずensemble axis・utilityを確認

- **Orthogonal additive GP**
  構造仮定：関数を低次元項へ分解
  実装と学習上の重要点：`models/structured/additive.py`、sparsityとは違う仮定

- **ALEBOGP**
  構造仮定：低次元埋め込みに適したMahalanobis計量
  実装と学習上の重要点：`models/high_dimensional/alebo.py`、surrogateと探索strategyの連結が必要

- **PCA / PLS / AE / VAE**
  構造仮定：特徴表現の変換・潜在座標
  実装と学習上の重要点：`models/high_dimensional/reduced/`、次元削減と元空間へ戻す操作を分離

- **TuRBO / BAxUS**
  構造仮定：候補探索空間の局所化・適応変化
  実装と学習上の重要点：optimizer/strategy側、GP priorと区別

## Fully Bayesian と MAP はposteriorの意味が違う

Fully Bayesian SAASはハイパーパラメータのposteriorをMCMCで近似し、その不確実性を予測へ反映する。対してMAP-SAASは事後モード推定を利用する。
したがって同じ「SAAS」と呼んでも、uncertainty、学習時間、acquisitionとの互換性は同一ではない。
標準GPの`fit()`や`make_mll()`契約を無条件にすべての高次元モデルへ広げない。

## 実装上のカテゴリ処理

`MixedAdditiveMapSaasSingleTaskGP`と`MixedEnsembleMapSaasSingleTaskGP`は
`CategoricalOneHotInputTransform`を使い、モデル自身がカテゴリ変換を所有する。
これはnative categorical kernelを使う`MixedSingleTaskGP`と異なるため、「robotorchanのMixedは常にone-hot禁止」という一般化は誤り。
各モデルの元入力軸`cat_dims`と変換後の次元数を区別する。

## Verification と使い分け

[既存Mapping](../../.github/project-records/theory-guide/mapping/models.md)・
[Verification](../../.github/project-records/theory-guide/mapping/verification.md)は履歴を保持する。利用者向け：
[High-dimensional models](../models/high_dimensional.md)、
[MAP-SAAS Notebook](../../examples/notebooks/09_map_saas_and_additive_gp.ipynb)、
[High-dimensional search](../optimization/high_dimensional_search.md)。関連：
[高次元GP](08_high_dimensional_gp.md)、[低次元表現の理論](../theory/18_dimensionality_reduction_gp.md)、
[探索strategyの理論](../theory/20_high_dimensional_search.md)。

## 追加：Fully Bayesian SAASとVariational GPの学習契約

src/robotorchan/models/high_dimensional/fully_bayesian.pyのSaasFullyBayesianSingleTaskGPと
SaasFullyBayesianMultiTaskGPはBoTorchの同名モデルを継承し、supports_mll=Falseとしている。
つまりmodel.make_mll()によりExact GP型の学習目的を組み立てる経路ではない。推論はBoTorchのfit_fully_bayesian_model_nutsを使う。

$$
p(f_*\mid\mathcal D)=\int p(f_*\mid\theta,\mathcal D)\,p(\theta\mid\mathcal D)\,d\theta
$$

と表されるhyperparameter uncertaintyの周辺化を、MCMC sampleによって近似する。
これに対してMAP-SAASでは点推定されたhyperparameterの条件付きposteriorを利用し、
Variational GPではinducing variablesの近似分布に対するELBOを最適化する。いずれも同じ「GP」だが、学習目的・不確実性の意味・計算コストは同じではない。

### 実装との対応

この章に対応する公開実装の契約は次のとおりです。

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

### 公開モデルと学習経路

- **Fully Bayesian**
  モデル：`SaasFullyBayesianSingleTaskGP`, `SaasFullyBayesianMultiTaskGP`
  学習と座標の注意：NUTS。`supports_mll=False`

- **Mixed Fully Bayesian**
  モデル：`MixedSaasFullyBayesianSingleTaskGP`, `MixedSaasFullyBayesianMultiTaskGP`
  学習と座標の注意：model-owned one-hot encoding

- **MAP-SAAS**
  モデル：`AdditiveMapSaasSingleTaskGP`, `EnsembleMapSaasSingleTaskGP`
  学習と座標の注意：Exact GP MLLの学習契約

- **Mixed MAP-SAAS**
  モデル：`MixedAdditiveMapSaasSingleTaskGP`, `MixedEnsembleMapSaasSingleTaskGP`
  学習と座標の注意：raw入力とencoded geometryを分ける

- **Orthogonal additive**
  モデル：`OrthogonalAdditiveGP`, `MixedOrthogonalAdditiveGP`
  学習と座標の注意：入力componentの加法分解

- **Embedded-space**
  モデル：`ALEBOGP`
  学習と座標の注意：fixed-noise single-output。`train_Yvar`が必須

Fully Bayesianモデルの`fit()`はNUTSへ委譲する。直接呼ぶ場合はBoTorchの`fit_fully_bayesian_model_nuts`を使う。
`fit_gpytorch_mll(model.make_mll())`をFully Bayesian SAASへ適用しない。MultiTask版のtask列はlong-formatの構造列であり、
通常の入力特徴としてtask ID自体をsparse selectionするという意味ではない。

Mixed SAASのraw APIには元のカテゴリ値を渡すが、疎性のgeometryはencoded numeric space上にある。
MAP-SAASではNUTSによるhyperparameter integrationを行わず、AdditiveとEnsembleのposterior semanticsも個別に確認する。
`AdditiveMapSaasSingleTaskGP`の名称を、Orthogonal Additive Kernelの入力component分解と同一視しない。
MAP-SAASの`ExactGPModelMixin`、表現削減の`ReducedGP`、観測側の外れ値を扱う`RobustRelevancePursuitSingleTaskGP`も役割が異なる。

ALEBOのmetric uncertaintyでは、Laplace / samplingによるmetric sampleごとのposteriorをGaussianへmoment
matchingする経路がある。`ALEBOGP`の予測と`ALEBOStrategy`の候補探索は別の責務である。

## 共通mixinと専用の推論入口

[fully_bayesian.py](../../src/robotorchan/models/high_dimensional/fully_bayesian.py)の
SAAS wrapperは`SupervisedTrainingDataMixin`・`ModelTrainingMixin`とBoTorchのSAASモデルを継承します。
`fit()`は上流のNUTS fittingへ委譲し、`make_mll()`によるExact GP目的を構成しません。MCMC sampleから予測分布を作る処理もBoTorch側が担当します。

[map_saas.py](../../src/robotorchan/models/high_dimensional/map_saas.py)は
MAP-SAASの上流モデルへraw保持とMLL契約を加えます。同じSAASという名称でも、目的のモード推定とハイパーパラメータ周辺化の計算経路は異なります。
Mixed版のカテゴリ変換はモデル自身が所有し、公開入力とencoded内部空間を接続します。
