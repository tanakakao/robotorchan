# Non-GP Surrogates：経験分布をBoTorchへ接続する

## 1. GP以外のモデルがBayesian Optimizationに参加する条件

BOに必要なのは必ずしもGPではなく、獲得関数が消費できる予測値や**sampleable predictive representation**である。[理論第23章](../theory/23_non_gp_surrogates.md)では木・boosting・NGBoostにおけるuncertaintyの意味を区別する。

同じ`posterior()`インターフェースでも、**GPの条件付き確率posterior**と**木ごとの予測の経験分布**は数学的に同義ではない。

## 2. モデルとposteriorの実装対応

| Family | 公開API | サンプルの意味 |
| --- | --- | --- |
| Random Forest | `RandomForestSurrogate` | fitted treeごとの予測 |
| Extra Trees | `ExtraTreesSurrogate` | 強い分割ランダム性を含むtreeごとの予測 |
| Gradient Boosting | `GradientBoostingSurrogate` | bootstrapごとに学習した**完全なboosting model**の予測 |
| Histogram Boosting | `HistGradientBoostingSurrogate` | bootstrapごとの完全なhistogram boosting model |
| NGBoost | `NGBoostSurrogate` | 学習された条件付き予測分布からのサンプル |

**Extended:** robotorchanは`src/robotorchan/models/non_gp/`にあるsklearnベースのモデル出力をBoTorchの`EnsemblePosterior`へ接続する。`fit()`で明示的に学習し、GPのMLLは使わない（`supports_mll=False`）。NGBoostはオプション依存で、初期契約は単一出力の数値回帰である。

重要な誤解として、boostingの逐次stage `h_1,h_2,…` を独立posterior sampleにしてはいけない。stageは残差を順番に修正する加法項であり、robotorchanはbootstrap単位の**完成したモデル**をmemberにしている。

## 3. 予測から候補探索まで

```text
train_X / train_Y → non-GP surrogate.fit()
                             ↓
           model.posterior(candidate_X)
                             ↓
                  EnsemblePosterior
                             ↓
               compatible MC acquisition
                             ↓
               gradient-free search strategy
```

posteriorの軸は概念的に`… × ensemble_samples × q × outputs`であり、sample軸と`q`軸の取り違えはacquisitionの意味を壊す。sklearn経由のcandidate評価は一般にCPU/numpyを介しており、入力へのPyTorch gradientは期待できない。そのため`optimize_acqf`の勾配ベース経路ではなく、`TreeEnsembleSearchStrategy`等の勾配不要探索を検討する。

## 4. 公開APIの入口例

```python
import torch
from robotorchan.models import RandomForestSurrogate

train_X = torch.rand(20, 3, dtype=torch.double)
train_Y = train_X[:, :1].square()
model = RandomForestSurrogate(train_X=train_X, train_Y=train_Y)
model.fit()
posterior = model.posterior(train_X[:2])
print(posterior.mean.shape)
```

上記は設計意図を示す例で、すべての条件で実行を保証するものではない。具体的なconstructorとposterior contractは実装・Notebookで検収する。

## 5. 制限と検証

MC qEI/qEHVIは条件を満たす場合の候補であり、全獲得関数が`EnsemblePosterior`を解釈できるという主張ではない。Gaussian analytic EIは一般に経験ensembleの仮定に合わない。tree disagreementはcalibrated epistemic posteriorではなく、NGBoostの予測分散からepistemic/aleatoricを自動的に分離できるわけでもない。

詳細：[Non-GPモデル](../models/non_gp.md)、[Non-GP Notebook](../../examples/notebooks/25_non_gp_surrogates.ipynb)、[Acquisition Foundation](04_acquisition_function.md)、[旧Mapping](../../.github/project-records/theory-guide/mapping/models.md)、[検証記録](../../.github/project-records/theory-guide/mapping/verification.md)。

## 6. EnsemblePosteriorと予測分布の数理的な違い

Random Forestやbootstrap boostingで構成する経験分布は、学習済みmemberの予測\(\hat f_s(x)\)を有限個集めたものとして理解する。経験的な予測平均は次式となる。

$$
\widehat\mu(x)=\frac1S\sum_{s=1}^S\hat f_s(x).
$$

member間の分散は有限ensembleのdisagreementであり、確率的GPのposterior varianceや測定ノイズの分散そのものではない。src/robotorchan/models/non_gp/posterior.pyのmake_ensemble_posteriorはBoTorch EnsemblePosteriorにvaluesを渡すが、値の意味をGaussian posteriorへ変更する処理ではない。

| family | member/sampleの生成 | 契約と拒否条件 |
| --- | --- | --- |
| RandomForestSurrogate | RandomForestRegressor.estimators_の各treeの予測 | train_Yは単一出力のみ。fit前posteriorを拒否。observation_noise=Trueとoutput_indicesが[0]以外を拒否 |
| ExtraTreesSurrogate | tree-level predictions | 木ごとのランダム性はBayesian posterior sampleの較正を保証しない |
| GradientBoostingSurrogate / HistGradientBoostingSurrogate | BootstrapEnsembleSurrogateが各resampleで**完成したestimator**を学習 | n_members>=2。複数出力ではMultiOutputRegressor等を使用。boosting stage単位を独立sampleとしない |
| NGBoostSurrogate | pred_dist(X)のNormal(loc,scale)をGaussianDistributionPosteriorへ変換 | 単一出力。ngboost extraが必要。observation_noise=Trueは拒否 |

## 7. MC sample軸・候補軸とgradientの互換性

BoTorch EnsemblePosteriorへ渡す予測tensorのshapeは`batch_shape + [s,q,m]`であり、`s`はensemble member数、`q`は候補点数、`m`は出力数である。src/robotorchan/models/non_gp/bootstrap.pyでは、memberごとの予測をstackしてからこの位置へ並べ替える。値のshapeが適合していても、全てのMC acquisitionやposterior transformと互換になることは保証されない。

sklearnによるfit/predictはnumpy/CPUへ変換される。学習済み予測をPyTorchへ戻しても通常、候補Xからposterior値への微分は存在しない。acquisitionの値評価と、勾配ベースoptimizerでの最適化を分けて考え、適切なgradient-free candidate searchを選択する。

NGBoostSurrogateはNormalのlocとscale^2から構築した独立Gaussian predictive distributionを返す。src/robotorchan/models/non_gp/distribution_posterior.pyのGaussianDistributionPosteriorはrsampleとrsample_from_base_samplesを実装する一方、これはGaussian GPの点間共分散行列を提供するものではない。NGBoostの予測scaleはaleatoric/epistemicの分解を自動的には与えない。

**検証境界：** 今回はsrc/robotorchan/models/non_gp/{random_forest,bootstrap,gradient_boosting,posterior,ngboost,distribution_posterior}.pyの静的実装契約を照合した。実際のfit・posterior数値再実行、MC acquisitionとのend-to-end互換性、uncertainty calibrationの検収は未実施。

## 実装補足：理論本文から分離した対応情報

### 9. 実装上の fitting contract

robotorchan の non-GP surrogate は BoTorch `Model` interface を提供しますが、GP marginal
likelihood を持ちません。constructor で raw training data を保持し、`fit()` を明示的に呼びます。

したがって GP の `make_mll()` / `fit_gpytorch_mll()` contract を適用しません。


### 15. Capability-based compatibility

robotorchan は model / acquisition registry の metadata から static compatibility を判定します。

non-GP model では特に、

- `non_gp`
- `ensemble_posterior`
- `supports_multi_output`
- `supports_posterior_samples`
- `posterior_sampling_type`
- `supports_fantasize`

が重要です。

BoTorch `Model` を継承していることだけから acquisition compatibility を推論しません。

## 理論章から分離した実装上の条件

### 11. Multi-output bootstrap contract

`BootstrapEnsembleSurrogate` は `train_Y` の `n x m` output を扱えます。multi-output の場合も
同一 ensemble member 内では同じ bootstrap row indices を共有します。

ただし output-specific regressors が独立に fit されるため、member alignment があることを
MultiTask GP / Kronecker GP の fully modeled cross-output covariance と同一視しません。

`output_indices` は valid range、non-empty、duplicateなしを要求します。

### 12. Mixed tree inputs の意味

Random Forest / Extra Trees は `cat_dims` の category values を検証します。

ただし sklearn tree backend が raw numeric category codes を native categorical variable として
扱うわけではありません。threshold split は数値コード上で行われます。

したがって

```text
cat_dims validation
!= native categorical split semantics
```

です。

### 13. Observation-noise semantics

現在の empirical tree / boosting surrogate は
`posterior(..., observation_noise=True)` による追加 noise modeling を提供しません。

ensemble spread を observation noise とみなして自動加算することもありません。

これは GP の latent-function posterior と likelihood noise の分離とは異なります。

### 14. IndexSampler と MC acquisition

`validate_non_gp_acquisition` は empirical non-GP model に `MCAcquisitionFunction` を要求します。

explicit sampler が設定される場合は `IndexSampler` を要求します。
これは continuous Gaussian base samples ではなく ensemble member indices を sampling するためです。

qEI / qEHVI など sample-based acquisition と接続できても、すべての acquisition が自動的に
互換になるわけではありません。

joint Gaussian、fantasize、multi-output、MultiFidelity などの追加要件は別途満たす必要があります。

### 16. TreeEnsembleSearchStrategy

sklearn tree / boosting prediction は candidate input に関して piecewise / non-differentiable で、
CPU / NumPy boundary も通ります。

robotorchan は `TreeEnsembleSearchStrategy` により gradient-free random candidate search を
提供します。

この strategy は、

- continuous bounds
- integer dimensions
- categorical allowed values
- non-GP MC acquisition validation

を扱います。

integer dimensions は feasible integer range、categorical dimensions は明示された allowed values
から sampling します。

これは GP の gradient-based `optimize_acqf` / `optimize_acqf_mixed` route とは別です。

### 18. NGBoost の現在の実装境界

現在の `NGBoostSurrogate` は、

- single-output
- Gaussian `Normal` predictive law
- explicit `fit()`
- sampleable `GaussianDistributionPosterior`
- additional observation noise unsupported

という contract です。

NGBoost library 一般が複数 distribution を扱えることと、robotorchan adapter が現在公開している
capability は区別します。

### 23. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- non-GP model は BoTorch interface を持つが GP ではない
- empirical ensemble は `EnsemblePosterior` を使う
- forest系は各tree prediction、boosting系は各complete bootstrap modelをmemberにする
- Random Forest / Extra Trees は現在 single-output
- `cat_dims` validation は native categorical tree semantics を意味しない
- boosting stages 自体を posterior members にしない
- Gradient / HistGradient Boosting は complete bootstrap models を members にする
- bootstrap boosting は multi-output を扱える
- shared bootstrap index は full Bayesian cross-output covariance を意味しない
- empirical ensemble MC acquisition の explicit sampler は `IndexSampler`
- NGBoost Gaussian predictive posterior は Gaussian MC sampler を使う
- analytic Gaussian acquisition を empirical ensemble に適用しない
- TreeEnsembleSearchStrategy は gradient-free search を行う
- integer / categorical candidate sampling は search strategy が扱う
- capability registry で acquisition requirements を確認する
- 現NGBoost adapterは single-output Gaussian predictive law
- NGBoost posteriorはcandidate間GP covarianceを構築しない
- NGBoost variance を GP posterior variance と同一視しない
- BALD compatibility を predictive variance だけから推論しない

#### 1. Posterior interface without a Gaussian process：実装上の補足

BO には予測分布または sample 可能な表現が必要で、すべての surrogate が GP である必要はありません。経験 ensemble の予測を f_s(X) とすると、実装では次のメンバー集合を表します。

#### 2. Random Forest and Extra Trees：実装上の補足

現行forest surrogateでは、sklearn ensemble内部の各tree predictionをempirical memberとして使います。
これはcomplete modelを複数回bootstrap fitするboosting surrogateとは異なります。

#### 3. Why boosting stages are not posterior samples：実装上の補足

robotorchan は bootstrap データ D_s ごとに完成した boosting モデル F_M^(s) を学習し、次を経験予測 sample として使います。

#### 5. Acquisition functions：実装上の補足

Compatible MC acquisitions can consume a sampleable posterior through the BoTorch interface.
qEIやqEHVIも候補ですが、output数などacquisition固有のrequirementsを満たす必要があります。

#### 5. Acquisition functions：実装上の補足

sklearn の tree・boosting 予測は入力に対し区分的・非微分で、CPU・NumPy を経由します。そのため勾配法の `optimize_acqf` ではなく微分不要の探索を使います。

#### 7. Distributional surrogate: NGBoost：実装上の補足

`NGBoostSurrogate` は stage をメンバーとせず条件付き分布を予測します。現在の adapter は Gaussian NGBoost の平均・分散・sample を BoTorch posterior へ公開します。

#### 19. NGBoost uncertainty：実装上の補足

現在の adapter は

#### 22. candidate間のjoint uncertainty：実装上の補足

`GaussianDistributionPosterior` は各candidateのmean / varianceからGaussian sampleを生成します。
現行NGBoost adapterはGPのようなcandidate間posterior covariance matrixを構築しません。

#### 24. この章で覚えておくこと：実装上の補足

- BoTorch `Posterior` interfaceとBayesian posteriorという統計的意味は同一ではない
- forest、bootstrap boosting、NGBoostではuncertaintyの生成機構が異なる
- boosting stageそのものをposterior sampleとして扱わない
- ensemble disagreementをobservation noiseやcalibrated epistemic uncertaintyと決めつけない
- NGBoostのGaussian marginalだけからcandidate間joint Gaussian covarianceを仮定しない
- sampleable posteriorでも全acquisitionへ自動対応するわけではない
- tree系candidate optimizationはgradient-free searchと組み合わせる

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。


## 10. HistGradientBoostingSurrogate

`HistGradientBoostingSurrogate` も complete bootstrap models を predictive members とします。
boosting stage の spread を posterior とするのではなく、独立 bootstrap resamples 上で fit した
complete `HistGradientBoostingRegressor` の predictions を使います。
