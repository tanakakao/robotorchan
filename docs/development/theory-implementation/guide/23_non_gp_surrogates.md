# Non-GP Surrogates：経験分布をBoTorchへ接続する

## 1. GP以外のモデルがBayesian Optimizationに参加する条件

BOに必要なのは必ずしもGPではなく、獲得関数が消費できる予測値や**sampleable predictive representation**である。[理論第23章](../../../theory/23_non_gp_surrogates.md)では木・boosting・NGBoostにおけるuncertaintyの意味を区別する。

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

上記は設計意図を示す例で、今回の作業で実行したものではない。具体的なconstructorとposterior contractは実装・Notebookで検収する。

## 5. 制限と検証

MC qEI/qEHVIは条件を満たす場合の候補であり、全獲得関数が`EnsemblePosterior`を解釈できるという主張ではない。Gaussian analytic EIは一般に経験ensembleの仮定に合わない。tree disagreementはcalibrated epistemic posteriorではなく、NGBoostの予測分散からepistemic/aleatoricを自動的に分離できるわけでもない。

詳細：[Non-GPモデル](../../../models/non_gp.md)、[Non-GP Notebook](../../../../examples/notebooks/25_non_gp_surrogates.ipynb)、[Acquisition Foundation](04_acquisition_function.md)、[旧Mapping](../models.md)、[検証記録](../verification.md)。

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
