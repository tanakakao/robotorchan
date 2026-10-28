# Gaussian Process：数式が実装される経路を読む

## この章で学ぶこと

[GPのTheory](../theory/02_gaussian_process.md)で学んだ事前分布、観測への条件付け、周辺尤度が、
どのクラスとメソッドで計算されるかを追います。
対象は単一タスク・単一出力のGaussian観測モデルです。
数式の導出はTheoryの2.4・2.6・2.11・2.12節を参照してください。
ここでは、その構成要素と実際の計算処理を対応付けます。

処理は、確率モデルの構成、学習目的によるパラメータ更新、
固定したパラメータのもとでの条件付けに分かれます。
robotorchanはデータ保持と学習の共通入口を追加し、
GPの確率モデルと推論はBoTorch・GPyTorchへ委譲します。

## 1. 数式に対応するオブジェクト

Theoryと同じく、学習入力を $X$、観測を $y$、予測入力を $X_*$、
平均を $m$、カーネルを $k$、ノイズを加えた訓練共分散を $K_y$ と書きます。
次の対応は、入出力変換後のモデル内部の座標と応答スケールで読みます。

| Theory上の構成要素 | 計算を担当する実装 |
| --- | --- |
| 事前平均 $m(X)$ | `SingleTaskGP.forward()`が`mean_module`を評価 |
| 事前共分散 $K_{XX}$・$K_{*X}$・$K_{**}$ | `covar_module`が入力間のカーネルを評価 |
| 観測共分散 $K_y$ | Gaussian likelihoodが潜在共分散へ観測ノイズを加算 |
| 周辺対数尤度 | `ExactMarginalLogLikelihood.forward()`と`MultivariateNormal.log_prob()` |
| 観測残差への $K_y^{-1}$ の作用 | `DefaultPredictionStrategy._mean_cache()`内の`solve()` |
| 条件付き平均・共分散 | `exact_predictive_mean()`・`exact_predictive_covar()` |
| 元の応答スケールの分布 | `Standardize.untransform_posterior()` |

平均とカーネルは事前分布を作り、likelihoodが観測モデルを加えます。
条件付き平均・共分散を求めるのは、それらを利用するGPyTorchの予測処理です。
`posterior()`という入口だけを見ても、この責務の違いは分かりません。
以下で、モデルの構成から条件付けまでを順に追います。

## 2. クラス構造とrobotorchanの拡張

入口の[SingleTaskGP][wrapper]は、
`SingleTaskGP(FittableExactGPModelMixin, BoTorchSingleTaskGP)`と定義されています。
ここで`BoTorchSingleTaskGP`は、BoTorchの`SingleTaskGP`のimport時の別名です。
robotorchan側では`forward()`と`posterior()`を上書きしていません。

継承関係のうち、本章の処理に関係する部分は次のとおりです。

| クラス | 継承先と役割 |
| --- | --- |
| `FittableExactGPModelMixin` | `ExactGPModelMixin`を拡張し、学習入口を追加 |
| `ExactGPModelMixin` | `SupervisedTrainingDataMixin`・`ModelTrainingMixin`を継承し、MLLを構成 |
| `SupervisedTrainingDataMixin` | `RawDataMixin`を継承し、入力・応答・観測分散の保持を共通化 |
| BoTorchの`SingleTaskGP` | `BatchedMultiOutputGPyTorchModel`・GPyTorchの`ExactGP`などを継承 |

[共通mixin][base]は、データ保持と学習への接続を提供します。
事前平均・カーネル・likelihoodの構成は[BoTorchモデル][b-gp]、
条件付けは[GPyTorchのモデル群][g-models]が担当します。
この継承によって、robotorchanが独自のGP推論式を持たずに共通機能を追加できます。

### 元データと推論用データを分ける

コンストラクタは、親モデルを初期化する前に`train_X`・`train_Y`・
任意の`train_Yvar`を`detach().clone()`します。
親へは元の引数を渡し、初期化後にコピーを`_store_supervised_training_data()`へ渡します。
親の変換処理による変更から、呼び出し時点のデータを分離するためです。

`RawDataMixin._store_raw_tensor()`は、コピーを非永続bufferとして登録します。
そのためdtype・deviceの変更には追従しますが、`state_dict`には含まれません。
`raw_train_X`・`raw_train_Y`・`raw_train_Yvar`は、この保存先への参照です。

これらは、`condition_on_observations()`や`fantasize()`の後の
最新の訓練状態へ書き換えるためのデータではありません。
条件付けに使われる状態は、上流モデルの`train_inputs`・`train_targets`です。
元データの来歴と、現在の推論状態を区別する設計です。

## 3. 事前分布と観測モデルを組み立てる

BoTorchの`SingleTaskGP.__init__()`は、入力形状の検証、入出力変換、
likelihood・平均・カーネルの設定を行います。
この初期化の中で、`ExactGP`へ訓練入力・訓練応答・likelihoodを登録します。
モデルの構成だけでは、パラメータの最適化は実行されません。

### 平均・カーネル・ノイズの構成

BoTorch 0.18.1の既定平均は`ConstantMean`です。
既定カーネルは`get_covar_module_with_dim_scaled_prior()`が作るARD RBFで、
長さ尺度に入力次元に応じた事前分布と下限制約を置きます。
常に独立した`ScaleKernel`が付く構成ではありません。
これらの部品は、親へ渡された`mean_module`・`covar_module`によって置き換わります。

likelihoodが指定されていない場合、`train_Yvar`がなければ
`get_gaussian_likelihood_with_lognormal_prior()`が作る`GaussianLikelihood`を使い、
観測ノイズ分散も学習対象になります。
`train_Yvar`があれば`FixedNoiseGaussianLikelihood`へ既知分散を渡します。
この量は標準偏差ではなく分散であり、Theory 2.4節のノイズ共分散に対応します。
部品生成の詳細は[BoTorchのGP部品][b-utils]の`gpytorch_modules.py`にあります。

### 変換後の座標で計算し、元の尺度へ戻す

既定の出力変換は[Standardize][b-transforms]です。
実装は`models/transforms/outcome.py`にあります。
`Standardize.forward()`は、観測を中心化して標準偏差で割り、
既知の観測分散があれば標準偏差の二乗で割ります。
平均と分散では変換の係数が異なるため、応答だけを変えて分散を放置しません。
観測が一つの場合や標準偏差が小さすぎる場合は、尺度に1を使う処理もあります。

予測の最後に`untransform_posterior()`が平均と共分散を逆変換します。
このため、モデル内部の訓練応答と、外へ返る予測分布の応答スケールは異なり得ます。
rawデータのコピーを別に保持することは、この違いを追ううえでも意味があります。

入力変換は、出力標準化とは別の部品です。
`input_transform`がなければ、入力を自動で単位箱へ正規化する処理は追加されません。
`SingleTaskGP.forward()`は学習時に入力変換を適用し、
予測時の変換は`posterior()`側が担当します。

## 4. 周辺尤度の数式から学習処理へ

`ExactGPModelMixin.make_mll()`は、
`ExactMarginalLogLikelihood(self.likelihood, self)`を構成します。
このメソッドは、数式の計算や最適化を自ら実行するものではありません。

GPyTorchの`mlls/exact_marginal_log_likelihood.py`にある
[ExactMarginalLogLikelihood.forward()][g-mll]は、潜在関数の分布へ
likelihoodを適用し、観測の周辺分布を作って`log_prob(target)`を呼びます。
Theory 2.11節で潜在関数を積分して得たGaussian分布が、この観測分布に対応します。
学習モードの`ExactGP.__call__()`は訓練入力に対する事前分布を返すため、
学習目的を計算するときに事後分布を再びlikelihoodへ渡す構造ではありません。

`distributions/multivariate_normal.py`の
[MultivariateNormal.log_prob()][g-dist]の線形演算子を使う経路では、
観測残差を右辺として`inv_quad_logdet()`を呼びます。
これがTheoryの二次形式と対数行列式を計算し、正規化定数と合わせる箇所です。
この演算を提供するのは[linear_operator][lin-op]であり、
実装は`operators/_linear_operator.py`にあります。
robotorchanが逆行列や行列式を直接計算しているわけではありません。

その後、MLLの`_add_other_terms()`が登録済みの追加損失と対数事前分布を加え、
`forward()`がデータ数で割ります。
したがって、その返り値はTheoryの対数尤度の総和そのものとは異なります。
既定の長さ尺度・ノイズ事前分布を使えば、学習目的にはその対数確率も含まれます。

`FittableExactGPModelMixin.fit()`は、`supports_fit`を確認してから
`fit_gpytorch_mll(self.make_mll(), optimizer_kwargs=...)`へ委譲します。
[BoTorchのfitting][b-fit]は学習目的の型などに応じて処理を選び、
標準Exact MLLでは既定のSciPy経路で負のMLLを最小化します。
警告や失敗に応じた再試行も上流の責務です。
robotorchanの`fit()`は返り値を保持せず、`None`を返します。

学習入口は意図的にopt-inです。
`FittableExactGPModelMixin.__init_subclass__()`は、subclassが
`supports_fit`を明示しなければ`False`へ戻します。
標準の`SingleTaskGP`は`True`を宣言しますが、特殊な学習目的を持つ派生モデルへ
Exact MLLによる学習を無条件に引き継がせない設計です。
`supports_mll`と`supports_fit`は、それぞれ目的の構成と学習入口の利用可否を表します。

## 5. 事前分布から条件付き分布へ進む呼び出し経路

予測の入口は、継承した`BatchedMultiOutputGPyTorchModel.posterior()`です。
[BoTorchのGPyTorch接続部][b-gpt]とGPyTorchの`exact_gp.py`を続けて読むと、
次の順序で処理が進みます。

```text
BatchedMultiOutputGPyTorchModel.posterior()
  → 評価モードへ切り替え、予測入力を変換
  → self(X)：ExactGP.__call__()による観測への条件付け
      → 訓練側のprediction strategyを必要時に構成
      → 事前平均と訓練・予測間の共分散を構成
      → exact_prediction()で条件付き平均・共分散を計算
  → _apply_noise()で必要な観測ノイズを加算
  → GPyTorchPosteriorへ格納
  → 出力変換の逆変換、任意のposterior_transform
```

### forward()だけでは条件付けにならない理由

BoTorchの`SingleTaskGP.forward()`は、`mean_module(x)`と`covar_module(x)`から
`MultivariateNormal`を作ります。この処理だけなら事前分布です。
評価時の`self(X)`では、GPyTorchの`ExactGP.__call__()`が介在し、
訓練データに条件付けた分布を返します。
同じモデルでも、`forward()`の直接評価と`posterior()`を同じ処理とはみなしません。

`ExactGP.__call__()`は、訓練側のキャッシュがなければprediction strategyを作り、
学習・予測間の共分散を`exact_prediction()`へ渡します。
標準経路の`DefaultPredictionStrategy`は、
GPyTorchの`models/exact_prediction_strategies.py`にあります。

### 平均と共分散のどの項を計算するか

`_mean_cache()`はlikelihoodを加えた訓練分布を作り、
中心化した訓練応答を右辺とする`solve()`の結果を保存します。
これはTheory 2.6節の $K_y^{-1}(y-m(X))$ に対応します。
`exact_predictive_mean()`は、そのキャッシュへ $K_{*X}$ を作用させ、
予測点の事前平均を加えます。
訓練データだけで決まる項を再利用することで、予測点ごとの解き直しを避けます。

`exact_predictive_covar()`は、事前共分散 $K_{**}$ から条件付けの補正を引きます。
`fast_pred_var`が無効な経路では、訓練・予測間の共分散を右辺にして`solve()`し、
その結果と逆向きの共分散を掛けて補正を構成します。
有効な経路では、キャッシュされた逆ルート分解に基づいて補正を計算します。

この線形方程式と共分散補正が、Theoryの予測式を実現する処理です。
条件付けの式をrobotorchan側で再実装しているわけではありません。

### 潜在関数とノイズ付き観測の分布を分ける

`posterior()`内の`_apply_noise()`は、潜在関数の条件付き分布を受け取り、
必要な場合にlikelihoodで観測ノイズを加えます。
これはTheory 2.8節の、潜在関数の分散と将来観測の分散の違いに対応します。
既知の訓練分散を持つだけで、未知の場所のノイズ分散が推定されるわけではありません。

BoTorch 0.18.1のfixed-noise経路では、ノイズ追加を`True`とした場合に
訓練分散の平均を使います。Tensorが指定された場合は、それをlikelihoodへ渡します。
この処理は出力の逆変換より前なので、Tensorはモデル内部の応答スケールの分散です。
引数・形状の詳細は上流の`posterior()`の定義を参照してください。

最後に`GPyTorchPosterior`が同時分布を保持します。
その`variance`は共分散の対角成分であり、候補間の依存をすべて表す量ではありません。
内部では線形演算子による共分散表現を保てる一方、
`distribution.covariance_matrix`を取り出すと密行列の実体化が必要になる場合があります。

## 6. 数値計算の設計と適用範囲

Theory 2.12節では、小規模な密行列に対するCholesky分解を説明しています。
実装の「Exact GP」は推論モデルの区分であり、
全ての行列計算が常に密行列Choleskyに固定されるという意味ではありません。
GPyTorchとlinear_operatorの設定・行列サイズによって、
`solve()`・`inv_quad_logdet()`・逆ルート分解の計算経路が変わります。

BoTorchの`gpt_posterior_settings()`は、`fast_pred_var`が既定の設定である場合に
それを有効化します。したがって、予測共分散は逆ルートのキャッシュを使う経路も含めて
読む必要があります。数値安定化のjitterは、測定過程を表す観測ノイズとは目的が異なります。

このモデルは、点推定したハイパーパラメータを固定して条件付けます。
パラメータの不確実性を積分するFully Bayesianモデルとは異なります。
また、このExact MLLをVariational GPのELBOやDeepGPの学習目的へ一律に適用できません。
学習が終了したことも、大域最適性や予測の較正を保証するものではありません。

## 7. 実装の担当範囲と次の参照先

| 区分 | 本章での担当範囲 |
| --- | --- |
| Delegated | BoTorch・GPyTorchによる確率モデル、変換、MLL計算、最適化、条件付け |
| Extended | `SingleTaskGP`への元データ保持と学習機能の追加 |
| Implemented | raw buffer管理、MLLオブジェクトの構成、fitへの接続と対応検査 |
| Approximated | 上流の設定に応じた線形代数・予測分散の数値計算 |

数式との接点は、事前分布を作る`forward()`、観測分布の密度を計算するMLL、
訓練側の線形方程式を解くprediction strategyの三か所です。
これらを区別すると、モデルを包むrobotorchanの拡張と、上流が担うGP推論を追えます。

共分散部品の詳細は[KernelのGuide][kernel]、
予測分布を意思決定へ接続する構造は[BOのGuide][bo]へ進んでください。
構築・学習・予測と可視化の使用例は[SingleTaskGP Notebook][notebook]、
数値照合は[条件付き分布の参照テスト][reference-test]、
学習入口とデータ保持の境界条件は[学習契約][fit-test]・[データ来歴][raw-test]を参照できます。

上流コードへのリンクは、BoTorch 0.18.1・GPyTorch 1.15.2・
linear_operator 0.6.1の実装を指します。
ディレクトリへのリンクでは、本文中のファイル名とメソッド名を手掛かりに読めます。

[wrapper]: ../../src/robotorchan/models/standard/single_task.py
[base]: ../../src/robotorchan/models/base.py
[b-gp]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/models/gp_regression.py
[b-gpt]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/models/gpytorch.py
[b-utils]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models/utils
[b-transforms]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models/transforms
[b-fit]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/fit.py
[g-models]: https://github.com/cornellius-gp/gpytorch/tree/v1.15.2/gpytorch/models
[g-mll]: https://github.com/cornellius-gp/gpytorch/tree/v1.15.2/gpytorch/mlls
[g-dist]: https://github.com/cornellius-gp/gpytorch/tree/v1.15.2/gpytorch/distributions
[lin-op]: https://github.com/cornellius-gp/linear_operator/tree/v0.6.1/linear_operator/operators
[kernel]: 03_kernel.md
[bo]: 01_bayesian_optimization.md
[notebook]: ../../examples/notebooks/01_single_task_gp.ipynb
[reference-test]: ../../tests/models/standard/test_phase7_exact_gp_reference.py
[fit-test]: ../../tests/models/contracts/test_exact_gp_fit.py
[raw-test]: ../../tests/models/contracts/test_raw_data_provenance.py
