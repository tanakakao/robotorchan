# Preference Learning：比較尤度からLaplace近似への実装経路

## この章の対象

[選好学習のTheory](../theory/10_preference_learning.md)で説明した比較確率、
潜在utilityのMAP推定、Laplace近似が、どの処理で計算されるかを追います。robotorchanの`PairwiseGP`はデータ保持と学習入口を拡張し、
比較尤度・推論・条件付けはBoTorchへ委譲します。上流の内部処理はBoTorch 0.18.1のソースに基づきます。

## Theoryの構成要素と処理の対応

Theoryの潜在utilityを $u$、訓練点の共分散を $K$ とします。比較尤度の負の対数Hessianを $C$ とすると、
Laplace近似の共分散は $(K^{-1}+C)^{-1}$ です。式の導出はTheoryの10.6・10.7・10.11・10.13・10.14節を参照してください。

| 理論上の構成要素 | 実装するクラス・メソッド | 計算の担当 |
| --- | --- | --- |
| 勝者と敗者のutility差 | `PairwiseGP.set_train_data()`の比較行列`D` | BoTorch |
| probit比較確率 | `PairwiseProbitLikelihood._calc_z()`・`p()` | BoTorch |
| 負の対数尤度の勾配・Hessian | `negative_log_gradient_sum()`・`negative_log_hessian_sum()` | BoTorch |
| utilityのMAP推定 | `PairwiseGP._update()`・`_grad_posterior_f()` | BoTorch・SciPy |
| Laplace近似共分散 | `PairwiseGP.forward()`の学習モード | BoTorch・PyTorch |
| 近似対数周辺尤度 | `PairwiseLaplaceMarginalLogLikelihood.forward()` | BoTorch |
| 新しい点での予測 | `PairwiseGP.forward()`の評価モード・`posterior()` | BoTorch |
| 元の比較データの保持 | `RawDataMixin._store_raw_tensor()` | robotorchan |
| 学習目的の構成と最適化への接続 | wrapperの`make_mll()`・`fit()` | robotorchanからBoTorchへ委譲 |

表の`PairwiseGP`の内部メソッドは、特記したwrapper部分を除きBoTorchの実装です。robotorchan側に別のLaplaceソルバはありません。

## 継承とデータの来歴

[wrapper][wrapper]の継承は
`PairwiseGP(RawDataMixin, ModelTrainingMixin, BoTorchPairwiseGP)`です。`BoTorchPairwiseGP`はimport時の別名で、
上流クラスは`Model`・`GP`・`FantasizeMixin`を継承します。Gaussian回帰の`ExactGP`を継承する構造とは異なります。

wrapperのコンストラクタは、親を初期化する前に`datapoints`と`comparisons`を`detach().clone()`します。親の初期化後に、
[RawDataMixin][base]がコピーを非永続bufferとして登録します。`raw_datapoints`・`raw_comparisons`は入力の来歴であり、
`raw_train_Y`へ比較indexを詰め替える設計ではありません。

上流の`_consolidate_duplicates()`は近接・重複点を統合し、比較indexを再対応付けます。
このためrawデータと推論用の`datapoints`・`comparisons`はshapeが異なる場合があります。
`train_inputs`・`train_targets`は上流の学習関数へ接続するための状態であり、wrapperが保存したsnapshotとは別です。

## 比較indexを尤度の計算へ変換する

BoTorchの`set_train_data()`は、比較の各行について勝者の列を+1、敗者の列を−1とする行列`D`を構成します。
これにより`D @ utility`が全比較のutility差をまとめて表します。

既定の[PairwiseProbitLikelihood][likelihood]の`_calc_z()`は、その差を $\sqrt{2}$ で割ります。
Theory 10.7節の独立な比較ノイズの差に対応し、上流実装ではノイズ尺度を1に固定しています。`p()`は標準正規CDFを適用します。
Bradley–Terry型のlogistic linkとは別の尤度です。

数値上は`z`を`[-3,3]`へclampし、極端な比較確率での不安定さを抑えます。勾配とHessianは正規密度・CDFに由来するhazard量と`D`から集計します。
これは理想的なprobit式に対する安定化処理であり、clampの存在も数式との対応に含まれます。

既定カーネルはARD RBFを包む`ScaleKernel`で、平均は固定された`ConstantMean`です。比較ノイズ尺度とutility尺度を別々に同定する代わりに、
kernel output scaleで尺度を扱います。`posterior()`の`observation_noise`は上流で無視され、回帰GPと同じノイズ加算にはなりません。

## MAP utilityを解く内部更新

`_update()`は、現在のカーネルで`_update_covar()`を呼んだ後、負の対数事後密度の停留条件をSciPyの`optimize.fsolve()`で解きます。
`_grad_posterior_f()`は比較尤度の勾配に、Cholesky solveで計算する $K^{-1}(u-m)$ を加えます。
`_hess_posterior_f()`はpriorとlikelihoodのHessianを組み合わせます。

初期値には前回解を再利用し、未設定なら勝数に基づく値へ小さな摂動を加えます。SciPy側の解法はCPUで動き、batchは個別に処理されます。
`xtol`・`maxfev`はこの内部解法の停止条件であり、カーネルの学習回数ではありません。

得られたMAP点から`_util_newton_updates()`でPyTorch上のNewton更新を行い、カーネルパラメータへ勾配がつながるutilityを構成します。
比較データに対するMAP推定と、ハイパーパラメータの最適化を分離しつつ接続するためです。`PairwiseGP`ではコンストラクタの親初期化中にもこの内部更新が起こり得ます。
単にオブジェクトを構築するだけの軽いGaussian回帰wrapperとは処理が異なります。

## Laplace共分散と近似周辺尤度

学習モードの`forward()`はMAP点を平均として、共分散を$K-K(I+CK)^{-1}CK$ の形で構成します。
これはTheory 10.14節の $(K^{-1}+C)^{-1}$ の変形です。実装は`torch.linalg.solve()`を使い、Hessianを含む逆行列を式どおりに二重に反転しません。

wrapperの`make_mll()`が作るのは、
[PairwiseLaplaceMarginalLogLikelihood][pairwise]です。
その`forward()`はMAP点での比較log likelihood、priorの二次形式、`I + K @ likelihood_hess`のlog determinantを組み合わせ、
負の対数事後項には0以上へのclampを適用し、登録されたパラメータpriorの対数確率も加えます。したがってGaussian観測を厳密積分した`ExactMarginalLogLikelihood`
とは別の学習目的です。

wrapperの`fit()`は、この目的を`fit_gpytorch_mll()`へ渡します。最適化器の選択は上流のfittingに委譲し、wrapperに独自のパラメータ更新則はありません。

## 予測と近似上の制限

評価モードの`forward()`は、訓練側のMAP utilityとHessianを再利用します。予測平均は、
Cholesky solveで得る $K^{-1}(\hat u-m)$ へ交差共分散を掛け、予測点のprior meanを加えます。
共分散補正は`hlcov_eye = C @ K + I`をキャッシュし、`solve(hlcov_eye, C @ K_X*)`を使って構成します。比較グラフ由来の`C`自体が特異でも、
その逆を直接必要としない表現です。

`posterior()`はこの分布を`GPyTorchPosterior`に包みます。返るGaussianは非Gaussianな比較posteriorのLaplace近似であり、
潜在utilityを厳密にGaussianとして条件付けた結果ではありません。共分散のCholesky処理にはjitterによる安定化も入ります。

単一utilityの表現では、評価者間の差や循環選好を独立した構造としてモデル化しません。EUBOなどの次の比較対象の選択、BOPEのoutcomeモデルと選好モデルの分離は、
このwrapperの推論とは別の層です。獲得関数側の接続は[Lookaheadの実装](acquisition/06_lookahead.md)も参照してください。

## 関連資料

実行手順は[Preference Notebook][notebook]、引数と利用方法は[モデルのAPI資料](../models/preference_learning.md)、
境界条件の確認は[Pairwiseテスト][tests]を参照してください。理論の参考文献は対応するTheory章にまとめています。

[wrapper]: ../../src/robotorchan/models/preference/pairwise.py
[base]: ../../src/robotorchan/models/base.py
[pairwise]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/models/pairwise_gp.py
[likelihood]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/models/likelihoods/pairwise.py
[notebook]: ../../examples/notebooks/07_pairwise_gp.ipynb
[tests]: ../../tests/models/preference/test_pairwise_gp.py
