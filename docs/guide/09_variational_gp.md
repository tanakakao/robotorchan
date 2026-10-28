# Variational GP：誘導変数から予測分布とELBOを計算する

## この章で学ぶこと

[Variational GPのTheory](../theory/09_variational_gp.md)の9.7〜9.10・9.20・9.31節を、
誘導点、変分分布、変分戦略、ELBOの計算へ対応付けます。
入力次元dの大きさではなく、観測数nに対する近似推論を主に扱います。

Exact GPの観測共分散をそのまま条件付ける処理と、
誘導変数を介した変分分布から予測を作る処理は異なります。
同じ`posterior()`という入口でも、内側で何を計算するかを分けて読みます。

## 1. 外側のモデルと内側のGP

[models/standard/variational.py][local]の`SingleTaskVariationalGP`は、
`RawDataMixin`・`ModelTrainingMixin`・BoTorchの`SingleTaskVariationalGP`を継承します。
robotorchanは変換前のデータを保持し、`make_mll()`を追加します。
誘導点の配置、変分戦略、予測の主要な処理は上流へ委譲します。

| Theory上の構成要素 | 実装上の配置 |
| --- | --- |
| 誘導入力Z | `model.model.variational_strategy`が保持する誘導点 |
| 変分分布 | 変分戦略が持つ`CholeskyVariationalDistribution`など |
| 事前平均・カーネル | 内側の`_SingleTaskVariationalGP`の部品 |
| 近似分布q(f)の計算 | GPyTorchの`VariationalStrategy` |
| 観測の期待対数尤度 | likelihoodの`expected_log_prob()` |
| KL項 | 変分戦略の`kl_divergence()` |
| 学習目的の構成 | ローカルの`make_mll()`が作る`VariationalELBO` |

BoTorchの[approximate_gp.py][b-models]では、外側のモデルがlikelihoodと変換を持ち、
内側の`_SingleTaskVariationalGP`を`self.model`として保持します。
`make_mll()`が渡すのは、外側のwrapperではなく、この内側のGPです。
内側に変分戦略があるため、ELBOがそこからKLを取得できます。

## 2. 誘導点と変分分布の構成

上流の`_SingleTaskVariationalGP.__init__()`は、誘導点がTensorで渡されればそれを使います。
個数または省略であれば、allocatorに配置を委譲します。
既定のallocatorは`GreedyVarianceReduction`です。
誘導点の初期配置と、その後の位置をパラメータとして更新するかどうかは別の選択です。

変分分布を指定しない場合は、`CholeskyVariationalDistribution`を構成します。
誘導点数に対応する平均と共分散のパラメータを持ち、
単に観測の一部を固定して残す構造ではありません。
既定の`VariationalStrategy`は、その分布と誘導点、内側のGPを結び付けます。

複数出力では、出力別のbatchを持つ構成に
`IndependentMultitaskVariationalStrategy`を重ねます。
出力が複数あるだけで、多タスクGPのような学習された出力間共分散が追加されるわけではありません。
構造化出力や独自の変分戦略は、同じ経路として無条件に説明しません。

## 3. 変分戦略が予測を作る計算

[GPyTorchのvariational_strategy.py][g-variational]にある
`VariationalStrategy.forward()`は、誘導入力と予測入力を連結し、
内側のGPの`forward()`から事前分布を得ます。
そこから誘導点間、誘導点・予測点間、予測点間の共分散ブロックを分けます。

誘導点間の共分散にjitterを加えてCholesky因子を求め、
`_compute_predictive_updates()`で平均と共分散の補正を計算します。
既定戦略は白色化した変数の事前分布を標準正規として扱うため、
保持する変分分布を、そのまま未変換のu=f(Z)の分布と読み替えません。

白色化座標の平均を $m_v$、共分散を $S_v$、
誘導点間のCholesky因子を $L_Z$、補間を $A=L_Z^{-1}K_{ZX_*}$ と書けば、
通常の全共分散経路の補正は $A^\top m_v$ と $A^\top(S_v-I)A$ に対応します。
`solve()`で補間を求め、事前平均・事前共分散へ補正を加える箇所が、
Theoryの誘導変数を積分してq(f)を作る処理との接点です。

学習時に対角のみで足りる条件では、分散の対角に特化した計算も選びます。
したがって、全ての入力で同じ密な共分散補正を実体化するわけではありません。
jitterは数値安定化であり、観測ノイズを表すlikelihoodとは別です。

## 4. ELBOの各項を計算するメソッド

TheoryのELBOは、観測の期待対数尤度と誘導変数のKLに分かれます。

$$
\mathcal L_{\mathrm{ELBO}}=
\mathbb E_{q(f)}[\log p(y\mid f)]
-\mathrm{KL}(q(u)\Vert p(u)).
$$

`SingleTaskVariationalGP.make_mll()`は、保持したlikelihoodと内側のGPを使い、
`VariationalELBO(likelihood=self.likelihood, model=self.model, num_data=num_data)`を構成します。
この時点では、観測に対する目的値の評価や最適化は実行しません。

[GPyTorchのvariational_elbo.py][g-mlls]にある
`VariationalELBO._log_likelihood_term()`は、
`likelihood.expected_log_prob(target, variational_dist_f)`を求め、観測軸について足します。
これは期待対数尤度であり、予測密度の対数を取る処理と同一ではありません。
likelihoodを変えると、この項の計算も変わります。

`_approximate_mll.py`の`_ApproximateMarginalLogLikelihood.forward()`は、
その和を現在の観測数bで割り、KL項を`num_data / beta`で割ります。
ローカルのfactoryは`beta`を指定しないため既定値1となり、
全観測数Nを使ったKL/Nと、バッチ内の期待対数尤度の平均を組み合わせます。
Theory 9.20節の総和形式と比較するときは、この正規化を対応付けます。

さらに登録済みの対数事前分布と追加損失も集約します。
したがって、返り値を上の正規化前の式そのものとして数値比較しません。
KLは既定戦略の白色化座標で計算されますが、対応する可逆変換をそろえたKLは同じ意味を持ちます。

## 5. データ保持と学習の境界

`num_data`を省略すると、ローカルのfactoryは`raw_train_X`の行数を使います。
0以下は拒否します。コンストラクタに渡した入力が全データの一部なら、
この既定値だけで全観測数が推定されるわけではありません。
モデルを初期化する入力、現在ELBOへ渡すバッチ、正規化に使う全観測数は別です。

`train_Y`は任意であり、`raw_train_Y`が`None`になる場合があります。
上流のコンストラクタに観測分散を直接渡す入口はなく、`raw_train_Yvar`も`None`です。
raw bufferは来歴の保存であり、保存した応答が自動で全てのELBO評価へ渡る仕組みではありません。

このモデルは標準Exact GPの便宜的な`fit()`を継承しません。
学習では、内側のGPが作るq(f)と観測をELBOへ渡し、その目的に対するパラメータ更新を行います。
入力・出力変換を使う場合も、バッチとモデルが同じ座標・応答スケールを使う必要があります。

## 6. 予測入口・Mixed・分類の拡張

上流の`ApproximateGPyTorchModel.posterior()`は、入力を変換してから
内側のGPを評価し、必要ならlikelihoodを適用します。
得られた分布を`GPyTorchPosterior`などへ包み、出力を逆変換します。
学習時の期待対数尤度と、予測時にlikelihoodを適用する処理は異なる段階です。

`MixedSingleTaskVariationalGP`は、カテゴリ列を検査し、
共通mixedヘルパーで共分散を構成して親へ渡します。
nativeカテゴリ共分散の部品は[Mixed Guide](05_mixed_variables.md)を参照してください。
変分戦略が同じでも、全モデルが同じ入力表現を使うわけではありません。

[BinarySingleTaskGPClassifier][classifier]は、この変分モデルを拡張し、
`BernoulliLikelihood`を渡します。
ELBOの構成を共有していても、Gaussian回帰とは尤度項が異なります。
予測ラベルの確率と潜在関数の分布は[分類Guide](24_classification.md)で分けて説明します。
Fully Bayesian SAASのパラメータ積分も、このELBOの構成とは異なります。

## 7. 実装の担当範囲と参照先

誘導点配置、変分分布、変分戦略、予測計算はDelegatedです。
robotorchanはrawデータ保持とELBOの構成をExtended/Implementedとして接続します。
Exact MLL、変分ELBO、MCMCを、同じ名前の学習目的として扱わないことが要点です。

使用とminibatch学習は[Variational Notebook][notebook]、
wrapperとELBOの境界条件は[モデルテスト][model-test]を参照できます。
数学的な文献はTheoryの参考文献に集約しています。
上流コードはBoTorch 0.18.1・GPyTorch 1.15.2を対象とします。

[local]: ../../src/robotorchan/models/standard/variational.py
[b-models]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models
[g-variational]: https://github.com/cornellius-gp/gpytorch/tree/v1.15.2/gpytorch/variational
[g-mlls]: https://github.com/cornellius-gp/gpytorch/tree/v1.15.2/gpytorch/mlls
[classifier]: ../../src/robotorchan/models/classification/binary/standard/single_task.py
[notebook]: ../../examples/notebooks/06_variational_gp.ipynb
[model-test]: ../../tests/models/standard/test_variational_gp.py
