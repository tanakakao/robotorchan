# High-dimensional GP：SAASの事前分布と推論の実装

## この章で学ぶこと

[高次元GPのTheory](../theory/08_high_dimensional_gp.md)の8.9・8.11〜8.13・
8.19〜8.23節を、SAASの変数、推論処理、事後分布の表現へ対応付けます。
疎性を置く事前分布と、MCMCまたはMAPでパラメータを扱う計算方法を分けて読みます。

高次元の問題と観測数の多い問題は異なります。
SAASは元の入力座標の長さ尺度に疎性の仮定を置く方法であり、
PCAやAEのような新しい探索座標を生成する次元削減ではありません。
加法モデルも別の構造仮定であり、SAASと同義ではありません。

## 1. クラス構造と担当範囲

| モデル | ローカルの基底と上流への接続 |
| --- | --- |
| `SaasFullyBayesianSingleTaskGP` | データ保持・学習能力mixinとBoTorch SAASモデルを継承 |
| `SaasFullyBayesianMultiTaskGP` | 同じmixinと、タスク列を持つBoTorch SAASモデルを継承 |
| `AdditiveMapSaasSingleTaskGP` | `ExactGPModelMixin`とBoTorchの加法MAP-SAASを継承 |
| `EnsembleMapSaasSingleTaskGP` | `ExactGPModelMixin`とBoTorchのensemble MAP-SAASを継承 |

Fully Bayesianのラッパーは[fully_bayesian.py][local-fb]、
MAPのラッパーは[map_saas.py][local-map]にあります。
raw観測を保存する処理と学習能力の表示を追加しますが、
SAASの事前分布、NUTS、標準のMAPカーネルを独自に再実装していません。

Fully Bayesianの二クラスは`supports_mll=False`です。
その`make_mll()`は、[ModelTrainingMixin][base]の非対応処理へ進みます。
一方、MAPラッパーの`ExactGPModelMixin.make_mll()`は
`ExactMarginalLogLikelihood`を構成します。
両者の学習目的を同じ入口で扱う構成ではありません。

## 2. SAASの縮小が置かれる変数

BoTorchの[fully_bayesian.py][b-models]にある
`SaasPyroModel.sample_lengthscale()`は、逆二乗長さ尺度を階層的に生成します。
名前に`Pyro`が残っていますが、BoTorch 0.18.1のこの処理はNumPyroの標本生成を使います。
識別子の名称から、実際の推論基盤を判断しません。

`kernel_tausq`は全次元に共通する正の変数で、尺度0.1の半Cauchy分布から生成します。
`_kernel_inv_length_sq`は各次元の局所変数で、尺度1の半Cauchy分布から生成します。
前者をg、後者をr_jとすると、コードは $1/\ell_j^2=g r_j$ を作り、
その逆数の平方根を`lengthscale`へ写します。
Theory 8.9節の、逆長さ尺度と逆二乗長さ尺度を区別する必要がある箇所に対応します。

`postprocess_mcmc_samples()`も、この積から長さ尺度を再構成します。
中間変数を除き、JAX側の配列を訓練入力と同じdtype・deviceのTorch Tensorへ変換します。
事前分布からの計算と、GPyTorchのカーネルへパラメータを載せる処理の接点です。

## 3. NUTSからモデルへ標本を読み込む

[BoTorchのfit.py][b-fit]にある`fit_fully_bayesian_model_nuts()`は、
JAXとNumPyroの`NUTS`・`MCMC`を使います。
`model.pyro_model.sample`を確率モデルとしてNUTSを構成し、
warmupと標本生成を実行した後、後処理とthinningを行います。

その後に`model.load_mcmc_samples()`を呼び、評価モードへ移ります。
単一タスクの基底クラスでは、`pyro_model.load_mcmc_samples()`から
平均、共分散、likelihood、必要な入力変換を受け取ります。
構築時点から、学習済みのこれらの部品がそろっているわけではありません。
`_check_if_fitted()`は、共分散部品がまだない状態での予測を拒否します。

このように、確率モデルの定義、MCMC、後処理、Torchモデルへの読み込みは別の段階です。
raw観測の保持はrobotorchanの役割であり、NUTSの遷移や標本の生成は上流へのDelegatedです。
依存するJAX・NumPyroが必要な経路を、モデルのクラス名や通常のExact MLLだけで代替できません。

## 4. パラメータ標本ごとの条件付けと混合分布

単一タスクの基底クラスの`posterior()`は、入力にMCMC用の次元を追加して
上流の条件付けへ進み、`GaussianMixturePosterior`へ包みます。
各成分は、一つのハイパーパラメータ標本に条件付けたGaussian分布です。
Theory 8.13節のパラメータ不確実性の積分を、有限個の成分で表現します。

[posteriors/fully_bayesian.py][b-posteriors]の`mixture_mean`は成分平均を集約し、
`mixture_variance`は成分内分散の平均と、成分平均のばらつきを合わせます。
単に各成分の分散だけを平均しているわけではありません。
標本を表す次元を、実験候補数qや独立した出力数と混同しないことが重要です。

多タスクSAASでは、ロング形式のタスク列と潜在的なタスク表現を扱う
`MultitaskSaasPyroModel`などへ接続します。
その確率モデルと標本の読み込みもBoTorchへ委譲されます。
タスクの配置は[多タスクGuide](07_multitask_multioutput.md)、
SAASと他の高次元モデルの関係は[高度な高次元Guide](21_advanced_high_dimensional_models.md)へ進めます。

## 5. MAP・加法・ensembleの異なる構成

BoTorchの`map_saas.py`にある`add_saas_prior()`は、
カーネルへ長さ尺度の制約と事前分布を登録します。
`SaasPriorHelper`のclosureは、カーネルの現在値から
`tau / lengthscale**2`を取り出して事前分布の評価へ渡します。
固定tauを使う場合と、`raw_tau`をパラメータとして持つ場合も分かれます。
Fully Bayesian側の中間変数と、同じ名前の変数だと仮定して読み替えません。

`AdditiveMapSaasSingleTaskGP`は、
`get_additive_map_saas_covar_module()`が作る部品を受け取ります。
この関数はtauを生成し、SAAS priorを付けたMatérn 5/2をスケール部品で包み、
複数の成分を`AdditiveKernel`で加算します。
これは一つのGPの共分散を足す構造であり、複数のGP予測を混合する処理ではありません。

`EnsembleMapSaasSingleTaskGP`は、tauごとに訓練データをbatchとして複製し、
各成分のカーネルを構成します。`posterior()`は`GaussianMixturePosterior`を返しますが、
その成分はNUTSで得た事後標本と同じものではありません。
分布を包むクラスが同じでも、成分の生成方法は異なります。
この上流クラスは単一出力・二次元の訓練入力を要求するため、全てのbatch構造へ一般化しません。

MAPラッパーはExact MLLの構成を提供し、最適化は上流のfittingへ接続します。
標準Exact GPの`fit()`便宜機能を、全ての高次元モデルが継承するわけではありません。

## 6. カテゴリ・加法構造・探索との境界

Mixed SAAS系は、モデルが所有する`CategoricalOneHotInputTransform`を使います。
Fully Bayesian版は符号化後の入力を親へ渡し、予測入口でも同じ変換を適用します。
Mixed MAP版ではその変換を上流の`input_transform`へ渡します。
いずれもnative categorical kernelと同じ内部構成ではありません。
変換と列番号の詳細は[Mixed Variables Guide](05_mixed_variables.md)にまとめています。

Theory 8.27〜8.30節の加法構造、特に`OrthogonalAdditiveGP`の考え方も、
軸方向のSAAS priorとは異なります。
加法共分散、疎な長さ尺度、入力の次元削減を、一つの近似手法としてまとめません。

また、TuRBO・ALEBO・BAxUSなどの高次元探索は、候補を探す座標や領域の側の処理です。
モデルの仮定だけで候補最適化まで決まるわけではありません。
[高次元探索Guide](20_high_dimensional_search.md)と
[獲得関数最適化Guide](optimization/09_high_dimensional.md)を参照してください。

## 7. 実装の担当範囲と参照先

SAAS prior、MCMC、MAP用カーネル、事後分布の構成は上流へのDelegatedです。
robotorchanはデータ保持と学習能力の区別をExtendedとして加え、
Mixed版の符号化と入力の検査をImplementedとして接続します。
推論方式と、モデルが置く疎性・加法性の仮定を分けて読むことが要点です。

使用は[SAAS Notebook][notebook]と[MAP-SAAS・加法Notebook][map-notebook]、
wrapperの接続は[単一タスクSAASのテスト][fb-test]と[Mixed MAPのテスト][map-test]を参照できます。
文献の詳細はTheoryの参考文献へ集約しています。上流コードはBoTorch 0.18.1を対象とします。

[local-fb]: ../../src/robotorchan/models/high_dimensional/fully_bayesian.py
[local-map]: ../../src/robotorchan/models/high_dimensional/map_saas.py
[base]: ../../src/robotorchan/models/base.py
[b-models]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models
[b-fit]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/fit.py
[b-posteriors]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/posteriors
[notebook]: ../../examples/notebooks/08_saas_gp.ipynb
[map-notebook]: ../../examples/notebooks/09_map_saas_and_additive_gp.ipynb
[fb-test]: ../../tests/models/high_dimensional/test_fully_bayesian_single_task_gp.py
[map-test]: ../../tests/models/high_dimensional/test_mixed_map_saas.py
