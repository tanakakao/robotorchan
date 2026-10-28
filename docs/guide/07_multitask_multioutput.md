# MultiTask / MultiOutput GP：相関構造と出力の所有関係

## この章で学ぶこと

[多タスク・多出力のTheory](../theory/07_multitask_multioutput.md)の
7.2〜7.6・7.10節を、タスク共分散、観測の配置、出力の合成へ対応付けます。
出力が複数あるというだけでは、タスク間の依存を学習する構造は決まりません。
本章では、相関を持つモデル、独立GPの合成、異種モデルのcontainerを分けて読みます。

## 1. モデル構造を決める観測の配置

| モデル | 訓練状態と内部構造 |
| --- | --- |
| `MultiTaskGP` | 入力の一列がタスクを表すロング形式。各行に一つの応答を持つ |
| `KroneckerMultiTaskGP` | 同じ入力で全タスクを観測する完全ブロック。応答はn×m |
| `ModelListGP` | 子GPごとに訓練データを持ち、子の独立性に基づいて合成 |
| `HeterogeneousModel` | 子ごとの予測の意味を保持し、共通posteriorを作らない |

[models/standard/multitask.py][local]の前二者は、
`ExactGPModelMixin`とそれぞれのBoTorchモデルを継承します。
robotorchanはrawデータ保持とMLL構成を追加し、タスク共分散や予測は上流へ委譲します。
`KroneckerMultiTaskGP`は、ロング形式モデルの別名ではありません。

## 2. ロング形式で共分散を組み立てる

ICMの共分散は、設計入力とタスクの部品を次のように対応付けられます。

$$
\operatorname{Cov}(f_t(x),f_{t'}(x'))=k_X(x,x')B_{tt'}.
$$

BoTorchの[multitask.py][b-models]にある`MultiTaskGP.__init__()`は、
設計入力の`data_covar_module`とタスクの`PositiveIndexKernel`を作り、積で接続します。
この積は、各観測行のタスクを選んだカーネル値の積です。
全入力と全タスクの直積を先に作る処理ではありません。
Theoryの一般的なBの議論と、実装で選ばれた具体的なタスクカーネルの構成を区別します。

`forward()`は、`_split_inputs()`でタスク列と前後の設計列を分けます。
タスク値を内部添字へ写し、その添字を戻した入力で共分散を評価します。
設計側の`active_dims`はタスク列を除外し、タスク側はタスク列だけを使います。
既定平均も`MultitaskMean`としてタスクに応じて評価します。
既知の観測分散がなければ、既定の`HadamardGaussianLikelihood`がタスク列を用いてノイズを扱い、
分散が与えられた場合は`FixedNoiseGaussianLikelihood`へ渡します。
観測のタスク相関と、各観測に加わるノイズは別の部品です。

予測入口は、BoTorchの`gpytorch.py`にある`MultiTaskGPyTorchModel.posterior()`です。
入力にタスク列があれば、その行で指定したタスクを予測します。
タスク列がなければ、対象タスクごとに入力を展開して条件付けし、
複数出力の分布へ戻します。タスク値の写像と、この出力軸への再配置は別の処理です。

## 3. Kronecker構造を使う経路

`KroneckerMultiTaskGP`は、GPyTorchの`MultitaskKernel`を使います。
[GPyTorchのmultitask_kernel.py][g-kernels]では、
設計側の共分散とタスク側の共分散を`KroneckerProductLinearOperator`で組み合わせます。
これが、完全ブロックでの設計行列とタスク行列のKronecker積に対応する箇所です。

BoTorchの`train_full_covar`は訓練入力でこの構造を作り、
`predictive_mean_cache`はlikelihoodのノイズ共分散を加えて、中心化した観測に対し`solve()`します。
ここでは観測の行列をベクトル化するため、入力とタスクの配置順も計算の一部です。
同じn×m個の値を保持していても、配列順を変えれば同じ計算にはなりません。

likelihoodには`MultitaskGaussianLikelihood`を使う構成があり、
タスク共分散と観測ノイズの構造は別です。
このラッパーのコンストラクタは観測ごとの`train_Yvar`を受け取らず、
`raw_train_Yvar`にも`None`を保存します。ロング形式の既知観測分散の入口と同一ではありません。
完全ブロックという仮定を外す場合の数学的な限界は、Theory 7.10節を参照してください。

## 4. Mixedモデルが追加する設計共分散

`MixedMultiTaskGP`は`MultiTaskGP`、
`MixedKroneckerMultiTaskGP`は`KroneckerMultiTaskGP`を拡張します。
共通の`make_mixed_covar_module()`を使って、設計側にカテゴリと連続の共分散を組み込みます。
この部品の構成式は[Kernel Guide](03_kernel.md)に集約しています。

ロング形式のMixed版では、タスク列を通常の設計カテゴリから除外します。
子カーネルがraw列番号を使うため、最上位の`active_dims`でも列の対応を保ち、
親が非タスク列へ独自に付け替えることを避けます。
カテゴリの符号化や許容値を扱う処理と、タスク相関を扱う部品を分ける設計です。

## 5. 独立した子GPをModelListGPでまとめる

[models/standard/model_list.py][model-list]の`ModelListGP`は、
`ModelTrainingMixin`とBoTorchの`ModelListGP`を継承します。
タスク共分散を学習するクラスではなく、独立な子GPをまとめるクラスです。
子モデルごとに異なる訓練入力を持てることが、完全ブロックとは異なります。

`raw_train_Xs`・`raw_train_Ys`・`raw_train_Yvars`は、子モデルの保存値を順に集めます。
container全体で一つの訓練X・Yを作りません。
`make_mll()`は`SumMarginalLogLikelihood(self.likelihood, self)`を構成し、
子のMLLをまとめる上流の学習目的へ接続します。
共通の予測入口があっても、子の間に学習された相関が追加されるわけではありません。

## 6. 異種の予測をHeterogeneousModelで保持する

[models/heterogeneous.py][heterogeneous]の`HeterogeneousModel`は、
`torch.nn.Module`を継承し、子を`ModuleList`へ登録します。
このためdevice変更や学習・評価モードは子へ伝わりますが、
container自身は回帰と分類を混ぜた共通の`posterior()`を定義しません。

コンストラクタは子の出力数を集め、累積した`_output_offsets`を作ります。
`output_owner()`は全体の出力番号がどの区間に入るかを調べ、
その子の添字と内部の出力番号を返します。
この写像が、[Acquisition Guide](04_acquisition_function.md)で説明する目的・制約の接続に使われます。

`entry_posterior()`は指定した子の分布をそのまま返し、
`entry_make_mll()`・`entry_fit()`は子の学習処理へ接続します。
分類では`entry_latent_posterior()`と`entry_predict_proba()`を分け、
分類mixinを持つ子であることを検査します。
潜在関数の分布と、ラベルに対する確率を同じ応答としてまとめないためです。

これはrobotorchanがImplementedとして持つ、出力の所有関係と意味を保つ仕組みです。
共有されたGaussian事後分布を新たに近似するアルゴリズムではありません。

## 7. 実装の担当範囲と参照先

タスク共分散、Kronecker演算、GPの条件付け、独立GP合成の主要処理はDelegatedです。
rawデータとMLL構成はExtended、Mixedの部品構成と異種出力の管理はImplementedです。
予測の出力数だけでモデルを分類せず、観測配置、共分散、出力所有関係を順に読みます。

使用例は[多タスクNotebook][notebook]と[ModelList Notebook][list-notebook]、
出力管理の境界条件は[HeterogeneousModelのテスト][container-test]、
共分散の配列順は[Kronecker参照テスト][reference-test]を参照できます。
上流の参照コードはBoTorch 0.18.1・GPyTorch 1.15.2です。

[local]: ../../src/robotorchan/models/standard/multitask.py
[model-list]: ../../src/robotorchan/models/standard/model_list.py
[heterogeneous]: ../../src/robotorchan/models/heterogeneous.py
[b-models]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models
[g-kernels]: https://github.com/cornellius-gp/gpytorch/tree/v1.15.2/gpytorch/kernels
[notebook]: ../../examples/notebooks/04_multitask_gp.ipynb
[list-notebook]: ../../examples/notebooks/05_model_list_gp.ipynb
[container-test]: ../../tests/models/contracts/test_heterogeneous_model.py
[reference-test]: ../../tests/models/standard/test_phase7_kronecker_covariance_reference.py
