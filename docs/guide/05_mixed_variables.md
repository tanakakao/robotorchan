# Mixed Variables：カテゴリ表現と共分散の実装

## この章で学ぶこと

[Mixed VariablesのTheory](../theory/05_mixed_variables.md)の5.2〜5.6節を、
カテゴリ比較、混合共分散、one-hot変換、候補探索の処理へ対応付けます。
同じ混合変数でも、モデルごとにカテゴリを扱う実装経路は異なります。
本章では、BoTorchの標準Mixedモデルと、robotorchanの構造化モデル用の共通部品を分けます。

## 1. 標準MixedSingleTaskGPの継承構造

[models/standard/single_task.py][single-task]の`MixedSingleTaskGP`は、
`FittableExactGPModelMixin`とBoTorchの`MixedSingleTaskGP`を継承します。
コンストラクタはrawな訓練データをコピーし、カテゴリ列などを親へ渡した後に
共通のデータ保持処理へ接続します。
`make_mll()`と`fit()`の構成は[GP Guide](02_gaussian_process.md)と共通です。

カテゴリ共分散の構築とGP推論は、親の[gp_regression_mixed.py][b-models]へ委譲します。
このラッパーは、独自の`make_mixed_covar_module()`で親の部品を上書きしません。
また、ラッパーの`cat_dims`保存値と、親が内部で正規化して使う列番号は区別します。
入力検査が全てrobotorchanの共通ヘルパーを通るわけではありません。

## 2. カテゴリ比較が共分散になるまで

BoTorchの[CategoricalKernel][b-kernels]は、GPyTorchの`Kernel`を継承します。
`forward()`は二つの入力のカテゴリ値を`!=`で比較し、不一致を表すTensorを作ります。
それを列ごとの長さ尺度で割り、通常の経路ではカテゴリ列について平均し、
符号を反転した指数関数へ渡します。
これはTheory 5.2節の、カテゴリ同一性に基づく共分散を計算する箇所です。

カテゴリ番号の差や大小を距離として使う処理ではありません。
そのため、同一性を保つ一対一の再符号化では、同じパラメータの共分散が保たれます。
これはカテゴリ表現の性質であり、番号を変えたモデルが常に同じ学習経路を通るという主張ではありません。

`Kernel.__call__()`の`active_dims`が、比較対象の入力列を選びます。
カテゴリ列へ通常の連続入力用の正規化を無条件に適用すると、
モデルと候補の間で同一性の比較がずれる場合があります。
BoTorchのMixedモデルが受け取る入力変換も、カテゴリ列を変更しないことが前提です。

## 3. 連続項・カテゴリ項・交互作用項

BoTorchの`MixedSingleTaskGP.__init__()`は、カテゴリ列とそれ以外の列を分けます。
連続部がある場合、連続とカテゴリの加法項、および両者の積の項を組み立てます。
加法の組と積の組をそれぞれ`ScaleKernel`で包み、加法側のカテゴリ部品にもスケールを置きます。
連続部がなければ、scaled categorical kernelだけを作ります。
この部品構成が、Theory 5.3節の和と相互作用に対応します。

一方、[models/base.py][base]の`make_mixed_covar_module()`は、
他のmixedモデル向けの共通構築関数です。
連続項・scaledカテゴリ項・積の項を`AdditiveKernel`で組み合わせますが、
BoTorch標準Mixedモデルと同じスケール部品の配置ではありません。
対応式と部品ごとの違いは[Kernel Guide](03_kernel.md)にまとめています。

共通ヘルパーは、`normalize_feature_dims()`でraw入力列を正規化してから
構造列を除外し、残りの連続列とカテゴリ列を子カーネルの`active_dims`に割り当てます。
負の列番号は正規化し、その後に重複、範囲外、構造列との重なりを拒否します。
カテゴリ共分散を構成することと、入力列の意味を検査することは別の処理です。

## 4. 構造列を通常のカテゴリ列と分ける

| モデル系 | 主な実装とカテゴリの扱い |
| --- | --- |
| `MixedSingleTaskGP` | BoTorchがnative mixed covarianceを構成 |
| `MixedMultiTaskGP` | 共通mixedヘルパーで設計共分散を構成し、タスク列を除外 |
| `MixedSingleTaskMultiFidelityGP` | 同ヘルパーから忠実度列を除外し、上流の忠実度モデルへ接続 |
| `MixedSaasFullyBayesianSingleTaskGP` | モデルがone-hot変換を所有し、符号化後にSAASモデルへ接続 |

[multitask.py][multitask]の`MixedMultiTaskGP`は、`MultiTaskGP`を拡張します。
`task_feature`を通常の設計カテゴリに混ぜず、タスク共分散はBoTorch側に任せます。
共通ヘルパーの子カーネルはraw列番号を使うため、最上位の`active_dims`でも
元の列配置を維持し、親が非タスク列だけへ再番号付けすることを避けます。

[multi_fidelity.py][fidelity]の`MixedSingleTaskMultiFidelityGP`は、
`SingleTaskMultiFidelityGP`を拡張し、忠実度列を設計共分散から除外します。
`linear_truncated=True`では忠実度カーネルが全入力の共分散を所有するため、
この組合せを拒否します。カテゴリの列番号を追加するだけで、
どの忠実度カーネルとも合成できる設計ではありません。

## 5. one-hot変換を所有するモデル

[fully_bayesian.py][fully-bayesian]の`MixedSaasFullyBayesianSingleTaskGP`は、
`SaasFullyBayesianSingleTaskGP`を拡張し、
`CategoricalOneHotInputTransform`を使って訓練入力を符号化します。
この経路は、カテゴリ比較のnative kernelを組み立てる経路とは異なります。

`CategoricalOneHotInputTransform.__init__()`は、訓練データで観測したカテゴリ値を
列ごとに収集・整列し、bufferへ保存します。
`transform()`はその値との等値比較からone-hot列を作り、
カテゴリ以外の列と順に連結します。未知のカテゴリは、全て零の符号へ黙って写さず拒否します。

raw列がone-hotの複数列へ変わるため、符号化後の入力次元と列番号も変わります。
`encoded_scalar_index()`は、カテゴリ以外のraw列に対応する符号化後の位置を求めます。
カテゴリ列は一つのスカラー位置にならないので、この操作では拒否します。

モデルの`posterior()`はraw入力を同じ変換へ通してから親へ渡します。
外部で別のカテゴリ入力変換を同時に渡すことは拒否し、
訓練と予測の符号化をモデル自身が一致させます。
入力warpingを使う場合も、カテゴリ列を対象から除外して列番号を変換します。

## 6. 予測モデルと候補探索の境界

整数で保存された値が、数量か名義カテゴリかによってカーネルの意味は変わります。
標準Mixedモデルは、`cat_dims`以外の列を連続側の部品へ渡します。
数量を表す整数を含めたからといって、予測モデル自身が候補を整数へ丸めるわけではありません。

候補探索側では、[MixedVariableSpace][variable-space]などが
入力列の型と許容値を保持します。
[BoTorch mixed backend][mixed-backend]では、離散割当を固定する設定を
上流の`optimize_acqf_mixed()`へ渡します。
この探索の処理と、カテゴリ同一性を使う予測カーネルの計算は異なる責務です。
学習・予測・候補生成で同じraw列とカテゴリ値を対応付ける必要があります。

q点を同時に選ぶ場合やone-shotの補助変数を含む場合は、割当の単位も変わります。
詳しくは[混合変数最適化Guide](optimization/04_mixed_discrete.md)と
[バッチ・先読みGuide](optimization/08_batch_one_shot.md)を参照してください。

## 7. 実装の担当範囲と参照先

標準Mixedモデルの共分散と推論はDelegated、
rawデータ保持と学習入口はExtendedです。
構造化モデル向けの部品構成、入力列の検査、one-hot変換はImplementedです。
モデル名だけから、全てが同じカーネルや符号化を使うとは判断できません。

使用例は[Mixed GP Notebook][notebook]、
標準モデルの接続は[上流との対応テスト][model-test]、
カテゴリの同一性と共分散の関係は[再符号化の参照テスト][reference-test]を参照できます。
上流の参照コードはBoTorch 0.18.1を対象とします。

[single-task]: ../../src/robotorchan/models/standard/single_task.py
[base]: ../../src/robotorchan/models/base.py
[multitask]: ../../src/robotorchan/models/standard/multitask.py
[fidelity]: ../../src/robotorchan/models/standard/multi_fidelity.py
[fully-bayesian]: ../../src/robotorchan/models/high_dimensional/fully_bayesian.py
[b-models]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models
[b-kernels]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models/kernels
[variable-space]: ../../src/robotorchan/optim/domains/variable_space.py
[mixed-backend]: ../../src/robotorchan/optim/backends/botorch.py
[notebook]: ../../examples/notebooks/02_mixed_single_task_gp.ipynb
[model-test]: ../../tests/models/standard/test_mixed_single_task_gp.py
[reference-test]: ../../tests/models/standard/test_phase7_mixed_categorical_reference.py
