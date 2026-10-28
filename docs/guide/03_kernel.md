# Kernel：共分散部品とモデル内部の構成

## この章で学ぶこと

[KernelのTheory](../theory/03_kernel.md)の共分散行列、RBF、Matérn、ARD、
和・積の構成を、GPyTorchのカーネル部品へ対応付けます。
標準GPに渡された部品がどこで評価され、混合変数用の部品がどう組み立てられるかを追います。
条件付き分布の計算そのものは[GP Guide](02_gaussian_process.md)を参照してください。

## 1. カーネルの定義と評価の境界

[SingleTaskGP][single-task]は、`covar_module`をBoTorchの親クラスへ渡します。
親の`forward()`は平均部品と共分散部品を評価し、潜在関数の事前分布を作ります。
訓練点間、訓練・予測点間、予測点間で同じ部品を評価することが、
Theory 3.2節の共分散ブロックを構成する処理に対応します。

GPyTorchの[Kernel][g-kernels]は、カーネル計算の基底クラスです。
`Kernel.__call__()`は`active_dims`で入力列を選び、
`diag`などの条件に応じて具体的な`forward()`または遅延評価の表現へ進みます。
`forward()`の中の距離計算を読む前に、どの入力列が渡るかを確認する必要があります。

通常の行列評価では、設定により`LazyEvaluatedKernelTensor`を返せます。
これは、呼び出した時点で常に全共分散行列を密に作る設計ではないことを意味します。
対角のみの評価と、全ての点間共分散の評価も別です。

## 2. Theoryの式とGPyTorchの部品

| Theory上の対象 | 主な実装 |
| --- | --- |
| RBFの距離と指数関数（3.5節） | `rbf_kernel.py`の`RBFKernel.forward()` |
| Matérnの距離・多項式・指数関数（3.6節） | `matern_kernel.py`の`MaternKernel.forward()` |
| 次元ごとの長さ尺度（3.9節） | `Kernel`の`raw_lengthscale`・`lengthscale` |
| 共分散の振幅（3.8節） | `scale_kernel.py`の`ScaleKernel.forward()` |
| カーネルの和・積（3.12・3.13節） | `kernel.py`の`AdditiveKernel`・`ProductKernel` |
| カテゴリ一致による共分散（3.16節） | BoTorchの`CategoricalKernel.forward()` |

これらのファイルは[GPyTorchのkernels][g-kernels]と
[BoTorchのcategorical.py][b-categorical]にあります。
数値計算は上流が担当し、robotorchanの標準モデルはこれらを受け取る拡張です。

### RBF・Matérn・ARD

RBFの通常のTensor演算経路では、入力を`lengthscale`で割り、
`covar_dist(..., square_dist=True)`で距離を求めて指数関数へ写します。
Matérnでは距離に対する指数因子と、`nu`に応じた因子を掛けます。
GPyTorchの`MaternKernel`が扱う`nu`は0.5・1.5・2.5です。
勾配の要否などに応じて専用の自動微分経路も選ぶため、全条件で同じ関数本体だけを通るとは限りません。

`ard_num_dims`は、選んだ入力列に対応する長さ尺度の数を決めます。
`lengthscale`は、登録した制約を通じて`raw_lengthscale`から得る値です。
正値化の変換を使う構成と、変換を無効にして下限を最適化側へ渡す構成があるため、
全てが同じパラメータ変換を使うとは仮定しません。

### 振幅と和・積

`ScaleKernel`は基底カーネルの値へ`outputscale`を掛けます。
`AdditiveKernel`は子の値を加算し、`ProductKernel`は要素ごとに掛けます。
後者は共分散行列同士の通常の行列積ではありません。
これが、Theoryでのカーネルの和・積を実装する箇所です。

標準`SingleTaskGP`の既定部品は、BoTorch 0.18.1では
`get_covar_module_with_dim_scaled_prior()`が作るARD RBFです。
この部品に独立した`ScaleKernel`が必ず付くわけではありません。
部品の生成は[BoTorchのgpytorch_modules.py][b-utils]にあり、
既定の構成と、明示的に渡したカーネルの構成を分けて読みます。

## 3. robotorchanの混合カーネル構築

[models/base.py][base]の`make_mixed_covar_module()`は、
他の構造化mixedモデルで使う共通の構築関数です。
標準の連続GP全体の既定カーネルを変更する処理ではありません。

連続部を $x_c$、カテゴリ部を $x_{\mathrm{cat}}$ とすると、構成は次の式に対応します。

$$
k_{\mathrm{mixed}}(x,x')=k_c(x_c,x_c')
+s_k k_{\mathrm{cat}}(x_{\mathrm{cat}},x'_{\mathrm{cat}})
+k'_c(x_c,x'_c)k'_{\mathrm{cat}}(x_{\mathrm{cat}},x'_{\mathrm{cat}}).
$$

これは一般の混合カーネルの再導出ではなく、実装された部品の対応式です。
$s_k$は加法側のカテゴリ部品を包む`ScaleKernel`の倍率です。
和の最上位を`AdditiveKernel`、交互作用を`ProductKernel`で構成します。
交互作用側のカテゴリ部品は、このヘルパーでは`ScaleKernel`で包みません。

二回の`continuous_kernel()`は、それぞれ既定部品または
`cont_kernel_factory`を呼びます。既定では別々のカーネルが生成されます。
式の $k_c$ と $k'_c$ を同じパラメータを共有する部品とみなしません。
独自factoryが部品を再利用する場合は、そのfactoryの設計も読む必要があります。
連続次元がない場合は、積の項を作らずscaled categorical kernelだけを返します。

構築前には`normalize_feature_dims()`が負の列番号を正規化し、
重複・範囲外・構造列との重なりを拒否します。
`continuous_feature_dims()`は、カテゴリ列と除外された構造列の補集合を取ります。
この処理が、カーネルに使う入力列とモデル全体の入力列を対応付けています。

## 4. 標準Mixedモデルとの違い

[MixedSingleTaskGP][single-task]は、この共通ヘルパーを呼びません。
BoTorchの[gp_regression_mixed.py][b-mixed]がmixed covarianceを組み立てます。
BoTorch側にも連続項・カテゴリ項・交互作用項がありますが、
加法の組全体と積の組全体にもそれぞれ`ScaleKernel`を置きます。
共通ヘルパーと部品木やパラメータ共有が同じであるとはいえません。

この違いを保ったまま、各モデルがどの構築経路を使うかを読む必要があります。
カテゴリの比較とone-hot変換の詳細は[Mixed Variables Guide](05_mixed_variables.md)へ集約しています。

## 5. 変換・構造化共分散との関係

入力の`Normalize`は、カーネルへ渡す座標と長さ尺度の意味を変えます。
出力の`Standardize`は、カーネルだけでなく観測とノイズの応答スケールも変えます。
これらはGPの入出力変換部品であり、カーネルの種類そのものではありません。

カーネルをモデルの構造へ組み込む方法も複数あります。
入力とタスクの共分散を組み合わせる`MultiTaskGP`、
Kronecker構造を使う`KroneckerMultiTaskGP`は[多タスクGuide](07_multitask_multioutput.md)、
`HierarchicalConditionalKernelGP`・`HierarchicalConditionalKernelMultiTaskGP`や
contextの構造は[階層・文脈Guide](12_hierarchical_contextual_gp.md)を参照してください。
これらを標準RBF部品だけの説明で置き換えることはできません。

## 6. 実装の担当範囲と参照先

RBF・Matérn・和・積・スケール・カテゴリ比較の数値計算はDelegatedです。
robotorchanの標準モデルは部品を親へ渡すExtendedの層であり、
共通mixedヘルパーは部品の配置と入力列の検査をImplementedとして提供します。

行列の対称性・半正定値性という理論的要件と、
パラメータの制約や入力列の検査が担う役割も分けて考えます。
カーネルが妥当でも、仮定した類似性がデータに適するかは別の問題です。

使用の流れは[標準GP Notebook][notebook]と[Mixed GP Notebook][mixed-notebook]、
カテゴリ共分散の性質は[カテゴリ再符号化のテスト][category-test]へ進めます。
上流の参照コードはBoTorch 0.18.1・GPyTorch 1.15.2を対象とします。

[single-task]: ../../src/robotorchan/models/standard/single_task.py
[base]: ../../src/robotorchan/models/base.py
[g-kernels]: https://github.com/cornellius-gp/gpytorch/tree/v1.15.2/gpytorch/kernels
[b-categorical]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models/kernels
[b-utils]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models/utils
[b-mixed]: https://github.com/pytorch/botorch/tree/v0.18.1/botorch/models
[notebook]: ../../examples/notebooks/01_single_task_gp.ipynb
[mixed-notebook]: ../../examples/notebooks/02_mixed_single_task_gp.ipynb
[category-test]: ../../tests/models/standard/test_phase7_mixed_categorical_reference.py
