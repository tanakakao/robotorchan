# Structured Output / Higher-Order GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../theory/11_structured_output.md)は数学的仮定・導出を担当する。本章では理論の構成要素がクラスと内部処理へ対応する経路を説明する。

応答がスカラーではなく画像・曲線・テンソルなどの構造を持つ場合、出力要素間の共分散を利用する。構造化出力のjoint posteriorと、
単一のスカラーobjectiveを扱う通常BOの獲得関数との間には出力変換・shape契約が必要となる。

## 実装の責務と公開契約

- **Extended:** `src/robotorchan/models/structured/higher_order.py`の`HigherOrderGP`はBoTorch
  `HigherOrderGP`を継承しraw入力・tensor出力の保持と`make_mll()`を加える。
- **Delegated:** 出力軸の潜在パラメータ、tensor covariance、Kronecker構造の線形代数とposteriorはBoTorch。
- **Extended:** `MixedHigherOrderGP`はdesign-inputの共分散因子のみnative mixed kernelへ置き換える。
  現在はカスタム`covar_modules`指定を拒否する。
- ほかに`src/robotorchan/models/structured/latent_kronecker.py`の別familyがあり、同一アルゴリズムとみなさない。

## Tensor出力における共分散因子と出力座標

入力Xを[n,d]、tensor出力Yを[n,r1,...,rk]とするHigher-Order GPでは、入力側のcovarianceと出力各軸のcovarianceを因子として扱う。
概念的な分離可能共分散は、

$$
\operatorname{Cov}(f(x,a_1,\ldots,a_k),f(x',a'_1,\ldots,a'_k))
=k_X(x,x')\prod_{j=1}^{k} k_j(a_j,a'_j)
$$

の形で表せる。これはfactorized priorの説明であり、全てのstructured-outputモデルが同じkernel因子やposterior構造を持つという主張ではない。

src/robotorchan/models/structured/higher_order.pyのHigherOrderGPはBoTorchHigherOrderGPを継承し、
covar_modules・num_latent_dims・learn_latent_pars・latent_initなどを親へ委譲する。
robotorchanはraw_train_X/raw_train_Yとraw_train_Yvar=Noneのsnapshot、ExactGPModelMixin.make_mll()を追加する。
構造出力のpredictive covariance、Kronecker処理およびposteriorサンプリングはBoTorch側の責務となる。

MixedHigherOrderGPはdesign入力だけをカテゴリ＋連続のmixed kernelへ変更する。
src/robotorchan/models/structured/higher_order.pyでcovar_modules[0]をmake_mixed_covar_moduleで置き換え、
出力軸側のfactorを保持する。custom covar_modulesを渡す場合はValueErrorで拒否し、cat_dimsは元の設計入力列の番号を取る。

**注意：** posteriorが多次元tensor出力を表す場合、tensorを単純にflattenしてscalar acquisitionへ渡しても意味論的な目的の定義にはならない。
最大・積分・特定画素などのfunctionalを別に定義する必要がある。

## HigherOrderGPの出力軸と学習

`HigherOrderGP`は`train_X: [n,d]`と`train_Y: [n,r1,...,rk]`を受け取る。例えば画像なら`[n,h,w]`であり、単なる出力ベクトルと違い、
各軸に共分散構造を置く。`num_latent_dims`、`learn_latent_pars`、`latent_init`は出力軸の潜在座標を制御する。
これは波長や時刻などの既知の物理座標をそのまま与えることとは異なる。

HOGPのstructured posteriorは出力shapeを保ち、構造を使ったsampling pathを持つ。
Matheron型のconditioningはこのsamplingを理解するための基礎となる。単に通常のExact GPと同じSciPy最適化器へ置き換える前に、
[構造化出力モデルガイド](../models/structured_output.md)の学習上の注意を確認する。

## LatentKroneckerGPの明示的な出力座標

`LatentKroneckerGP`は`X × T`の積空間をモデル化する。例えば`train_X: [n,d]`、`train_T: [m,1]`、
`train_Y: [n,m]`でスペクトルを表す。`T`は波長・時刻・位置などの座標であり、単なるカテゴリtask IDと同じではない。観測部分へのconditioning、反復線形ソルバ、
pathwise posterior samplingはBoTorchへ委譲する。

BoTorch 0.18.1のこのposteriorは、`mean`をXとTの積に沿う平坦な形で返す。batchなしの平均はXとTの積に沿う値を持ち、sampleではX軸とT軸に分かれる。
posterior sampleとmomentのshapeを同じと推測せず、batch次元も含めて確認する。

新しい`test_T`で予測できるが、補間と外挿の信頼性を区別する。`T`の尺度はlengthscaleに関係するため、物理単位と変換を明示する。欠損出力への対応は上流のmasking契約に従い、
欠損自体が応答に依存する場合の統計的な偏りまで自動的に解消するものではない。

入力と出力座標には別々に`covar_module_X` / `covar_module_T`、`mean_module_X` / `mean_module_T`を指定できる。
`MixedLatentKroneckerGP`はX側だけをmixed共分散にする。このモデルでも`cat_dims`は`train_X`の列であり、`train_T`の指定ではない。

## 出力表現を分けるクラス構造

| 観点 | HigherOrderGP | LatentKroneckerGP |
| --- | --- | --- |
| 主な応答 | 画像などのtensor | 時刻・波長などの座標に沿った応答 |
| 出力側の表現 | 各tensor軸の潜在座標 | 明示的な`T` |
| 新しい出力位置 | 学習したtensor軸の構造に依存 | `test_T`で指定 |
| 学習上の要点 | `_fast_solves`とPyTorch optimizer | `use_iterative_methods()` |
| 固有のraw API | `raw_train_X`, `raw_train_Y` | 左に加えて`raw_train_T` |
| Mixed版 | `MixedHigherOrderGP` | `MixedLatentKroneckerGP` |

いずれも`raw_train_Yvar`は`None`であり、raw値は変換・masking前の入力snapshotである。raw値と内部の変換済み・masking済みデータを区別する。

少数の独立出力なら`ModelListGP`、関連タスクなら`MultiTaskGP`や`KroneckerMultiTaskGP`も検討する。Kronecker構造の利用だけでは、
タスク共分散と物理出力座標のモデルを同一視できない。出力数が多くても軸構造がなければ、出力削減・潜在モデルが適することがある。Kronecker計算の効率は分離可能共分散という仮定と引き換えである。

## 構造化分布とscalar utilityの境界

代理モデルは応答全体の分布を返す。BO側では、特定位置、ピーク、積分値、目標スペクトルからの距離など、何を最大化するかを別に定義する。
非線形functionalはposterior sampleへ適用し、不確実性を伝播する。平均値へfunctionalを適用した結果が期待utilityと一致するとは限らない。
Objective / PosteriorTransformと獲得関数がtensorの軸を扱えるかも確認する。

[HOGPテスト](../../tests/models/structured/test_higher_order_gp.py)と
[LatentKroneckerテスト](../../tests/models/structured/test_latent_kronecker_gp.py)は、
境界条件はリンク先のテストを参照する。
[Structured Output Notebook](../../examples/notebooks/11_structured_output_gp.ipynb)に
学習・posterior比較・新しい`test_T`での予測例がある。

## 参考文献

1. Zhe, S., Xing, W., and Kirby, R. M. (2019). Scalable high-order Gaussian process regression.
   *Proceedings of AISTATS*.
2. BoTorch `HigherOrderGP` and `LatentKroneckerGP` documentation.
3. [Gaussian Process](../theory/02_gaussian_process.md)、[Kernel](../theory/03_kernel.md)、
   [Multi-task / Multi-output](../theory/07_multitask_multioutput.md)。

## 共分散因子からposteriorを組み立てる経路

[HOGP wrapper](../../src/robotorchan/models/structured/higher_order.py)は
`ExactGPModelMixin`とBoTorchの`HigherOrderGP`を継承します。上流の`_initialize_latents()`が出力軸の潜在座標を初期化し、
`forward()`が入力因子と出力各軸の因子をKronecker積として結合します。Theoryの分離可能共分散は、この因子ごとのkernel評価へ対応します。
robotorchanのMixed版が置き換えるのは先頭の入力因子だけです。

上流の`posterior()`と`make_posterior_variances()`は、この構造を保って条件付き分布・分散・sampling用の量を組み立てます。
wrapperはtensor出力の条件付けやMatheron型samplingを再実装しません。

[LatentKronecker wrapper](../../src/robotorchan/models/structured/latent_kronecker.py)も
`ExactGPModelMixin`を通じてMLLを加え、推論はBoTorchへ委譲します。上流の`forward()`はX側とT側の共分散を結合し、観測maskを適用します。
`_rsample_from_base_samples()`がpathwise conditioningを実行し、結果をX軸とT軸へ戻します。
平坦なmomentとstructured sampleの差はこの計算経路の違いに由来します。

上流ソースはBoTorch 0.18.1の
[HigherOrderGP](https://github.com/pytorch/botorch/blob/v0.18.1/botorch/models/higher_order_gp.py)と
[LatentKroneckerGP][lk]です。

[lk]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/models/latent_kronecker_gp.py
