# Kernel：共分散関数とrobotorchanの設定

## この章で解決する問題

GPにおいて、入力点どうしの関係を共分散として表現し、予測分布の形や滑らかさの仮定を制御する。[Kernelの理論](../theory/03_kernel.md)では半正定値性、RBF、Matérn、stationarity、ARDなどの数学を説明する。この章は実装・API上の接続に重点を置く。

## Kernelからposteriorまで

Kernel `k(x,x')` は訓練点間の `K_XX`、訓練・予測点間の `K_*X`、予測点間の `K_**` を作る。観測ノイズと組み合わせた共分散を条件付けすることでposteriorの平均・分散が定まる。kernelが入力の距離をどう解釈するかはBOの探索結果にも影響する。

共分散行列は対称半正定値である必要がある。正のlengthscaleやカーネルのparameter制約は通常GPyTorch側のmoduleで管理される。RBFの滑らかさの仮定とMatérnの仮定は交換可能ではない。

## 実装と所有権

- **Delegated:** 標準GPで用いるcovariance kernelの計算はBoTorch/GPyTorchが所有する。
- **Extended:** `src/robotorchan/models/standard/single_task.py` の `SingleTaskGP` は `covar_module` 引数をBoTorch基底クラスへ渡す。独自の標準RBFやMatérn式を再実装しているわけではない。
- **Implemented（専用ヘルパー）:** `src/robotorchan/models/base.py` の `make_mixed_covar_module` は、BoTorchの`CategoricalKernel`とGPyTorchの`AdditiveKernel`, `ProductKernel`, `ScaleKernel`を構成する。これは標準連続入力 `SingleTaskGP` のデフォルトkernelと同一視しない。

標準モデルのデフォルトkernelはBoTorchバージョンと初期化条件に依存する。ソース未確認の固定kernelを断言せず、実際の `model.covar_module` を検査する。

## 最小利用例：明示的なカーネル指定

```python
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import SingleTaskGP

X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
Y = torch.cos(2 * torch.pi * X)
kernel = ScaleKernel(RBFKernel(ard_num_dims=1))
model = SingleTaskGP(train_X=X, train_Y=Y, covar_module=kernel)
fit_gpytorch_mll(model.make_mll())
print(type(model.covar_module).__name__)
print(model.posterior(X[:2]).mean)
```

この例では意図的にRBFを選んでいる。デフォルトがRBFであるという意味ではない。学習データとモデルが同じdtype/deviceで動くことを確認する。多次元ARDの`ard_num_dims`は入力次元に合わせる。

## 適用限界と検証

カテゴリ変数のラベルを連続座標上の数値距離とみなすのは不適切な場合がある。[Mixed Variablesの理論](../theory/05_mixed_variables.md)と対応章（後続Phaseで作成）でnative categoricalとone-hot等を個別に区別する。Nonstationaryや構造化kernelもこの章の標準RBF例で検証した扱いにはしない。

PSDと行列対称性は数学的必要条件だが、kernelのクラスが存在することだけでは独立数値検証の完了を示さない。既存[verification Mapping](../../.github/project-records/theory-guide/mapping/verification.md)のproperty別証拠と結び付ける。関連：[GP](02_gaussian_process.md)、[モデルガイド](../models/standard.md)、[標準GP Notebook](../../examples/notebooks/01_single_task_gp.ipynb)。

## Mixed kernelの構成式と実装上の意味

src/robotorchan/models/base.pyのmake_mixed_covar_moduleは、通常の連続GPのデフォルトkernelを置き換える一般規則ではなく、連続設計変数とカテゴリ設計変数のための専用構築関数である。連続次元が存在する場合の構成は概念的に

$$
k_{\mathrm{mixed}}(x,x')=k_c(x_c,x_c')
+s_k k_{\mathrm{cat}}(x_{\mathrm{cat}},x'_{\mathrm{cat}})
+k'_c(x_c,x'_c)k'_{\mathrm{cat}}(x_{\mathrm{cat}},x'_{\mathrm{cat}}).
$$

ここで加法項のcategorical kernelはScaleKernelで包まれ、交互作用項のcategorical kernelはunscaledである。2つのcontinuous_kernel()呼出しは別moduleを作るため、同一のパラメータ共有を前提にしてはいけない。連続次元がない場合はscaled CategoricalKernelのみを返す。GPyTorch AdditiveKernel/ProductKernel、BoTorch CategoricalKernelを組み合わせるのはrobotorchan、各kernelの数値計算は上流への委譲である。

cat_dimsにはraw入力列の番号を使い、normalize_feature_dimsが負インデックスを正規化し、重複・範囲外・structural excluded_dimsとの重なりを拒否する。task/fidelity/context/hierarchyの構造列を通常のカテゴリ設計変数として無断使用しない。連続部にはget_covar_module_with_dim_scaled_priorまたは指定したcont_kernel_factoryを利用する。

このnative categorical kernel経路は、同じbase.pyに別途存在するCategoricalOneHotInputTransformとは異なる。MixedSingleTaskGPはBoTorch MixedSingleTaskGPの実装・cat_dims契約を継承するので、すべてのmixedモデルがone-hotを行うと記述してはならない。

**検証境界：** ソースのkernel構成とカテゴリ次元の検査を確認した。独立したPSD/パラメータ推定/予測精度の全面検証は今回行っていない。

## 理論章から分離した実装上の条件

#### 3.10 入力の正規化：実装上の補足

BoTorch では input transform として `Normalize` が利用できます。

#### 3.11 出力の標準化：実装上の補足

BoTorch では outcome transform として `Standardize` が利用できます。

#### 3.16 Categorical kernel：実装上の補足

robotorchan では `MixedSingleTaskGP` がこの用途に対応します。

#### 3.17 Conditional / Hierarchical kernel：実装上の補足

robotorchan では

## 数値 API と利用上の補足

ライブラリ固有のモデル選択・APIについては
[モデルドキュメント](../models/README.md) を参照してください。

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

したがって、モデル名が同じ `SingleTaskGP` でも kernel の選択・hyperparameter によって予測の性質は大きく変わります。

`MultiTaskGP` や `KroneckerMultiTaskGP` の理論を理解する際の基礎になります。

これは `KroneckerMultiTaskGP` の重要な考え方です。

- `HierarchicalConditionalKernelGP`
- `HierarchicalConditionalKernelMultiTaskGP`

```text
SingleTaskGP
  └─ 入力間 covariance

MixedSingleTaskGP
  └─ 連続 + category covariance

MultiTaskGP
  └─ 入力 covariance × task covariance

KroneckerMultiTaskGP
  └─ K_X ⊗ K_T

HierarchicalConditionalKernelGP
  └─ 条件付き探索空間を考慮した covariance

Contextual GP
  └─ context 間構造を含む covariance
```
