# Kernel：共分散関数とrobotorchanの設定

## この章で解決する問題

GPにおいて、入力点どうしの関係を共分散として表現し、予測分布の形や滑らかさの仮定を制御する。[Kernelの理論](../../../theory/03_kernel.md)では半正定値性、RBF、Matérn、stationarity、ARDなどの数学を説明する。この章は実装・API上の接続に重点を置く。

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

カテゴリ変数のラベルを連続座標上の数値距離とみなすのは不適切な場合がある。[Mixed Variablesの理論](../../../theory/05_mixed_variables.md)と対応章（後続Phaseで作成）でnative categoricalとone-hot等を個別に区別する。Nonstationaryや構造化kernelもこの章の標準RBF例で検証した扱いにはしない。

PSDと行列対称性は数学的必要条件だが、kernelのクラスが存在することだけでは独立数値検証の完了を示さない。既存[verification Mapping](../verification.md)のproperty別証拠と結び付ける。関連：[GP](02_gaussian_process.md)、[モデルガイド](../../../models/standard.md)、[標準GP Notebook](../../../../examples/notebooks/01_single_task_gp.ipynb)。
