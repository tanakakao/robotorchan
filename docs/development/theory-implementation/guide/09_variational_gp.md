# Variational GP：inducing pointsとELBOの実装

## 理論と利用場面

[Variational GP理論](../../../theory/09_variational_gp.md)は、観測数`n`が増えるとExact GPの密な行列計算が重くなる問題を扱う。`d`（入力次元）が大きい問題とは区別する。誘導点`Z`（通常`m<n`）とvariational distributionを用いてposteriorを近似し、ELBOを学習目的にする。これはExact GPの周辺尤度そのものを直接最大化する方法ではない。

## 実装と推論契約

`src/robotorchan/models/standard/variational.py`の`SingleTaskVariationalGP`はBoTorch `SingleTaskVariationalGP`を継承する。

- **Delegated:** inducing-point allocation、variational strategy、posterior、input/outcome transformsはBoTorchに委譲する。
- **Implemented（統合）:** robotorchanはraw dataを保持し、`make_mll(num_data)`がGPyTorchの`VariationalELBO`を構築する。
- **注意:** `make_mll()`の戻り値はExact GPの`ExactMarginalLogLikelihood`ではない。`num_data`はELBOで表現する**全観測数**で、入力がminibatchのときはそのミニバッチ行数と同一とは限らない。
- **Mixed variant:** `MixedSingleTaskVariationalGP`はnative categorical covarianceを利用する。カテゴリ変数があるからといって全variationalモデルの内部表現が同じとは限らない。

## 使用例：ELBOとinducing points

```python
import torch
from robotorchan.models import SingleTaskVariationalGP

X = torch.linspace(0, 1, 30, dtype=torch.double).unsqueeze(-1)
Y = torch.sin(6 * X)
model = SingleTaskVariationalGP(
    train_X=X,
    train_Y=Y,
    inducing_points=8,
)
elbo = model.make_mll(num_data=X.shape[-2])
print(type(elbo).__name__)
```

この例はモデルとELBOの構築を示すもので、パラメータ最適化を実行していない。学習時はVariational strategy・likelihood・ELBOのoptimizerステップに対応した処理が必要。モデルの`raw_train_Y`は任意入力により`None`になり得るため、必ず教師データが保持されるものとして扱わない。

## 何を検証するか

inducing pointの数、ELBOの符号とスケール、minibatchと`num_data`、posteriorの予測形状・数値、gradientの伝播を個別に検証する。尤度非Gaussianやmulti-outputへの拡張では同一計算路を仮定しない。shapeの一致やCI成功だけでは、近似posteriorの正確性は証明できない。

参考：[モデルガイド](../../../models/standard.md)、[Variational Notebook](../../../../examples/notebooks/06_variational_gp.ipynb)、[旧Mapping](../models.md)、[Verification](../verification.md)。関連：[Exact GP](02_gaussian_process.md)、[高次元GP](08_high_dimensional_gp.md)。
