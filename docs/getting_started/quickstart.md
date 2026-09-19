# Quickstart

このページでは robotorchan の共通モデル API を最小構成で確認します。

## Exact GP の最小例

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import SingleTaskGP

torch.set_default_dtype(torch.double)

train_X = torch.rand(20, 2)
train_Y = torch.sin(2 * torch.pi * train_X[:, :1]) + 0.2 * train_X[:, 1:2]

model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
fit_gpytorch_mll(model.make_mll())

test_X = torch.rand(5, 2)
posterior = model.posterior(test_X)

print(posterior.mean)
print(posterior.variance)
```

## 共通 API

多くの supervised wrapper は、BoTorch のモデル API を維持しながら次を提供します。

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.raw_data
model.raw_data_names
model.supports_mll
model.make_mll()
```

`raw_*` はコンストラクタに渡された training data のスナップショットです。
`condition_on_observations()` や `fantasize()` 後の現在状態ではありません。
現在の学習状態には BoTorch の `train_inputs` / `train_targets` を使用します。

## 学習方法はモデルによって異なる

`make_mll()` を利用できる Exact GP では通常
`fit_gpytorch_mll(model.make_mll())` を利用できます。一方で、Variational GP、
Fully Bayesian SAAS、Pairwise GP などは学習方法や入力表現が異なります。

特殊なモデルを利用するときは [モデルガイド](../models.md) と対応する
[Notebook](../../examples/README.md) を確認してください。

## 次に読むもの

問題設定からモデルを選ぶ場合は [Model selection](model_selection.md)、
数式や統計的仮定を確認する場合は [Theory](../theory/README.md) に進んでください。
