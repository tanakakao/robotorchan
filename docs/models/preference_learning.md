# Preference Learning: PairwiseGP Implementation

This guide contains executable API examples relocated from the theory chapter.
For mathematical assumptions, see [Preference Learning Theory](../theory/10_preference_learning.md).

## 10.20 robotorchan の `PairwiseGP`

基本構築は次の通りです。

```python
import torch
from robotorchan.models import PairwiseGP


datapoints = torch.tensor(
    [[0.1], [0.4], [0.8]],
    dtype=torch.double,
)
comparisons = torch.tensor(
    [[2, 1], [1, 0]],
    dtype=torch.long,
)

model = PairwiseGP(
    datapoints=datapoints,
    comparisons=comparisons,
)
```

---

## 10.21 constructor

robotorchan wrapper は現在の upstream contract に対応する主要引数を公開します。

```python
PairwiseGP(
    datapoints,
    comparisons,
    likelihood=None,
    covar_module=None,
    input_transform=None,
    jitter=1e-6,
    xtol=None,
    consolidate_rtol=0.0,
    consolidate_atol=1e-4,
    maxfev=None,
)
```

`datapoints` または `comparisons` が `None` の場合、upstream では prior-only model として構築できます。

---

## 10.22 `make_mll()`

robotorchan では

```python
mll = model.make_mll()
```

で

```python
PairwiseLaplaceMarginalLogLikelihood
```

を返します。

実体は概念的に

```python
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood

mll = PairwiseLaplaceMarginalLogLikelihood(
    model.likelihood,
    model,
)
```

です。

---

## 10.23 fitting

通常の robotorchan exact GP と同様、

```python
from botorch.fit import fit_gpytorch_mll

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

で hyperparameter fitting できます。

ただし objective は Gaussian regression の Exact MLL ではありません。
`PairwiseLaplaceMarginalLogLikelihood` は pairwise likelihood と Laplace approximation を使う
PairwiseGP 専用の学習 objective です。

---

## 10.70 robotorchan Notebook

実行例は

```text
examples/notebooks/07_pairwise_gp.ipynb
```

にあります。

Notebook では

1. latent utility を用意
2. comparison data を生成
3. PairwiseGP を構築
4. Laplace MLL を fitting
5. posterior utility を可視化
6. posterior-best point を確認

という流れを実行できます。

---

## 10.72 simple example

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import PairwiseGP


torch.set_default_dtype(torch.double)

x = torch.linspace(0.0, 1.0, 12).unsqueeze(-1)
true_utility = torch.sin(2 * torch.pi * x).squeeze(-1)

pairs = []
for i in range(11):
    j = i + 1
    if true_utility[i] >= true_utility[j]:
        pairs.append([i, j])
    else:
        pairs.append([j, i])

comparisons = torch.tensor(pairs, dtype=torch.long)

model = PairwiseGP(
    datapoints=x,
    comparisons=comparisons,
)

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

---

## 10.77 robotorchan で覚えておく API

```python
model = PairwiseGP(
    datapoints=datapoints,
    comparisons=comparisons,
)

model.raw_datapoints
model.raw_comparisons
model.supports_mll
model.make_mll()
model.posterior(test_X)
```

`make_mll()` は

```python
PairwiseLaplaceMarginalLogLikelihood
```

を返します。

---
