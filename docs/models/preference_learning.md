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


## Additional API examples and implementation notes

## 10.4 Pairwise comparison data

BoTorch / robotorchan の `PairwiseGP` では、入力候補を

```python
datapoints
```

比較結果を

```python
comparisons
```

として与えます。

例えば

```python
datapoints = torch.tensor(
    [
        [0.1],
        [0.4],
        [0.8],
    ],
    dtype=torch.double,
)

comparisons = torch.tensor(
    [
        [2, 1],
        [1, 0],
    ],
    dtype=torch.long,
)
```

なら、

```text
2番目の候補 > 1番目の候補
1番目の候補 > 0番目の候補
```

という意味です。

各行は

```text
[winner, loser]
```

です。

---

## 10.18 duplicate datapoints の consolidation

Preference dataset では同じ候補が複数回現れることがあります。

BoTorch の `PairwiseGP` は numerical stability のため、近接・重複 datapoints を内部で consolidation できます。

wrapper の引数には

```python
consolidate_rtol
consolidate_atol
```

があります。

robotorchan はこの upstream behavior をそのまま利用します。

---

## 10.19 raw data と consolidated data

robotorchan では constructor に渡された値を

```python
model.raw_datapoints
model.raw_comparisons
```

として保持します。

これは **caller input の provenance** です。

一方 BoTorch model 内部では duplicate consolidation 後の

```python
model.datapoints
model.comparisons
```

が使われる場合があります。

したがって両者は必ずしも同じ shape になるとは限りません。

---

## 10.25 posterior

学習後は通常の BoTorch model と同様に

```python
posterior = model.posterior(test_X)
```

で latent utility posterior を得られます。

ここで posterior が表しているのは

```text
観測スコア y
```

ではなく

```text
潜在 utility f(x)
```

です。

---

## 10.26 posterior mean の解釈

```python
mean = posterior.mean
```

が高い候補ほど、モデルは「より好まれやすい」と推定しています。

ただし absolute scale より ranking が重要です。

```text
posterior mean
x1: -0.8
x2:  0.2
x3:  1.1
```

なら

```text
x3 > x2 > x1
```

という ranking interpretation が自然です。

---

## 10.37 EUBO

BoTorch には preference optimization 向け acquisition function として

```python
AnalyticExpectedUtilityOfBestOption
```

があります。

EUBOは **Expected Utility of the Best Option** の略です。

候補集合を提示したとき、その集合から得られるbest-option utilityの期待値を評価します。
ただし「PairwiseGPを使えば自動的にEUBOになる」わけではなく、surrogateとは別に
preference acquisitionを構成します。

---

## 10.40 preference BO loop

概念的には

```python
model = PairwiseGP(datapoints, comparisons)
fit_gpytorch_mll(model.make_mll())

# preference acquisition を構築
# acquisition を最大化して next_X を選ぶ
# next_X 内の pairwise comparison を取得
# datapoints / comparisons に追加
# refit
```

となります。

---

## 10.43 raw data API

robotorchan では

```python
model.raw_datapoints
model.raw_comparisons
```

を提供します。

通常の supervised wrappers の

```python
raw_train_X
raw_train_Y
```

とは命名を分けています。

Preference data の semantics を明確にするためです。

---

## 10.44 input normalization

PairwiseGP でも kernel を使うため、連続入力なら unit cube への normalization が有効です。

例えば

```python
from botorch.models.transforms.input import Normalize

model = PairwiseGP(
    datapoints=datapoints,
    comparisons=comparisons,
    input_transform=Normalize(d=datapoints.shape[-1]),
)
```

とできます。

---

## 10.59 numerical stability

constructor には

```python
jitter = 1e-6
```

があります。

これは Cholesky 等の numerical stability のために使われます。

また

```python
xtol
maxfev
```

は latent utility MAP を求める数値解法の収束制御に関係します。

通常は default から始め、収束問題が出た場合に調整します。

---

## 10.71 raw-data provenance

Notebook / wrapper では

```python
model.raw_datapoints
model.raw_comparisons
```

を確認できます。

これは model fitting 後の consolidated state ではなく constructor input の snapshot です。

再現性やデバッグではこの区別が重要です。

---

## 10.73 posterior ranking

```python
test_X = torch.linspace(0.0, 1.0, 200).unsqueeze(-1)
posterior = model.posterior(test_X)
mean = posterior.mean.squeeze(-1)

best_idx = mean.argmax()
best_x = test_X[best_idx]
```

これにより比較データしか与えていなくても、latent utility posterior に基づいて好ましい領域を推定できます。

---
