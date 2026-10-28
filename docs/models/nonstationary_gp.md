# 非定常 Gaussian Process

入力によって潜在関数の滑らかさが変わる Exact GP です。Gibbs 共分散を使い、観測ノイズが入力依存になる heteroskedastic モデルとは区別します。

## 共分散と局所 lengthscale

`GibbsKernel` は正のアフィン lengthscale 場を使います。

```text
ell(x) = softplus(a + B x) + lengthscale_floor
```


切片・傾きのパラメータを GP と同時学習し、入力空間で滑らかさを変化させます。`lengthscale_floor` は正です。

## モデル

| モデル | 入力構造 | 局所 lengthscale の診断 |
| --- | --- | --- |
| `NonstationarySingleTaskGP` | 連続・単一タスク | `local_lengthscale(X)` |
| `MixedNonstationarySingleTaskGP` | Mixed・単一タスク、`cat_dims` | 加法・相互作用の連続 branch の組 |
| `NonstationaryMultiTaskGP` | long-format task 列 | task ID を除くデータ特徴の値 |
| `MixedNonstationaryMultiTaskGP` | long-format task と Mixed データ | 連続 branch の組 |
| `NonstationaryKroneckerMultiTaskGP` | block-design 多出力 | データ特徴の `local_lengthscale(X)` |

Mixed は連続データ列に Gibbs kernel を適用し、カテゴリ共分散と合成します。連続データ列が 1 個以上必要で、`task_feature` と `cat_dims` は重複できません。

## 単一タスクの例

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import NonstationarySingleTaskGP

train_X = torch.rand(24, 2, dtype=torch.double)
train_Y = torch.sin(6 * train_X[:, :1])

model = NonstationarySingleTaskGP(
    train_X,
    train_Y,
    lengthscale_floor=1e-3,
)
fit_gpytorch_mll(model.make_mll())

test_X = torch.rand(5, 2, dtype=torch.double)
posterior = model.posterior(test_X)
local_scales = model.local_lengthscale(test_X)
```


通常の `make_mll()` 学習、BoTorch posterior、元訓練データの保持に対応します。lengthscale 診断は学習したカーネル構造で、観測ノイズや因果的な regime 境界の直接推定ではありません。

## モデル化の境界

- Heteroskedastic GP：入力依存の観測分散。
- 非定常 GP：入力依存の潜在共分散・滑らかさ。
- 不確実入力 GP：訓練座標の不確実性。
- Risk-aware acquisition：意思決定時の目的集約。

この API が扱うのはアフィン lengthscale 場です。neural・GP 駆動の場や change-point kernel の対応を意味しません。特に疎なデータでは定常基準法と比較してください。[Robust / Noise](robust_noise.md)と[モデル選択](../getting_started/model_selection.md)も参照してください。
