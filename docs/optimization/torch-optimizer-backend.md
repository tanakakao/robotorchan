# PyTorch backend

BoTorch の獲得関数 API を維持し、restart 初期化と候補選択に `botorch.optim.optimize_acqf`、局所最適化に `botorch.generation.gen.gen_candidates_torch` を使います。

`adam`、`adamw`、`sgd` または `torch.optim.Optimizer` の subclass を指定できます。既定は Adam です。BoTorch が closure と停止ループを管理するため、固有の意味とテストが必要な LBFGS は対応対象としません。

連続 box、q-batch、fixed features、device・dtype の保持に対応します。獲得関数と Tensor が GPU 上にあれば GPU で実行できます。候補制約は未対応として拒否し、暗黙の罰則で近似しません。制約付き探索には[対応する backend](candidate-constraints.md)を使ってください。

## 使用例

```python
from robotorchan.optim.backends import optimize_acqf_torch

candidates, value = optimize_acqf_torch(
    acq_function=acq,
    bounds=bounds,
    q=3,
    num_restarts=10,
    raw_samples=512,
    optimizer="adam",
    options={
        "optimizer_options": {"lr": 0.025},
        "stopping_criterion_options": {"maxiter": 200},
    },
)
```
