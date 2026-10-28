# Genetic Algorithm backend

実数値の Genetic Algorithm（GA）による獲得関数最適化器です。追加依存は不要で、集団生成・選択・交叉・変異・バッチ評価に PyTorch Tensor を使います。

個体は joint 候補 `q x d` 全体です。内部では `q*d` に平坦化し、獲得関数には `[population, q, d]` を渡します。各候補を独立に最適化しません。

エリート保存、トーナメント選択、算術交叉、変数範囲でスケールした Gaussian 変異を使い、変異後は box に収めます。専用のローカル `torch.Generator` を使い、グローバル乱数系列を変更しません。

集団は bounds の device に保持され、GPU 上の進化演算と獲得関数のバッチ評価に対応します。

連続座標が対象です。整数・カテゴリには別の Mixed GA を使います。候補制約では実行可能な個体を優先し、実行不可能な個体間は違反量で順位付けします。[候補制約](candidate-constraints.md)も参照してください。

```python
from robotorchan.optim.backends import optimize_acqf_ga

candidates, value = optimize_acqf_ga(
    acq_function=acq,
    bounds=bounds,
    q=3,
    population_size=128,
    generations=100,
    seed=123,
)
```
