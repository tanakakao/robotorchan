# NSGA-II とベイズ最適化

通常の BoTorch 多目的獲得関数の最適化では、NSGA-II を scalar `optimize_acqf` の backend として使いません。qEHVI、qNEHVI、qLogEHVI、qLogNEHVI は多目的改善をスカラー獲得値へ集約済みであり、BoTorch・DE・CMA-ES・GA・Hybrid で最適化できます。そのスカラーに NSGA-II を適用しても多目的の意味は加わりません。

NSGA-II は数値最適化の対象自体がベクトル値の場合に使います。複数の posterior 統計量、明示的な獲得基準ベクトル、後段の選択まで Pareto 集合を保持する BO と進化計算の組合せが例です。専用 API を提供します。

```python
from robotorchan.optim.backends import optimize_vector_nsga2

pareto_X, pareto_values = optimize_vector_nsga2(
    objective=vector_objective,
    bounds=bounds,
    population_size=128,
    generations=100,
    seed=123,
)
```


目的関数は `[population, d]` を受け取り `[population, m]` を返します。すべての目的を最大化し、最終集団の非支配 front だけを返します。

PyTorch によるバッチ評価、非支配ソート、crowding distance、二者トーナメント、交叉、変異、エリート保存の環境選択を使います。

現在は候補制約なしの連続変数が対象です。scalar `optimize_acqf` とは接続しません。制約付き多目的進化最適化と NSGA-III は未対応であり、BoTorch のスカラー獲得関数契約と混同しません。
