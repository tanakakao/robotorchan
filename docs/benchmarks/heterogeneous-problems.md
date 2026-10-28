# 異種出力の benchmark 問題

`strength_conductivity_pass` は、競合する連続 Strength・Conductivity の最大化と、Pass=1・Fail=0 の二値実行可能性を組み合わせます。入力 2 変数は [0,1]、真値の実行可能性は `x1 - 0.25 - 0.5*x0 >= 0` です。

```python
from robotorchan.benchmarks import (
    strength_conductivity_pass,
    strength_conductivity_pass_labels,
)

problem = strength_conductivity_pass()
X = problem.bounds.mean(dim=0).unsqueeze(0)
regression_targets = problem.evaluate_truth(X)  # (q, 2)
binary_targets = strength_conductivity_pass_labels(X)  # (q, 1), values 0 or 1
ground_truth_margin = problem.evaluate_constraints(X)  # (q, 1)
```


## 学習 target と真値の分離

`BenchmarkProblem` は連続目的と符号付き margin を保持し、`g(X) >= 0` の出力制約契約に従います。二値 label は別の明示関数から取得し、margin を分類 target として使いません。label は決定的で、境界は Pass です。

問題と label の提供だけでは、分類器の予測確率、semantic adapter、制約付き qEHVI・qNEHVI、学習 pipeline との接続を認定しません。それらには専用の E2E テストが必要です。
