# 問題 registry

可変 Tensor や評価器状態を実験間で共有しないよう、instance ではなく factory を保存します。

```python
import torch

from robotorchan.benchmarks import BenchmarkProblem, get_problem, list_problems, register_problem


def make_quadratic() -> BenchmarkProblem:
    return BenchmarkProblem(
        name="quadratic",
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        objective=lambda X: -(X - 0.5).square(),
        directions=("maximize",),
        variable_types=("continuous",),
        optimal_value=torch.tensor([0.0], dtype=torch.double),
    )


register_problem("quadratic", make_quadratic)
problem = get_problem("quadratic")
assert "quadratic" in list_problems()
```


テスト・独立実験にはローカルな `BenchmarkProblemRegistry()` を推奨します。重複名、未知名、不正な factory 結果は明示的に失敗し、一覧はソートして安定化します。

import 時に標準問題を暗黙登録せず、global seed も変更しません。Branin・Hartmann・制約・多目的・異種問題は実行ごとに明示的に登録します。[全体手順](end_to_end.md)を参照してください。
