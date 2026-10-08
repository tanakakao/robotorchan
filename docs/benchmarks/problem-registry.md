# Benchmark Problem Registry (Phase 3)

The registry stores **factories**, not problem instances. This prevents shared
mutable tensors or evaluator state between independent benchmark runs.

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

A local `BenchmarkProblemRegistry()` is recommended for tests and independent
experiments. Duplicate names, unknown names, and invalid factory results fail
explicitly. Registry listings are sorted to ensure stable output.

Phase 3 deliberately does **not** register Branin, Hartmann, constrained,
multiobjective, or heterogeneous benchmark problems; those are introduced in
subsequent problem-family phases. No registry entry is created by importing
the module, and no global random seed is modified.
