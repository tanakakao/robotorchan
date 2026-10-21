# Benchmark problem registry

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

The registry itself does not implicitly register built-in problem families
on import. Branin, Hartmann, constrained, multiobjective, and heterogeneous
problems are defined separately and can be registered explicitly for a run.
Importing the registry module does not change the global random seed.
See the [End-to-end benchmark guide](end_to_end.md) for explicit
registration and execution.
