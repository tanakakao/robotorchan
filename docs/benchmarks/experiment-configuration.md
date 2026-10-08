# Benchmark Experiment Configuration (Phase 4)

`BenchmarkExperimentConfig` is a frozen, JSON-compatible record describing
the experimental protocol independently of any GP, acquisition function,
optimizer, or result persistence implementation.

```python
from robotorchan.benchmarks import BenchmarkExperimentConfig

config = BenchmarkExperimentConfig(
    problem="quadratic",
    strategy="sobol",
    seeds=(0, 1, 2, 3, 4),
    initial_points=12,
    evaluation_budget=60,
    q=3,
    dtype="float64",
    device="cpu",
)
record = config.to_dict()
restored = BenchmarkExperimentConfig.from_dict(record)
assert restored == config
```

- `problem` is resolved against a registry with `resolve_problem(registry)`.
- `strategy` is an identifier only; dispatch is reserved for the runner.
- `seeds` are unique nonnegative integers, with ordering retained.
- `initial_points` are evaluated before the optimization budget.
- `evaluation_budget` counts **additional evaluations**, not BO iterations.
- `q` is the requested batch size; a final partial batch is permitted.
- `dtype` is float32 or float64; `device` is cpu or cuda.
- `to_dict` and `from_dict` support JSON roundtrips, rejecting unknown fields.

A configuration alone does not modify the global RNG, allocate CUDA tensors,
train models, or run a benchmark. The Phase 5 runner will be responsible for
deterministic seed handling, paired initial designs, and device/dtype transfer.
