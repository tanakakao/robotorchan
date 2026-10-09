# Benchmark execution profiles

Phase 28 standardizes experiment sizes without changing the benchmark runner
or introducing implicit device selection.

| Profile | Seeds | Initial points | New evaluations per seed | q |
| --- | ---: | ---: | ---: | ---: |
| smoke | 1 | 4 | 4 | 1 |
| standard | 3 | 8 | 40 | 1 |
| extended | 5 | 16 | 100 | 4 |

All profiles default to CPU and float64. The evaluation budget excludes
initial design points. The final batch can be smaller than q.

```python
from robotorchan.benchmarks import get_execution_profile, run_benchmark

profile = get_execution_profile("smoke")
config = profile.make_config(problem="branin", strategy="random")
# Supply a registry containing the named problem and a candidate callback.
trajectories = run_benchmark(config, random_candidates, registry=registry)
```

Precision and device may be overridden explicitly:
`get_execution_profile("standard", dtype="float32", device="cuda")`.
CUDA availability is checked when requesting the profile. The profile
does not guarantee that a particular problem, strategy, or optional
dependency supports CUDA or float32; those are validated by execution.

The presets are resource budgets, not benchmark performance baselines.
For fair comparisons, use the same profile and problem registry across
strategies, and record the resulting configuration in result artifacts.
