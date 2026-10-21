# Benchmark environment provenance and reproducibility

Use the existing benchmark runner and save an environment manifest beside every
result artifact. The manifest captures the repository SHA when Git is available,
Python, package versions, OS, CPU information, thread settings, device, precision,
and experiment configuration. A missing SHA or package version is recorded as
`null`; it is not evidence of a pinned environment.

```python
from pathlib import Path

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.empirical_environment import save_empirical_environment

config = BenchmarkExperimentConfig(
    problem="branin",
    strategy="random",
    seeds=(0, 1),
    initial_points=10,
    evaluation_budget=10,
    q=1,
    dtype="float64",
    device="cpu",
)
save_empirical_environment(Path("benchmark_outputs") / "environment.json", config)
```

The existing `verify_random_baseline_reproducibility` checks repeated CPU
Random trajectories and JSON roundtrips. Run its regression tests and the
manifest tests before treating this baseline as verified:

```bash
pytest -q tests/benchmarks/test_empirical_phase2_reproducibility.py tests/benchmarks/test_empirical_environment.py
ruff check src/robotorchan/benchmarks/empirical_environment.py tests/benchmarks/test_empirical_environment.py
ruff format --check src/robotorchan/benchmarks/empirical_environment.py tests/benchmarks/test_empirical_environment.py
```

## Limits

- The manifest is captured on the machine executing the benchmark, not on
  the developer's machine merely preparing the PR.
- Numerical reproducibility excludes wall-clock timing.
- GP fitting, acquisition MC, optimization, observation noise, and CUDA
  determinism require separate validation.
- For noisy experiments, an explicit observation-noise policy is available,
  including `SeededObservationNoise(policy="paired")`. Record the policy
  and seed mapping; paired policy does not by itself guarantee pointwise
  common random numbers at different adaptive candidate locations.
  See the [Comparative fairness protocol](comparative-fairness-protocol.md)
  before certifying noisy paired comparisons.
- This utility does not record peak memory, CPU model reliably on all systems,
  or the full resolved dependency lockfile. Preserve those externally where
  needed.
- Never treat a successful CI smoke test as a multi-seed performance result.
