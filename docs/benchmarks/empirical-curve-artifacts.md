# Phase 31: empirical curve and timing artifacts

Phase 31 adds a reproducible CSV export layer over the Phase 30
completeness-gated report. It does **not** manufacture benchmark results.

```python
from robotorchan.benchmarks.empirical_curves import export_empirical_curves

export_empirical_curves("results/branin_q1", "results/branin_q1_artifacts")
```

The output directory contains:

- `curves.csv`: truth-based simple regret or fixed-reference hypervolume
  after every evaluated point, including the 20-point initial design.
  `total_points` is the actual history length; `evaluations` counts
  additional evaluations (zero throughout the initial design).
- `timing.csv`: per-seed candidate generation, evaluation, initial
  evaluation and total elapsed seconds.
- `summary.json`: Phase 30's paired-seed summary and bootstrap intervals.

Only a fully validated 10-seed cohort is eligible for export. Separate
cohorts must be used for each problem, q and observation-noise condition.
The CSV files are plotting-ready but this phase does not produce plots,
execute expensive benchmark cohorts, or make performance claims.
