# Phase 32: evidence-backed empirical visualization

Render charts from an existing complete Phase 31 artifact directory:

```python
from robotorchan.benchmarks.empirical_visualization import render_empirical_charts

render_empirical_charts("results/branin_q1_artifacts", "results/branin_q1_charts")
```

This requires optional `matplotlib` and creates `performance.svg`,
`timing.svg`, and `provenance.json`. The performance chart plots
mean truth-based regret or hypervolume across ten seeds from evaluation
budget 0 through 50. The timing chart plots mean elapsed wall-clock time.

The renderer refuses missing/incomplete source artifacts and existing
destinations. Cohorts remain separated by problem, q and noise condition.
The figures are descriptive and do not establish statistical superiority.
No expensive benchmark experiments are executed by this module.
