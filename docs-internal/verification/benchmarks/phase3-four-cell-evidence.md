# Phase 3: four-cell empirical evidence (2026-11-18)

This is a **record of measured evidence**, not a new benchmark implementation or a claim of general algorithm superiority. The protocol is in [empirical-evaluation-protocol.md](../../../docs/benchmarks/empirical-evaluation-protocol.md).

## Frozen runs and provenance

| Problem | q | GitHub Actions run | Original experiment SHA | Artifact |
|---|---:|---|---|---|
| Branin | 1 | [44093235539](https://github.com/tanakakao/robotorchan/actions/runs/44093235539) | `5fe440a8318a6c9a97e401180379fe16bc13bab3` | [recovered report, 44100622002](https://github.com/tanakakao/robotorchan/actions/runs/44100622002/artifacts/14334959201) |
| Branin | 3 | [44105379822](https://github.com/tanakakao/robotorchan/actions/runs/44105379822) | `8b1b8121b5fdd6264e50c902f1facafbf54d4fd9` | [14338740901](https://github.com/tanakakao/robotorchan/actions/runs/44105379822/artifacts/14338740901) |
| Hartmann6 | 1 | [44106195610](https://github.com/tanakakao/robotorchan/actions/runs/44106195610) | `8b1b8121b5fdd6264e50c902f1facafbf54d4fd9` | [14339551141](https://github.com/tanakakao/robotorchan/actions/runs/44106195610/artifacts/14339551141) |
| Hartmann6 | 3 | [44107676004](https://github.com/tanakakao/robotorchan/actions/runs/44107676004) | `8b1b8121b5fdd6264e50c902f1facafbf54d4fd9` | [14337814349](https://github.com/tanakakao/robotorchan/actions/runs/44107676004/artifacts/14337814349) |

The original Branin q=1 optimization run completed its trajectories but failed during report generation due to a `torch.quantile` dtype mismatch. The report was subsequently generated from **the preserved trajectories**, not by rerunning the optimizer, using the fixed reporting code. The recovered artifact contains both source and reporting provenance.

All four saved manifests report `runtime_validated`, CPU float64, 10 seeds (0–9), 20 initial observations and 50 additional evaluations. Random, Sobol, qEI and qNEI were evaluated. q=3 uses 17 batches, the final batch containing two points.

## Final simple regret

Lower is better. Means over the 10 seeds, computed from each saved `report.json` `final_metric_by_seed` array.

| Problem | q | Random | Sobol | qEI | qNEI |
|---|---:|---:|---:|---:|---:|
| Branin | 1 | 0.755294 | 0.949458 | 0.002399 | 0.002155 |
| Branin | 3 | 0.755294 | 0.949458 | 0.001094 | 0.001751 |
| Hartmann6 | 1 | 1.582809 | 1.390789 | 0.257661 | 0.163215 |
| Hartmann6 | 3 | 1.582809 | 1.390789 | 0.484626 | 0.294021 |

Within each problem/q cell, qEI and qNEI outperformed the random and Sobol baselines in the saved paired-bootstrap comparisons. This is limited evidence for the specified synthetic problems, implementation and evaluation budget.

## Exploratory q=1 vs q=3 paired-seed comparison

For each strategy and problem, define the per-seed difference as **regret(q=1) minus regret(q=3)**. Positive values favor q=3. Pair by the shared seed index; bootstrap the 10 differences with replacement for 10,000 replicates, using NumPy `default_rng(20261118)` independently for each comparison; percentile 2.5% and 97.5% intervals.

| Problem | Strategy | Mean difference | Exploratory 95% bootstrap interval |
|---|---|---:|---|
| Branin | qEI | +0.001305 | [-0.000996, +0.003777] |
| Branin | qNEI | +0.000404 | [-0.001287, +0.002298] |
| Hartmann6 | qEI | -0.226965 | [-0.548844, +0.023189] |
| Hartmann6 | qNEI | -0.130806 | [-0.290586, +0.055591] |

All four intervals include zero. **Do not claim q=1 or q=3 is statistically superior** from these results. The Branin q=1 cohort has a different experiment SHA from the other three cells; protocol fields match, but source-code equivalence has not been independently established. The q=1 and q=3 experiments also have different model update counts; a smaller recorded runtime for q=3 is not proof of better wall-clock efficiency under parallel execution.

The q=1 and q=3 runs were not explicitly designed as a cross-q paired statistical experiment. Matching seed indices permits an exploratory paired difference but does not by itself establish identical initialization across q or justify confirmatory inference. Before any stronger cross-q conclusion, validate identical initial observations across q from the saved trajectories and check source-code differences.

### Reproduce the exploratory intervals from downloaded artifacts

Download each linked artifact and extract to folders `branin-q1`, `branin-q3`, `hartmann6-q1`, `hartmann6-q3` respectively. Each folder must contain `report.json`. The following snippet uses only NumPy and the saved reports:

```python
import json
from pathlib import Path
import numpy as np

for problem in ("branin", "hartmann6"):
    reports = {q: json.loads((Path(f"{problem}-q{q}") / "report.json").read_text()) for q in (1, 3)}
    for strategy in ("qEI", "qNEI"):
        q1 = np.asarray(reports[1]["final_metric_by_seed"][strategy], dtype=float)
        q3 = np.asarray(reports[3]["final_metric_by_seed"][strategy], dtype=float)
        assert q1.shape == q3.shape == (10,)
        delta = q1 - q3
        rng = np.random.default_rng(20261118)
        sampled = delta[rng.integers(0, 10, size=(10_000, 10))].mean(axis=1)
        print(problem, strategy, delta.mean(), np.quantile(sampled, [0.025, 0.975]))
```

## Scope and next steps

- Four specified **noiseless single-objective** cells are measured and archived; they are not a comprehensive validation of all robotorchan optimizers, acquisitions, constraints, Mixed, or Active Learning.
- Retain artifact IDs and SHA provenance; GitHub Actions artifacts expire, so durable archival is a separate task.
- Verify cross-q initialization and experiment-code comparability before formalizing the q comparison.
- Avoid new benchmark machinery or extensive regression tests solely for this report.
