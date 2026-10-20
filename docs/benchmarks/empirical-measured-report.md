# Phase 30: measured empirical comparison report

Phase 30 is the reporting gate, not a claim that the complete expensive
benchmark matrix has been executed.

## Source and validation

Use `summarize_empirical_directory(directory)` to read the existing
`manifest.json` and 40 persisted seed trajectories (four strategies times
ten seeds). Only `runtime_validated` cohorts with matching frozen protocol,
paired initial design/truth, exact evaluation budget, complete timing and
correct output dimensions can produce a report. Missing files and failed
runs block the comparison. `save_empirical_report` refuses overwrite.

## Quantities

- Branin/Hartmann6: final truth-based simple regret (lower is better).
- Branin-Currin: final truth-based hypervolume against the problem's
  predeclared reference point (higher is better).
- Candidate and evaluation elapsed time per seed, including initial
  evaluation time.
- Seed-paired mean improvements relative to Random and Sobol, with
  deterministic 10,000-resample percentile bootstrap 95% intervals.
- Noisy and noiseless cohorts must remain separate. Do not pool q=1 and
  q=3, different problems, or different commits into one comparison.

## Interpretation and outstanding evidence

A confidence interval is descriptive, not a corrected multiple-testing
significance claim. The Phase 30 JSON summary does not itself include performance curves,
CSV exports, or rendered plots. These are available as **separate**, later
artifact stages: [Phase 31 curve and timing CSV exports](empirical-curve-artifacts.md)
and [Phase 32 SVG visualization](empirical-visualization.md).
Neither stage performs expensive cohort execution or supplies missing
measured results. Multiple-comparison correction is not established by
these artifact stages. The cohort runners must be executed to generate
actual evidence before reporting numerical performance or superiority. Constrained GP methods
remain blocked by Phase 29's classification acquisition eligibility gate.

To produce an artifact from a completed cohort:

```python
from robotorchan.benchmarks.empirical_report import save_empirical_report

save_empirical_report("results/branin_q1", "results/branin_q1_report.json")
```
