# Comparative Benchmark — Phase 24 Final Audit

## Scope and evidence

This audit describes the benchmark implementation present on `main` at the
start of Phase 24. The existence of an API or a passing smoke test is not
evidence of statistically significant optimizer superiority.

| Area | Evidence | Status |
| --- | --- | --- |
| Experiment contract | `benchmarks/config.py`, `runner.py` | Implemented |
| Seeded initial design | `runner.sobol_initial_design` | Implemented |
| Random baseline | `runner.random_candidates` | Implemented |
| Single-objective comparison | `comparative_single_objective.py` | Implemented |
| Constrained comparison | `comparative_constrained.py` | Implemented |
| Paired statistics | `comparative_statistics.py` | Implemented |
| Runtime / cost | `comparative_efficiency.py` | Implemented |
| Sensitivity sweeps | `comparative_sensitivity.py` | Implemented |
| BoTorch candidate adapter parity | `comparative_botorch_parity.py` | Limited scope |
| Visualization data | `comparative_visualization.py` | Data only |
| Markdown reports | `comparative_reporting.py` | Summary only |
| Reproducibility | `comparative_reproducibility.py` | CPU random smoke |

## Fairness and interpretation

- Use identical initial observations, seeds, evaluation budgets, and batch
  sizes when comparing algorithms. Matching the seed integer alone does not
  guarantee that different methods consume identical random streams.
- Do not compare simple regret with hypervolume or feasible regret. Report
  the metric direction and checkpoint semantics.
- A between-seed standard error is not a confidence interval or a
  significance test. Paired comparisons must retain seed alignment.
- Before any feasible point exists, feasible regret may be `+inf`; its
  uncertainty band is undefined. Do not silently convert it to zero.
- Candidate-generation wall time is hardware-dependent and must not be
  conflated with oracle evaluation cost.

## Important limitations

1. Phase 20 parity checks exercise the benchmark adapter against its
   shared native BoTorch candidate-generation path. They are **not** an
   independent optimizer-quality comparison.
2. Phase 21 emits renderer-independent plot specifications, not rendered
   Matplotlib or Plotly figures.
3. Phase 22 emits Markdown summary text; it does not yet package full
   experimental artifacts, environment metadata, or figures.
4. Phase 23 checks deterministic CPU Random trajectories and JSON encoding.
   It does not establish reproducibility for stochastic GP training,
   CUDA kernels, or all acquisition optimizers.
5. CI smoke tests do not replace repeated end-to-end experiments across
   problem families and statistically powered seed sets.

## Release criteria

- All CI jobs for this audit PR must pass.
- No unresolved correctness review comments may remain.
- Existing comparison, persistence, and reproducibility tests must pass.
- Publish actual benchmark datasets and quantitative conclusions only after
  running a documented comparison matrix with sufficient seeds and
  hardware/environment provenance.

**Decision:** Infrastructure-complete for the scoped comparative benchmark
cycle, conditional on CI and review. Not a claim of production performance
superiority or comprehensive optimizer parity.
