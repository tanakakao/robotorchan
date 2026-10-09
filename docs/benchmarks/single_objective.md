# Phase 6 — Single-objective benchmarks

The standard benchmark registry includes five deterministic minimization
problems:

| Problem | Dimension | Bounds | Known optimum |
| --- | ---: | --- | ---: |
| Ackley2 | 2 | [-5, 5] | 0 |
| Branin | 2 | [-5, 10] × [0, 15] | ≈0.397887 |
| Hartmann6 | 6 | [0, 1] | ≈-3.322368 |
| Rosenbrock2 | 2 | [-2, 2] | 0 |
| Sphere3 | 3 | [-5, 5] | 0 |

All problems expose native minimization values. The benchmark problem's
`to_maximization` method converts directions for downstream acquisition and
metrics code. Analytic minima and known-optimum regret are tested independently
of optimizer performance.

Use `register_standard_problems(registry)` to register the suite.
Run each method with the same `BenchmarkExperimentConfig` seeds and initial
design for paired comparisons. This phase expands problem coverage; statistical
performance claims require the later fair-comparison and multi-seed phases.
