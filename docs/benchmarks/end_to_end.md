# Benchmarking: end-to-end guide

This guide connects the benchmark problem registry, execution profiles, closed-loop
runner, truth-based metrics, multi-seed reporting, result storage, and checkpointing.

## Quick start

Run the self-contained example from the repository root:

```bash
python examples/benchmark_end_to_end.py
```

It creates `benchmark_outputs/branin_random_results.json`,
`benchmark_outputs/branin_random_report.json`, and
`benchmark_outputs/branin_random_report.csv`.

The example explicitly registers Branin in a local registry, obtains the
`smoke` execution profile, runs seeded random search, saves trajectories,
reloads them, computes truth-based simple regret, and exports the report.

## Experiment contracts

- `BenchmarkExperimentConfig.evaluation_budget` counts *new* evaluations
  only; the initial design is additional.
- `q` is the requested candidate batch size. The last batch may be smaller.
- A `BenchmarkProblemRegistry` must contain the named problem. Importing
  a problem module does not implicitly register it.
- `BenchmarkTrajectory.Y_observed` is what the strategy sees;
  `Y_truth` is for evaluation metrics. Do not leak truth to candidate selection.
- Feasibility constraints use `g(X) >= 0`. `simple_regret_curve`
  requires a known single-objective optimum; `hypervolume_curve`
  requires two objectives and a reference point.
- Report curves include the initial design. CSV `evaluation_index=0`
  is the first *initial* observation, not the first proposed candidate.

## Comparing strategies

Use the same problem, seeds, initial design settings, budget, q, dtype,
and device for every strategy. Compute the same metric on every seed.
Pass `{seed: curve}` mappings to `make_metric_report`, using
`interval_method="bootstrap"` for seed-resampled confidence intervals.
Use `compare_paired_curves` or `compare_paired_bootstrap` for
seed-aligned differences. The `smoke` profile contains only one seed:
use `standard` or `extended` for multi-seed uncertainty analysis.

## Storage and resumption

`save_benchmark_results` and `load_benchmark_results` persist complete
trajectories as versioned JSON. For interruption and continuation, use
`run_resumable_benchmark` and the separate checkpoint save/load APIs.
The resumable runner accepts **one seed** per invocation and saves the
run-local generator state. Exact continuation requires the strategy to
depend only on that generator and the stored trajectory; external optimizer
state, global RNG and stochastic observation noise are not captured.

## Reproducibility and limitations

Store the exact configuration and environment/package versions alongside
reports. Use identical initial designs and seed sets for paired comparisons.
Benchmark profiles are fixed resource presets, not quality guarantees.
CUDA support is problem- and strategy-dependent; requesting a CUDA profile
only checks CUDA availability. Do not compare wall-clock measurements across
different hardware as if they were algorithm-only differences.

## Related documentation

- [Final audit and release-readiness criteria](release_readiness.md)
- [Execution profiles](execution_profiles.md)
- [Result storage](storage.md)
- [Checkpointing](checkpoint.md)
- [Multi-seed reporting](reporting.md)
