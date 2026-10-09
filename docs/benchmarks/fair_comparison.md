# Fair comparison protocol

Phase 23 introduces `FairComparisonProtocol(first, second)` for
pairwise strategy comparisons. Both configurations must agree on:

- Problem identifier, seed set and seed order
- Initial design size and additional evaluation budget
- Batch size `q`, dtype and device

The strategy names must differ. The protocol checks **configured**
fairness before any run. After running both strategies with
`run_benchmark`, call `validate_trajectories(first_runs, second_runs)`
to confirm seed order, total evaluation counts, and exact equality of
initial X, observed Y, ground-truth Y, constraints, and costs.

For example:

```python
protocol = FairComparisonProtocol(random_config, sobol_config)
random_runs = run_benchmark(random_config, random_candidates, registry=registry)
sobol_runs = run_benchmark(sobol_config, sobol_candidates, registry=registry)
protocol.validate_trajectories(random_runs, sobol_runs)
```

This guards against accidentally comparing different initial designs
or unequal budgets. It does **not** prove equal model-fitting time,
hardware utilization, acquisition optimization effort, observation
noise streams after the initial design, or algorithmic hyperparameter
budgets. Report these separately. Paired performance analysis can use
`compare_paired_curves` after extracting metric curves from validated
trajectories.
