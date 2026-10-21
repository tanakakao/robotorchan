# Benchmark reporting

The benchmark package provides a portable reporting API for evaluation-aligned
metric curves.

`make_metric_report(strategy, metric, curves, interval_method="normal")`
accepts a dictionary mapping each independent run seed to a finite 1D
PyTorch metric curve. All curves must have equal length, dtype and device.
The result contains the sorted seeds, pointwise mean, and confidence bounds.

Choose `interval_method="bootstrap"` to use the seeded
percentile bootstrap. The default `"normal"` method uses the existing
normal-approximation confidence interval.

```python
report = make_metric_report(
    "random",
    "simple_regret",
    {0: regret_seed_0, 1: regret_seed_1},
    interval_method="bootstrap",
    n_resamples=2000,
    seed=0,
)
json_text = report.to_json()
csv_text = report.to_csv()
```

Both exports are deterministic. JSON includes `schema_version=1`,
strategy, metric, seed identifiers, mean, lower/upper, confidence and
interval method. CSV contains one row per **zero-based metric curve
index** and can be plotted with external tools. The index is not
automatically mapped to the number of additional BO evaluations.

The reporting API does not itself execute benchmarks, produce figures,
or guarantee fairness. Validate the experimental pairing and fairness
contract before cross-strategy comparisons, and use paired statistics
to assess differences. See the
[Comparative fairness protocol](comparative-fairness-protocol.md) and
[Multi-seed statistics](multiseed_statistics.md). For plotting, draw
`mean` with a shaded band between `lower` and `upper`.
