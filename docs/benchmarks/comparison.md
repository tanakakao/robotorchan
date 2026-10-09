# Seeded benchmark comparisons (Phase 12)

Aggregate equal-length evaluation-indexed metric curves using
`summarize_curves({seed: curve})`. The output contains:

- Pointwise mean across seeds.
- Sample standard deviation (`ddof=1` for at least two seeds).
- Pointwise two-sided **normal-approximation** confidence interval.
- Sorted seed identifiers for reproducible reporting.

`compare_paired_curves(first, second)` requires the **same seed set** and
computes `first[seed] - second[seed]` before aggregating. This preserves
within-seed pairing and is preferable to subtracting two independently
estimated confidence intervals.

```python
from robotorchan.benchmarks import compare_paired_curves, summarize_curves

summary = summarize_curves({0: curve_a, 1: curve_b})
paired = compare_paired_curves(strategy_a, strategy_b)
```

Both functions require equal-length 1D finite curves on the same dtype and
device. The interval is a normal approximation and is **not** an exact
Student-t interval; for one seed, the interval width is set to zero as a
descriptive convention, not a statistical confidence guarantee.

This phase does not perform significance testing, multiple-comparison
correction, nonfinite-regret handling, or resampling onto a shared cost axis.
