# Contextual policy benchmark (Phase 7)

Run from the repository root after installing robotorchan:

```bash
python benchmarks/contextual_policy.py
```

The synthetic objective is `f(x, c) = 1 - (x - c)^2`, with 21 uniformly weighted
contexts on `[0, 1]`. The fixed-decision baseline uses `x=0.5`. The
context-aware **analytical oracle** uses `x=c` and is an upper bound, not a
policy learned by Bayesian optimization.

The script reports exact weighted expected values, seeded Monte Carlo estimates
using shared sampled contexts, and wall-clock time for **policy evaluation only**.
It does not time GP fitting, acquisition optimization or Study ask/tell. Runtime
measurements depend on the machine and are not regression thresholds.

This establishes a reproducible evaluation baseline for subsequent contextual BO
convergence benchmarks. No claim of superiority of learned contextual BO over
standard BO can be inferred from this oracle comparison.
