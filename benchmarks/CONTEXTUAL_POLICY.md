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

## Learned Study comparison (exploratory)

```bash
python benchmarks/contextual_study.py
```

This additional script runs the **existing** contextual `Study.ask(context=...)`
and a context-blind `Study.ask()` on the same sequence of contexts and with
the same number of objective evaluations. Both use the default GP/acquisition
and BoTorch optimizer; seeds 7, 19, and 31 are printed separately.

The comparison reports mean and best **observed training rewards** and total
sequential training time. These metrics are not out-of-sample policy values:
the context-blind Study cannot condition on contexts and both methods see
different candidate decisions. The experiment is exploratory and does not
guarantee contextual superiority or establish convergence rates. A stronger
future benchmark would compare independently recommended policies on a held-out
context distribution, with uncertainty intervals across more seeds.
