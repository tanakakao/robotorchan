# Phase 5: paired binary GP active-learning evidence

This runner extends the existing single-seed closed-loop smoke example. It
**does not** claim that predictive entropy outperforms passive policies.

Run a cheap manual check first:

```bash
python examples/benchmarks/binary_gp_active_learning_multiseed.py \
  --output artifacts/gp-al-smoke --git-sha <exact-main-sha> \
  --seeds 1 --budget 1 --epochs 2
```

The full protocol defaults to ten paired seeds, fifty additional observations,
and eight training epochs per GP refit. It is computationally expensive and is
not part of CI. Use a fresh output directory for each run.

Each strategy uses the same initial twenty labels, same 256-point Sobol pool,
same fixed independent 1,024-point reference set, and the same binary GP
predictor and training budget. Random draws an available pool member uniformly,
Sobol chooses the first remaining member, and GP predictive entropy chooses the
highest-scoring member. Only the selected candidate is labelled. The pool is
finite: the budget must not exceed 256. The experiment does **not** compare
different surrogate families.

Files: `manifest.json` contains configuration and completion status;
`curves.csv` contains per-seed trajectories; `paired_differences.csv` contains
same-seed strategy differences against Random; `summary.csv` contains mean,
sample standard deviation and number of complete seeds; `failures.json`
contains errors and tracebacks. Failed seeds are excluded from all strategies
to preserve paired comparisons. Partial failures are recorded, not silently
treated as completed evidence.

Interpret accuracy as higher-is-better and Brier as lower-is-better. The
reported standard deviations describe seed variability, **not** uncertainty
intervals or a statistical significance test. Full results must be generated
and inspected before making empirical performance claims.
