# Phase 19: Heterogeneous E2E benchmark report

## Purpose and reproducibility

Run `notebooks/heterogeneous_bo_benchmark_phase19.ipynb` from the repository root in an environment with robotorchan, PyTorch, BoTorch, Matplotlib, and Jupyter installed. This is a **reproducible report generator**, not a report containing measured performance numbers. Do not interpret unexecuted notebook code as observed results.

The notebook runs `compare_strategies` with three paired seeds (1901, 1902, 1903), 12 initial design points per seed, and three additional objective evaluations per strategy. Change the parameters explicitly for a larger study and record the environment versions, hardware, seed list, budget, and git commit SHA in any published report.

## Experimental contract

| Property | Definition |
| --- | --- |
| Search domain | Continuous unit cube in three dimensions |
| Objective | Maximize synthetic Strength |
| Methods | Uniform Random, scrambled Sobol, refitted qEI |
| Initial data | Same design and noisy observations across methods for each seed |
| Evaluation budget | 12 shared initial + 3 additional per method and seed |
| Primary metric | Noise-free best-so-far Strength across evaluated points |
| Uncertainty summary | Across-seed sample standard deviation and standard error |
| Paired comparison | qEI minus Random final best-so-far for the same seed |
| Candidate optimizer for qEI | Sobol sampling, 32 raw candidate sets per iteration |

Each method receives an equal number of objective evaluations, not equal computational time. Random and Sobol do not fit surrogates. The truth metric is only used for offline evaluation, never as the acquisition objective. The notebook does not test Pass feasibility or Conductivity multiobjective optimization.

## How to interpret a run

1. Verify all trajectories have shape `(n_seeds, steps + 1)`, finite values, and nondecreasing best-so-far values.
2. Inspect per-seed final values and trajectories before the aggregate plot.
3. Report mean and sample standard deviation. The standard error is `std / sqrt(n_seeds)`; it is not a confidence interval.
4. Inspect paired differences. Do not claim significance from overlapping or nonoverlapping SEM bands.
5. Record runtime separately if comparing compute efficiency. For robust claims, prespecify a larger number of seeds and statistical analysis.

## E2E coverage and outstanding work

| Phase | Coverage | Caveat |
| --- | --- | --- |
| 13 | Sequential model fit, acquisition, candidate, observation | Smoke-test budget |
| 14 | Random / Sobol / qEI fixed-budget baselines | Single seed |
| 15 | Paired seeds, sample std, SEM | Smoke test has two seeds |
| 16 | Exact / variational regression integration | No accuracy comparison |
| 17 | Dtype, q-batch, fixed features, optional CUDA | Pending points and nonlinear constraints not covered |
| 18 | Invalid benchmark inputs and statistics | Optimizer failures not covered |
| 19 | Reproducible notebook and reporting protocol | No benchmark performance claim |

### Commands

```bash
pytest -q tests/benchmarks/test_heterogeneous_baseline_comparison_e2e.py tests/benchmarks/test_heterogeneous_multiseed_e2e.py
pytest -q tests/benchmarks/test_heterogeneous_benchmark_notebook.py
```

The Phase 20 audit should check the full suite, notebook execution in CI, and any outstanding heterogeneous classification-constrained / multiobjective benchmark gaps before making a release-readiness claim.
