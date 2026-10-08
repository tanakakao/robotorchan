# Phase 15: Paired multi-seed statistical evaluation

The Phase 14 fixed-budget runner is now reusable from `robotorchan.benchmarks.heterogeneous_baselines.run_strategy`. `compare_strategies` evaluates qEI, uniform random, and scrambled Sobol across distinct seeds, with an identical initial design and noisy initial observations for all three strategies within each seed. It returns a tensor of noise-free best-so-far Strength trajectories per strategy, shaped `(n_seeds, steps + 1)`.

`summarize_trajectories` computes per-evaluation mean, unbiased sample standard deviation (`ddof=1`), and standard error (`std / sqrt(n_seeds)`). The smoke test runs two seeds and two sequential evaluations; production-quality statistical comparisons require more seeds and a prespecified evaluation budget. All methods receive the same number of objective evaluations, but computational overhead is not normalized.

**Interpretation:** The standard error describes uncertainty in the estimated mean across seeds, not a confidence interval or predictive model uncertainty. The tests verify aggregation and reproducible data contracts, not superiority or statistical significance. Report per-seed trajectories and paired differences before claiming an advantage.

Run `pytest -q tests/benchmarks/test_heterogeneous_multiseed_e2e.py tests/benchmarks/test_heterogeneous_baseline_comparison_e2e.py`.
