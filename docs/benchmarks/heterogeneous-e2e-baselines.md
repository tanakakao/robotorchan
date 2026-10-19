# Phase 14: Random and Sobol baselines

Compare qEI-based closed-loop Bayesian optimization with uniform random search and a scrambled Sobol sequence on the synthetic Strength maximization benchmark. All strategies start from the same 12 input points, share their initial noisy observations, and receive exactly three additional objective evaluations. Each evaluation appends a noisy observation; qEI refits its GP before every candidate selection.

The recorded metric is **best-so-far noise-free Strength** across all evaluated inputs, including the shared initial design. This is an oracle benchmark metric evaluated only after candidates are chosen, never used to select qEI candidates. The test asserts shape, bounds, finite metrics, nondecreasing best-so-far curves, equal evaluation budgets, and correct final metric reconstruction.

The Sobol candidate baseline uses a persistent scrambled SobolEngine. The qEI acquisition optimizer also uses Sobol search, but this is distinct from the Sobol baseline strategy. Seeds are fixed for reproducibility. This test does not require qEI to outperform either baseline; statistical comparisons need multiple seeds and uncertainty summaries in Phase 15.

Run `pytest -q tests/benchmarks/test_heterogeneous_baseline_comparison_e2e.py`.
