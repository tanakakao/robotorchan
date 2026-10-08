# Phase 10: Classification-constrained multiobjective candidates

Fit a two-output regression GP for Strength and Conductivity and a binary GP classifier for Pass. Both regression outputs are maximized and class 1 is feasible.

For qEHVI, the existing constrained acquisition composition attaches marginal Pass probability weights. For qNEHVI, which does not currently expose a classification-constrained composition factory, this test explicitly wraps the existing unconstrained qNEHVI with the existing FeasibilityWeightedAcquisition adapter. Both paths validate q=1 and q=2 acquisition values and generate candidates with the Sobol optimizer.

**Limitations:** Both methods weight the objective acquisition by a product of marginal classifier probabilities. This is not joint posterior Monte Carlo constrained hypervolume improvement. In particular, the qNEHVI baseline is not feasibility corrected. The reference partitioning is based on observed regression outputs without feasibility filtering. The test asserts interface and numerical integration, not feasible hypervolume improvement or superior search quality.

Run `pytest -q tests/benchmarks/test_heterogeneous_constrained_multiobjective_e2e.py`.
