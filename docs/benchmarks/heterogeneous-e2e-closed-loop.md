# Phase 13: Closed-loop Bayesian optimization

This test executes three sequential Bayesian optimization iterations on the synthetic Strength objective, using either qEI or qNEI. Each iteration refits an exact GP on all observations, constructs a fresh acquisition through the existing semantic composition layer, selects one in-bounds point using the Sobol optimizer, observes the synthetic benchmark with a deterministic iteration-specific noise seed, and appends the result to the training dataset.

The test checks that the dataset grows from 16 to 19 rows, initial observations remain unchanged, every selected candidate and observation is finite, and the recorded history matches appended rows. For qNEI, the baseline is refreshed from the current training inputs on every iteration.

**Limitations:** This is a small, unconstrained, single-objective closed-loop integration smoke test. It does not establish convergence, superiority to random search, statistical repeatability across seeds, or classifier-constrained closed-loop behavior. Those require later benchmark phases.

Run `pytest -q tests/benchmarks/test_heterogeneous_closed_loop_e2e.py`.
