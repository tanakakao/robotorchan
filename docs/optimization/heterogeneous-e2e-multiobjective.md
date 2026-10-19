# Phase 9: Unconstrained two-objective qEHVI and qNEHVI

The benchmark's Strength and Conductivity outputs are jointly represented by a two-output exact SingleTaskGP. The model is fitted to noisy observations with known observation variance. Both objectives are maximized.

The test builds qEHVI with a BoTorch nondominated partitioning from observed objectives, and qNEHVI with baseline input points. It evaluates both acquisitions for q=1 and q=2, then generates in-bounds candidates through the existing Sobol optimizer.

Assertions cover finite and nonnegative acquisition values, valid candidate shapes, bounds, and finite benchmark truth. This phase does not include Pass/Fail constraints, which belong to Phase 10. It does not assert hypervolume improvement or optimization superiority over random search.

Run `pytest -q tests/benchmarks/test_heterogeneous_multiobjective_optimization_e2e.py`.

## Classification-constrained multiobjective validation (historical Phase 10)

Fit a two-output regression GP for Strength and Conductivity and a
separate binary GP classifier for Pass. Both regression objectives are
maximized; Pass class 1 defines feasibility. For qEHVI, the existing
composition attaches marginal classifier probability-of-feasibility (PoF)
weights. The qNEHVI composition factory does **not** currently expose a
classification-constrained path: its test manually wraps **unconstrained**
qNEHVI with `FeasibilityWeightedAcquisition` instead. These are distinct
interfaces and must not be presented as equivalent factory support.

Both tests evaluate q=1 and q=2 acquisition values and use the Sobol
optimizer to generate bounded candidates.

**Limitations:** The weighting is a product of marginal classifier PoFs,
not joint regression/classification Monte Carlo constrained hypervolume
improvement. The qNEHVI baseline is not classifier-feasibility corrected,
and its nondominated reference partitioning uses observed regression
outputs without feasibility filtering. The tests establish numerical
integration only; they do not establish feasible hypervolume improvement,
calibration, or superior search quality.

Run `pytest -q tests/benchmarks/test_heterogeneous_constrained_multiobjective_e2e.py`.
