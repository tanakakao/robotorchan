# Phase 9: Unconstrained two-objective qEHVI and qNEHVI

The benchmark's Strength and Conductivity outputs are jointly represented by a two-output exact SingleTaskGP. The model is fitted to noisy observations with known observation variance. Both objectives are maximized.

The test builds qEHVI with a BoTorch nondominated partitioning from observed objectives, and qNEHVI with baseline input points. It evaluates both acquisitions for q=1 and q=2, then generates in-bounds candidates through the existing Sobol optimizer.

Assertions cover finite and nonnegative acquisition values, valid candidate shapes, bounds, and finite benchmark truth. This phase does not include Pass/Fail constraints, which belong to Phase 10. It does not assert hypervolume improvement or optimization superiority over random search.

Run `pytest -q tests/benchmarks/test_heterogeneous_multiobjective_optimization_e2e.py`.
