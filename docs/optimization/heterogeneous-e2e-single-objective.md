# Phase 7 single-objective validation

Tests fit exact regression GPs to synthetic Strength and Conductivity observations. For each output they construct qEI and qNEI through the existing acquisition composition layer and generate a candidate using the Sobol optimizer.

Assertions cover candidate shape, bounds, finite acquisition values and finite benchmark truth. These are integration checks rather than evidence of optimization superiority.

Run: `pytest -q tests/benchmarks/test_heterogeneous_single_objective_optimization_e2e.py`.
