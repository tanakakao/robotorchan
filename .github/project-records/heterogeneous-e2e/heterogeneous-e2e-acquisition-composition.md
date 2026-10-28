# Phase 6: Acquisition composition numerical validation

This phase validates existing acquisition composition on the Phase 2 benchmark. The two regression outputs share a multi-output SingleTaskGP; Pass is represented by a separate binary GP classifier.

Tests verify output ownership and classifier feasibility binding, and evaluate native qEI and qNEI at q=1 and q=3 with batch size 2. A separate qEHVI test exercises the two regression objectives with a nondominated partitioning and reference point. Outputs must be finite, nonnegative and have the expected batch shape.

These tests intentionally do not claim classifier-constrained qEHVI or qNEHVI support, nor do they assess optimization quality. The classifier constraint is validated at the composition-plan level here; constrained acquisition evaluation belongs to later phases.

Run `pytest -q tests/benchmarks/test_heterogeneous_acquisition_composition_e2e.py`.

## Semantic adapter validation (historical Phase 5)

The benchmark resolves Strength, Conductivity, and Pass by output name or
index. Integration tests check objective-direction signs, continuous
constraint residuals, and binary classifier probability-of-feasibility
routing. Models are deliberately **not fitted** in this semantic-only test,
which isolates the adapter contract from training quality.

Run `pytest -q tests/benchmarks/test_heterogeneous_semantics_e2e.py`.

## Single-objective candidate-generation validation (historical Phase 7)

Separate exact regression GPs are fitted to synthetic Strength and
Conductivity observations. For each output, the composition layer builds
qEI and qNEI and a Sobol optimizer generates a candidate. Tests check
candidate shape, bounds, finite acquisition values, and finite benchmark
truth. They do **not** demonstrate optimizer superiority.

Run `pytest -q tests/benchmarks/test_heterogeneous_single_objective_optimization_e2e.py`.
