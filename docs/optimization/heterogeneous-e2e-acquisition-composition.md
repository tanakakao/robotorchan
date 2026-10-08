# Phase 6: Acquisition composition numerical validation

This phase validates existing acquisition composition on the Phase 2 benchmark. The two regression outputs share a multi-output SingleTaskGP; Pass is represented by a separate binary GP classifier.

Tests verify output ownership and classifier feasibility binding, and evaluate native qEI and qNEI at q=1 and q=3 with batch size 2. A separate qEHVI test exercises the two regression objectives with a nondominated partitioning and reference point. Outputs must be finite, nonnegative and have the expected batch shape.

These tests intentionally do not claim classifier-constrained qEHVI or qNEHVI support, nor do they assess optimization quality. The classifier constraint is validated at the composition-plan level here; constrained acquisition evaluation belongs to later phases.

Run `pytest -q tests/benchmarks/test_heterogeneous_acquisition_composition_e2e.py`.
