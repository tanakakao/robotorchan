# Phase 5: Semantic numerical validation

The Phase 5 integration tests validate output resolution by name or index for Strength, Conductivity and Pass, objective direction signs, continuous constraint residuals, and classifier probability-of-feasibility routing.

The model instances are not fitted in this phase: this isolates semantic adapter behavior from training performance, which is covered in Phases 3 and 4.

Run `pytest -q tests/benchmarks/test_heterogeneous_semantics_e2e.py`.
