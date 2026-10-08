# Phase 8: Binary-classification-constrained qEI and qNEI

The benchmark's Strength regression model and Pass binary GP classifier are fitted separately. The semantic constraint identifies Pass class 1 as feasible.

qEI is built as a native unconstrained objective acquisition and explicitly wrapped with the existing deterministic probability-of-feasibility bridge. qNEI uses the existing multiple-learned-constraint composition function. Both are evaluated at q=1 and q=2, and the Sobol optimizer generates candidates within the benchmark bounds.

**Important limitation:** This composition multiplies the objective acquisition by marginal classifier probability of feasibility, with a product reduction across q. It is not a joint regression/classification Monte Carlo constrained improvement, and qNEI does not correct baseline feasibility. These smoke tests check integration and arithmetic consistency, not optimization superiority or calibrated feasibility.

Run `pytest -q tests/benchmarks/test_heterogeneous_classification_constrained_e2e.py`.
