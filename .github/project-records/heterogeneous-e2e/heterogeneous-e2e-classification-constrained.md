# Phase 8: Binary-classification-constrained qEI and qNEI

The benchmark's Strength regression model and Pass binary GP classifier are fitted separately. The semantic constraint identifies Pass class 1 as feasible.

qEI is built as a native unconstrained objective acquisition and explicitly wrapped with the existing deterministic probability-of-feasibility bridge. qNEI uses the existing multiple-learned-constraint composition function. Both are evaluated at q=1 and q=2, and the Sobol optimizer generates candidates within the benchmark bounds.

**Important limitation:** This composition multiplies the objective acquisition by marginal classifier probability of feasibility, with a product reduction across q. It is not a joint regression/classification Monte Carlo constrained improvement, and qNEI does not correct baseline feasibility. These smoke tests check integration and arithmetic consistency, not optimization superiority or calibrated feasibility.

Run `pytest -q tests/benchmarks/test_heterogeneous_classification_constrained_e2e.py`.

## Mixed continuous and classification outcome constraints (historical Phase 12)

Fit a two-output regression GP for Strength and Conductivity alongside
a separate binary GP classifier for Pass. Maximize Strength subject to
`Conductivity <= 0.8`, `Strength <= 1.5`, and Pass class-1 feasibility.
These are **outcome constraints on model predictions**, not nonlinear
candidate-space input constraints.

The mixed qEI and multiple-learned-constrained qNEI composition factories
apply the continuous residuals within samples from the shared regression
posterior, then multiply the resulting acquisition by the **marginal**
classifier Pass probability of feasibility. Tests evaluate q=1 and q=2,
check PoF weighting, and generate bounded candidates with the Sobol
optimizer.

**Limitations:** Classifier feasibility is an independent marginal PoF
factor, not a joint regression/classification posterior sample. The qNEI
baseline is not corrected for classifier feasibility. These tests do not
establish true candidate feasibility, calibrated probability estimates,
or optimizer superiority.

Run `pytest -q tests/benchmarks/test_heterogeneous_mixed_constraints_e2e.py`.
