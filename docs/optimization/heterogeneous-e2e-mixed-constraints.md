# Phase 12: Multiple and mixed outcome constraints

Fit a two-output regression GP for Strength and Conductivity and a binary Pass GP classifier. Maximize Strength subject to two upper-bound continuous outcome constraints (Conductivity <= 0.8 and Strength <= 1.5) and Pass class-1 feasibility.

The existing mixed qEI and multiple-learned-constrained qNEI composition factories apply continuous sample-space residuals within the shared regression posterior, and multiply the resulting acquisition by classifier marginal Pass probability. Tests evaluate q=1 and q=2 acquisitions, verify PoF weighting, and generate bounded candidates through the Sobol optimizer.

**Limitations:** These are outcome constraints on model predictions, not candidate-space input constraints. Classification feasibility is a marginal probability factor, not a joint posterior sample. The qNEI baseline is not corrected for classifier feasibility. No claim of candidate feasibility or optimization superiority is made.

Run `pytest -q tests/benchmarks/test_heterogeneous_mixed_constraints_e2e.py`.
