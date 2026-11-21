# Scientific validity audit — Phase 4: optimizer and candidate-constraint integration

Status: **scoped audit complete; limitations retained** (2026-11-21).

This report covers targeted candidate-selection and constraint-handling references for representative optimizers. It does **not** certify global optimizer convergence, all backends, every mixed-variable configuration, or downstream Bayesian optimization performance. Acquisition-function mathematics is documented separately in [scientific-validity-phase3.md](scientific-validity-phase3.md).

## Evidence and scope

| Area | Evidence | Interpretation and boundary |
| --- | --- | --- |
| Continuous constrained q=2 sampling | PR #1965; `tests/optim/backends/test_phase4_constrained_sampling_reference.py` | Seeded Random/Sobol candidate pools; independent linear intrapoint/interpoint feasibility filter and exact feasible argmax. Validates selection **within sampled pool**, not continuous global optimality |
| Mixed q=2 Sobol with fixed categorical value | PR #1966; `test_phase4_mixed_sobol_constrained_reference.py` | Independent integer/category inverse-CDF mapping, fixed-feature substitution, linear feasibility and argmax; specific dimensions and values only |
| Nonlinear intrapoint + interpoint q=2 sampling | PR #1967; `test_phase4_nonlinear_interpoint_sampling_reference.py` | Independent product constraint and per-point bound; Random/Sobol feasible argmax reference. Does not cover arbitrary callable behavior |
| Impossible nonlinear q=2 constraint: sampling + DE | PR #1968; `test_phase4_infeasible_qbatch_contract.py` | Random/Sobol/DE must raise rather than report infeasible solutions as valid |
| Impossible nonlinear q=2 constraint: GA + PSO | PR #1969; `test_phase4_ga_pso_infeasible_qbatch.py` | GA/PSO must raise on the same mathematically impossible joint constraint; no convergence guarantee |
| Existing mixed and q-batch contracts | `tests/optim/contracts/test_mixed_q_batch_interpoint_constraints.py`, `tests/optim/contracts/test_mixed_de_q_batch_interpoint_constraints.py` | Integration checks cover selected optimizer/domain/constraint combinations; not a full Cartesian-product audit |
| Existing nonlinear BoTorch integration | `tests/optim/integration/test_nonlinear_candidate_constraints_e2e.py` | Selected GP acquisition with nonlinear constraints and joint q=2; stochastic fitting and optimization remain separate reliability questions |
| Constraint violation and ordering | `tests/optim/constraints/test_phase9_constraint_violation_reference.py` | Independent violation arithmetic and feasibility-first rank checks; numerical tolerance affects near-boundary classification |

## Mathematical and implementation boundaries

1. **Sampled-pool optimum versus global optimum.** Exact agreement with a feasible argmax over a seeded pool verifies sampling-backend selection, not convergence to the global maximum of the acquisition over the continuous feasible domain.
2. **Feasibility is a separate property.** A high acquisition value does not compensate for a violated hard candidate constraint. The tested strict backends reject a provably infeasible q=2 problem; penalty-based behavior in other APIs must not be assumed equivalent.
3. **Joint q-batch semantics matter.** Interpoint constraints act on the entire `[q,d]` batch. They cannot be reduced to independent pointwise constraints without changing the feasible set.
4. **Mixed-variable mapping matters.** Integer rounding, categorical encoding/decoding, and fixed-feature substitution must be applied consistently before objective and constraint evaluation. The tested mapping covers a specific Sobol configuration, not all optimizers.
5. **Numerical tolerances matter.** Exact sampled-pool references use fixed seeds; optimizers with iterative floating-point search may differ in boundary tolerance, numerical behavior, and feasible-point discovery.
6. **Infeasible versus not found.** A finite candidate search that finds no feasible point cannot generally prove that the feasible set is empty. The impossible-problem tests use a separate mathematical proof from the bounds; their error behavior must not be interpreted as a general infeasibility certificate.

## Remaining risk-based validation

- Independent known-optimum references for gradient-based BoTorch and selected iterative optimizers under feasible constraints, with justified numerical tolerances.
- q>2, nonlinear equality tolerances, near-boundary feasibility, and mixed-variable configurations not covered by the scoped tests.
- Clear capability/error contracts for unsupported combinations and penalty-versus-strict feasibility across all public optimizer dispatch paths.
- Multi-seed end-to-end Bayesian optimization regret and active-learning sample-efficiency benchmarks, including meaningful baselines and reproducible uncertainty estimates.
- Runtime and numerical-stability profiling for high-dimensional or large-q constrained optimization, prioritized by actual usage.

## Phase 4 exit decision

**Close the scoped optimizer/constraint audit with explicit limitations.** Representative independent references now cover constrained sampled-pool selection, mixed-domain mapping, nonlinear joint q-batch feasibility, and rejection of a provably impossible problem across Random, Sobol, DE, GA and PSO. Existing integration tests provide additional selected BoTorch and mixed-optimizer coverage. Do **not** claim universal optimizer correctness, global optimality, or complete backend compatibility.

Proceed to end-to-end statistical reliability and reproducible benchmark assessment without multiplying redundant contract tests.

## PR references

- https://github.com/tanakakao/robotorchan/pull/1965
- https://github.com/tanakakao/robotorchan/pull/1966
- https://github.com/tanakakao/robotorchan/pull/1967
- https://github.com/tanakakao/robotorchan/pull/1968
- https://github.com/tanakakao/robotorchan/pull/1969
