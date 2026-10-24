# Acquisition optimization theory–implementation mapping: Phase 3 inventory

## Baseline and separation of responsibilities

Baseline: `main` after Phase 2 exit-check merge
`b83afcc67e38d2b4dcc3c3a3ab794063f61705f6`.

| Layer | Source of truth | Responsibility |
| --- | --- | --- |
| Search-space strategy | `optim/strategies/`, `optim/embedding/`, `optim/latent/`, `optim/trust_region/` | Choose candidate coordinates, search region, and optional persistent state |
| Numerical backend | `optim/backends/` | Optimize an already constructed BoTorch `AcquisitionFunction` |
| Public convenience dispatch | `optim/dispatch.py` | Select backend with `optimizer` while rejecting unsupported argument combinations |
| Capability descriptors | `optim/capabilities.py` | Advertise backend behavior; metadata alone is not proof of feasibility |
| Domain | `optim/domains/` | Represent mixed/integer/categorical candidate variables |
| Candidate constraints | `optim/constraints/{contracts,validation,evaluation}.py` | Validate and evaluate constraints over candidate coordinates |
| One-shot handling | `optim/one_shot/` | Mixed augmented-batch initial conditions and one-shot optimizer handling |

## Named optimization dispatcher

`robotorchan.optim.optimize_acqf` accepts **eleven** named options:

| Names | Implementation dispatch | Qualification |
| --- | --- | --- |
| `botorch` | `backends/botorch.py` | Native continuous BoTorch optimization |
| `torch_adam`, `torch_adamw`, `torch_sgd` | `backends/torch.py` | PyTorch first-order gradient optimizers |
| `random`, `sobol` | `backends/sampling.py` | Sample-and-select strategies |
| `de` | `backends/differential_evolution.py` | Differential evolution |
| `cmaes` | `backends/cmaes.py` | CMA-ES |
| `ga` | `backends/genetic_algorithm.py` or `backends/mixed_genetic_algorithm.py` | Mixed GA selected by `MixedVariableSpace.is_mixed` |
| `pso` | `backends/pso.py` | Particle swarm |
| `hybrid` | `backends/hybrid.py` | Combined backend |

**Not dispatcher names:** BoTorch mixed optimizer (explicit low-level
path), mixed one-shot optimizer, NSGA-II, and search strategies. These
appear as separate public symbols or capability descriptors and must
not be mislabeled as available `optimizer=` values.

## Candidate-space constraints versus outcome constraints

`CandidateConstraints` in `optim/constraints/contracts.py` stores:

- `inequality_constraints`: BoTorch tuples `(indices, coefficients, rhs)`,
  feasible when their linear form is `>= rhs`.
- `equality_constraints`: the corresponding equality contract.
- `nonlinear_inequality_constraints`: `(callable, is_intrapoint)`,
  with feasibility `callable(X) >= 0` in either `[d]` or `[q,d]` coordinates.

The validation layer handles shape and sequential/inter-point restrictions;
`dispatch.py` rejects backend combinations beyond declared capabilities.
The `constraint_handling` descriptor differentiates native enforcement,
feasibility-first, penalty and unsupported modes. **Penalty is not native
hard feasibility**; no source-level flag by itself certifies that the
returned candidate satisfies the constraint.

An outcome constraint acts on **model predictions** and belongs in
acquisition/objective composition, not in `CandidateConstraints`.
Embeddings and latent strategies may reject original-coordinate candidate
constraints through `reject_unmapped_candidate_constraints` rather than
silently reinterpret them in reduced coordinates.

## Stateful and coordinate-changing algorithms

The `TuRBOState` and helpers in `optim/trust_region/turbo.py`
maintain success/failure counters, length expansion/shrinkage, restart
conditions and optional ARD dimension weights. `TuRBOStrategy` may
generate candidates by calling dispatch or a mixed BoTorch backend.
TuRBO is a **search-region adaptation strategy**, not a twelfth
dispatcher backend.

`REMBOStrategy`, `HeSBOStrategy`, `ALEBOStrategy`,
`BAxUSStrategy` and latent-space strategies change candidate
coordinates. Their constraint mapping and candidate reconstruction
must be reviewed separately from the numerical optimizer they invoke.

## Verification boundaries

The sources above establish routing contracts and intended semantics,
not independent numerical correctness. Phase 3 still needs the
test-to-route matrix, one-shot and mixed feasibility inspections,
public symbol inventory for strategies and direct backends, and a
fixed-scope exit check. Mathematical constraint satisfaction and
state-transition identities belong to Phase 9.

## Inspected optimizer and strategy assertion evidence

| Test module | Direct assertions | Evidence boundary |
| --- | --- | --- |
| `tests/optim/contracts/test_optimizer_dispatch.py` | Default BoTorch backend forwarding and restart/sample arguments; actual GA candidate/value shape; unknown optimizer and invalid bounds/`q` rejection; mixed GA routing; categorical PSO and integer Torch rejection; initialization-generator forwarding and unsupported-backend rejection | Mock-based routing is contract evidence, not evidence of numerical solver quality |
| `tests/optim/contracts/test_optimizer_dispatch.py` | Nonlinear bound is enforced for q=1 with both an explicit initial-condition tensor and a custom initializer | Two synthetic scalar problems do not prove general nonlinear feasibility |
| `tests/optim/integration/test_nonlinear_candidate_constraints_e2e.py` | BoTorch qLogEI returns a nonlinear-feasible candidate; linear + nonlinear + fixed-feature combination; inter-point separation with q=2; qLogNParEGO with nonlinear candidate constraint | This is actual candidate-level feasibility at 1e-6 tolerance for selected problems, **not** a global optimality guarantee or a test of all backends |
| `tests/optim/trust_region/test_turbo_strategy.py` | Success expansion, failure shrinkage and restart flag; minimum length guarding; geometric-mean normalized ARD weights and scale invariance; clipped public-space bounds; multi-fidelity coordinate preservation; named optimizer strategy route | Does not establish improvement in regret or statistical efficiency relative to global BO |

These examples deliberately distinguish dispatch/capability validation,
actual returned-candidate feasibility and optimization quality. Derivative-free
constraint backends may use penalty or feasibility-first ranking, which is
not identical to native hard-constraint enforcement.

## Fixed follow-up audit surface

The Phase 3 source-to-test matrix still needs direct inspection for
`optim/one_shot/`, specialized mixed BoTorch optimization, NSGA-II,
the complete backend-specific constraint matrix, and
`optim/embedding/`/`optim/latent/` search strategies. Formula and
state-transition reference tests remain the separate Phase 9 deliverable.
