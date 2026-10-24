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

## Specialized optimization paths: inspected source and assertions

The specialist paths are **not additional names** of the scalar
`optim.optimize_acqf(optimizer=...)` dispatcher.

| Source and tests | Implementation/numerical method | Assertions inspected | Unverified / unsupported |
| --- | --- | --- | --- |
| `optim/one_shot/mixed.py` / `tests/optim/one_shot/test_mixed.py` | Locally enumerates categorical choices for **each row** of an augmented one-shot q-batch, flattens the augmented coordinates, delegates each continuous subproblem to BoTorch `optimize_acqf`, and extracts actual candidates | A synthetic one-shot problem finds an actual/fantasy **cross-category** assignment and value > 0.9; assignment-count cap and invalid categorical domains rejected | **q=1 only**, exponential assignment count with augmented size; not general mixed qKG performance or proof of optimum |
| `optim/backends/nsga2.py` / `tests/optim/backends/test_nsga2_backend.py` | Local nondominated ranking, crowding distance, tournament, crossover, mutation and environmental selection for **vector-valued** maximization | Returned tradeoff set within bounds and mutually nondominated; same-seed reproducibility; scalar objective rejected; dtype/device retained | Scalar BoTorch qEHVI/qNEHVI acquisition values do **not** require this API; hypervolume convergence and reference Pareto front coverage unverified |
| `optim/backends/botorch.py` / `tests/optim/backends/test_botorch_optimizer_backend.py` | Native BoTorch continuous and mixed solver forwarding with local constraint and initialization validation | Native argument forwarding; fixed features under sequential q; mixed rejects interpoint nonlinear constraints before delegation; actual continuous q=1/joint-q nonlinear and interpoint feasibility within 1e-6 on synthetic cases | Mocked forwarding differs from actual feasibility; unrestricted optimizer convergence and cross-backend equivalence unverified |

### Coordinate-changing high-dimensional strategies

| Strategy and tests | Source-specific behavior and test evidence | Numerical limits |
| --- | --- | --- |
| REMBO / `tests/optim/embedding/test_rembo_strategy.py` | Repeatable normalized embedding, projection to public bounds, invariance to input units and original-space candidate API | Empirical effective-dimensionality and regret remain unverified |
| ALEBO / `tests/optim/embedding/test_alebo_strategy.py` | Reproducible projection, polytope feasibility, refusal to clip truly infeasible points, constrained feasible sampling and original-space optimized candidate | Polytope reference algebra and complex high-dimensional coverage not independently certified |
| BAxUS / `tests/optim/embedding/test_baxus_strategy.py` and `test_baxus_reference_schedule.py` | Signed sparse embedding, repeatable expansion/splitting, state adaptation, original-space candidates, and **500-dimensional reference schedule** checks | A schedule agreeing with a tutorial is not a benchmark proof of sample efficiency or of all state-transition numerics |

The embedding and trust-region algorithms choose a **search geometry**;
they may delegate numerical acquisition optimization. A strategy test that
checks original-coordinate reconstruction is not evidence of a distinct
acquisition function or backend optimizer. Nonlinear public-coordinate
constraints are rejected if a strategy has no mathematically valid mapping
into its embedded coordinates.

### Remaining Phase 3 audit surface

Trace HeSBO, remaining latent/reduced strategies, specialized backend
constraint-handling modes, and the native BoTorch acquisition/optimizer
call-site inventory. Then run the finite inventory/CI exit check. The
independent mathematical identities remain reserved for Phase 9.

## HeSBO, latent-space reconstruction and backend constraint semantics

The following source and named test modules were inspected directly.
These are **contract-level assertions**, not proof of optimality.

| Route and assertion source | Mathematical/implementation responsibility | Confirmed tests and limitations |
| --- | --- | --- |
| `HeSBOStrategy` (`optim/embedding/hesbo.py`; `tests/optim/embedding/test_hesbo_strategy.py`) | Sparse signed hash matrix maps each public feature to one embedded coordinate; normalized projection is clipped to the original box before evaluating the **original-space** acquisition. Continuous optimization delegates to BoTorch `optimize_acqf`. | One signed nonzero per input row; seed repeatability; original-space bounded projection and candidate metadata; invalid configuration/`q` guards. Embedding collision risk, clipping distortion and regret are not quantified. |
| `LatentSpaceStrategy` (`optim/latent/strategy.py`; `tests/optim/latent/test_latent_space_strategy.py`) | Fitted reconstruction maps latent optimizer coordinates back to public inputs, clamps public bounds and re-evaluates the original acquisition. Returns projection and round-trip distortion diagnostics. | PCA reconstruction yields bounded public-space candidates; random projection preserves original acquisition contract; wrong dimensions and invalid optimizer settings rejected. Round-trip distortion is reported, not proven negligible. |
| Input constraint boundary (both strategies) | `reject_unmapped_candidate_constraints` refuses supplied public-space candidate constraints before coordinate transformation. | Latent test explicitly rejects unmapped nonlinear constraints; HeSBO implementation calls the same guard. A box-clamped candidate is **not** automatically feasible for an arbitrary nonlinear input constraint. |
| Backend capability declarations (`optim/capabilities.py`; `tests/optim/test_optimizer_capabilities.py`) | Separate native constraints, feasibility-first ranking, penalties and unsupported combinations. BoTorch mixed advertises nonlinear intrapoint but not interpoint nonlinear constraints; PSO does not advertise categorical mixed optimization. | Test names explicitly cover these capability distinctions, including GPU claim limits. Flags alone cannot guarantee a returned candidate is feasible, or validate penalty tuning. |

### Phase 3 evidence limits

The inspected strategy tests exercise shape, determinism, feasibility of
public **box** bounds and model/acquisition call compatibility; they do
not validate the theoretical effective dimension or sample complexity.
For the exact equations and independent numerical identities, use the
planned Phase 9 optimization verification, without adding another phase.

Phase 3 still requires a finite cross-family reconciliation of exposed
optimization/acquisition names, delegated upstream paths and an explicit
CI-backed exit decision. Remaining source-to-test omissions should be
recorded as unverified, not implicitly treated as passed.

## Phase 3 registry reconciliation gate

The public scalar dispatcher `OptimizerName` and the concrete
`OPTIMIZER_CAPABILITIES` keys are checked for exact equality in
`tests/optim/test_optimizer_capabilities.py`. The same test verifies
that the specialist `optimize_vector_nsga2` and
`optimize_mixed_one_shot_acqf` are callable but **not** dispatcher names,
and that representative embedding/latent/TuRBO strategies remain public.
The acquisition registry ownership split is checked in
`tests/acquisition/test_api_inventory.py`.

These checks address Phase 3 namespace and metadata reconciliation, not
backend numerical equivalence. All remaining formula-level checks belong
to Phases 8–9. Phase 3 exit requires successful repository CI and a
source-to-test/known-limit signoff; only then advance to Phase 4.
