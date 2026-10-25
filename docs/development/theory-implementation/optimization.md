# Acquisition optimization theory–implementation mapping: Phase 3 inventory

> **Navigation:** [Mapping index](README.md) · [Canonical theory](../../theory/README.md) · [Optimizer mathematical verification](verification.md) · [Other mappings](README.md#reader-navigation-phase-10)

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

## Phase 6 — named optimizer mathematical and dispatch mapping

This section maps the public scalar-acquisition
dispatcher to concrete optimization procedures.
All eleven `OptimizerName` values are inspected
against `src/robotorchan/optim/dispatch.py`.
The acquisition `A(X)` is **maximized**; backend
choice does not change the underlying acquisition
formula.

| `optimizer=` | Optimization method / mathematical object | Concrete implementation and ownership |
| --- | --- | --- |
| `botorch` | Multi-start continuous acquisition maximization, with optional sequential q and native candidate constraints. | **Delegated** to `backends/botorch.py::optimize_acqf_botorch` and upstream BoTorch. A mixed `MixedVariableSpace` is rejected on this named route. |
| `torch_adam` | Differentiable first-order ascent via Adam updates on candidate coordinates. | **Extended** `backends/torch.py::optimize_acqf_torch`: upstream `botorch.optim.optimize_acqf` with `gen_candidates_torch` and `torch.optim.Adam`. |
| `torch_adamw` | AdamW adaptive gradient updates with decoupled weight decay. | Same upstream BoTorch candidate generator with `torch.optim.AdamW`; **Delegated/Extended**. |
| `torch_sgd` | Stochastic-gradient optimization of candidate coordinates. | Same upstream BoTorch candidate generator with `torch.optim.SGD`; **Delegated/Extended**. |
| `random` | Draw `num_samples` independent uniform candidate q-batches and select the highest acquisition value. | **Implemented** in `backends/sampling.py` using `torch.rand`, `argmax`. |
| `sobol` | Draw scrambled Sobol q-batches in `q*d` dimensions and select the highest acquisition value. | **Implemented** in `backends/sampling.py` using `torch.quasirandom.SobolEngine`, `argmax`. |
| `de` | Differential-evolution population search. | **Implemented** backend `backends/differential_evolution.py`; routed through `optimize_acqf_de`. |
| `cmaes` | Covariance-matrix adaptation evolution strategy. | **Implemented** backend `backends/cmaes.py`; routed through `optimize_acqf_cmaes`. |
| `ga` | Population-based genetic search; mixed-variable GA when the declared domain is mixed. | **Implemented** in `backends/genetic_algorithm.py` or `backends/mixed_genetic_algorithm.py`, selected by `variable_space.is_mixed`. |
| `pso` | Particle-swarm candidate optimization. | **Implemented** in `backends/pso.py`, called through `optimize_acqf_pso`. |
| `hybrid` | Combined search backend rather than a distinct acquisition function. | **Implemented** in `backends/hybrid.py`, called through `optimize_acqf_hybrid`; stage-level algorithm details require backend-specific audit. |

### Exact dispatch and mathematical limitations

- `botorch` forwards `num_restarts`,
  `raw_samples`, BoTorch `options`, candidate
  constraints, fixed features, initial conditions,
  `sequential`, and initial-condition generator
  arguments to the native-backed route.
- The three `torch_*` routes **reuse BoTorch**
  restart and best-candidate selection but replace
  local candidate generation with
  `gen_candidates_torch`. They reject nonempty
  candidate constraints and direct `seed` /
  `sequential` / custom `ic_generator` arguments.
  AdamW's weight decay must not be described as
  a Gaussian-process prior or BO regularizer.
- `random` and `sobol` score complete
  `[q,d]` batches; for a mixed domain, random
  samples integer/categorical values from legal
  discrete sets, while Sobol maps unit coordinates
  through equal-width inverse-CDF bins. Neither
  procedure proves global acquisition optimality.
- The named derivative-free backends use joint
  q-batch optimization; they reject BoTorch-only
  initialization arguments and `sequential=True`.
  Their constraint enforcement depends on each
  backend's explicit capability/handling mode.
- The `optimizer=` dispatcher does **not**
  include NSGA-II, native mixed BoTorch, or
  mixed one-shot optimization. These are separate
  public or low-level routes, as recorded in the
  Phase 3 inventory above.

**Source and tests:** `src/robotorchan/optim/dispatch.py`,
`src/robotorchan/optim/backends/{botorch,torch,sampling,differential_evolution,cmaes,genetic_algorithm,mixed_genetic_algorithm,pso,hybrid}.py`,
`tests/optim/contracts/test_optimizer_dispatch.py`,
`tests/optim/test_optimizer_capabilities.py`.
This is source-level mapping; Phase 9 retains
independent numerical correctness, feasibility,
and optimizer quality checks.

## Phase 6 — specialist optimizers and candidate-constraint semantics

These routes are **not** additional
`optimizer=` values in the eleven-name scalar
dispatcher.

| Route | Algorithm and concrete source | Verified scope and limitation |
| --- | --- | --- |
| Native mixed BoTorch | `backends/botorch.py::optimize_acqf_mixed_botorch` delegates to upstream `botorch.optim.optimize_acqf_mixed` with `fixed_features_list`. **Delegated/Extended**. | Enumerated mixed fixed-feature configurations; inter-point nonlinear constraints are rejected explicitly. Nonlinear constraints require feasible initial conditions or an initializer. |
| Mixed one-shot | `one_shot/mixed.py::optimize_mixed_one_shot_acqf` enumerates **row-specific categorical assignments across the augmented q-batch**, flattens each continuous subproblem and delegates to native BoTorch `optimize_acqf`. **Implemented/Delegated**. | Only public `q=1`; assignment count is `(product of category counts) ** augmented_q`, guarded by `max_assignments`. Calls `extract_candidates` after optimization. |
| Vector-valued NSGA-II | `backends/nsga2.py::optimize_vector_nsga2` performs nondominated sorting, crowding distance, binary tournament, arithmetic crossover, mutation and elitist environmental selection. **Implemented**. | Requires `[population,m]` objective values with `m>=2`, maximization, and returns rank-zero population. Not a scalar qEHVI/qNEHVI optimizer. |

### Constraint handling is part of the algorithm

The contracts in `optim/capabilities.py`
distinguish the following enforcement methods.
A capability flag is **not** proof of feasible
output for arbitrary inputs.

| Route | Linear / nonlinear candidate constraints | Meaning |
| --- | --- | --- |
| Continuous BoTorch | `NATIVE`, including inter-point nonlinear | Forwarded to BoTorch with input validation and nonlinear initial-condition requirements. |
| Mixed BoTorch | `NATIVE` for supported intrapoint nonlinear; inter-point nonlinear unsupported | Cannot silently convert a q-coupled nonlinear constraint into independent per-point constraints. |
| PyTorch Adam/AdamW/SGD | `UNSUPPORTED` | `optimize_acqf_torch` explicitly rejects nonempty candidate constraints. |
| Random / Sobol | `UNSUPPORTED` | Sampling and argmax alone do not implement arbitrary candidate constraint rejection. |
| DE, GA, Mixed GA | `FEASIBILITY_FIRST` | Constraint violations affect selection/ranking; this is not native hard-constraint enforcement. |
| CMA-ES, PSO, Hybrid | `PENALTY` | Penalized objective trades acquisition value against constraint violation; feasibility is not guaranteed. |
| NSGA-II | No candidate-constraint interface in the inspected vector API | Its nondominated sorting is over supplied objectives, not an implicit constraint-domination policy. |

### Tests and mathematical verification boundary

- `tests/optim/one_shot/test_mixed.py`
  exercises category assignments and augmented
  one-shot extraction; it does not establish
  general qKG optimality.
- `tests/optim/backends/test_nsga2_backend.py`
  checks a nondominated tradeoff set and
  reproducibility; it does not establish
  Pareto-front convergence.
- `tests/optim/backends/test_botorch_optimizer_backend.py`
  and
  `tests/optim/integration/test_nonlinear_candidate_constraints_e2e.py`
  provide forwarding and selected actual-feasibility
  evidence. They do not prove arbitrary constrained
  solver convergence.
- `tests/optim/test_optimizer_capabilities.py`
  checks declared capability contracts. Actual
  constraint satisfaction and algorithmic
  convergence remain Phase 9.

## Phase 6 — stateful and coordinate-changing search strategies

These methods control **where** an acquisition
is optimized, or which coordinate system it
receives. They are not additional values of
`OptimizerName`, and they do not replace the
acquisition's probabilistic objective.

| Strategy | Theory object and local source | Implementation status and limits |
| --- | --- | --- |
| `TuRBOStrategy` | Adaptive local trust region around an incumbent; `trust_region/turbo.py` tracks successes, failures, region length and restart state. | **Implemented/Extended**. State update doubles length after `success_tolerance`, halves it after `failure_tolerance`, and triggers restart below `length_min`. The strategy delegates candidate optimization to named or mixed BoTorch routes. |
| TuRBO ARD geometry | Axis widths scale with positive dimension weights normalized to geometric mean one; `turbo_dimension_weights_from_model` reads public-space kernel lengthscales. | **Implemented**. Reduced-space lengthscales are rejected as public-space weights; bounds scale and clipping are explicit. |
| `REMBOStrategy` | Fixed low-dimensional random embedding followed by public-space projection; `embedding/rembo.py`. | **Implemented** embedding strategy; effective-dimension assumptions and regret unverified. |
| `HeSBOStrategy` | Sparse signed hash embedding, with public-space clipping; `embedding/hesbo.py`. | **Implemented**; collisions and clipping distortion can affect search. |
| `ALEBOStrategy` | Linear embedding with feasible target-space polytope handling; `embedding/alebo.py`. | **Implemented**; mapping and feasible-sampling tests are contract evidence, not full optimization guarantees. |
| `BAxUSStrategy` | Budget-aware adaptive sparse embedding and trust-region state; `embedding/baxus.py`. | **Implemented**. `BAxUSState` sets target dimension and split budget; signed sparse bins are split on restart and target observations carried into child coordinates. |
| `BAxUSThompsonSamplingStrategy` | Thompson candidate selection over BAxUS's changing target geometry; `optim/embedding/`. | **Extended** strategy route; does not imply an independent GP posterior or new acquisition algorithm. |
| `LatentSpaceStrategy` | Optimize `A(reconstruct(z))` in fitted latent coordinates; `latent/strategy.py`. | **Implemented/Delegated**. BoTorch optimizes latent coordinates, candidates are reconstructed and clipped to public bounds, and the original acquisition is re-evaluated. |

### Coordinate and constraint invariants

- TuRBO's `length` is relative to the original
  bounds. With weights `w` of geometric mean
  one, the local half-width in dimension `i`
  is `0.5 * length * w[i] * (upper[i]-lower[i])`,
  clipped to the public box.
- TuRBO's observed best and state-decision best
  may differ when `state_values` are supplied;
  a robust or denoised trust-region decision is
  not identical to the raw observation.
- BAxUS target coordinates are transformed
  through a signed sparse embedding into the
  public box. Its restart can **increase target
  dimension** rather than merely reset the
  same low-dimensional search.
- Latent reconstruction can distort candidates:
  the strategy reports original-space clipping
  distance and latent round-trip distance.
  Clipping is **not** a guarantee of arbitrary
  nonlinear candidate feasibility.
- Embedding/latent strategies that lack a
  mathematically valid mapping of supplied
  public-coordinate candidate constraints
  reject them via
  `reject_unmapped_candidate_constraints`.
  Constraint rejection is safer than silently
  treating latent-space bounds as equivalent.

**Source and assertion evidence:**
`src/robotorchan/optim/trust_region/turbo.py`,
`src/robotorchan/optim/embedding/{rembo,hesbo,alebo,baxus}.py`,
`src/robotorchan/optim/latent/strategy.py`,
`tests/optim/trust_region/test_turbo_strategy.py`,
`tests/optim/embedding/test_baxus_strategy.py`,
`tests/optim/embedding/test_baxus_reference_schedule.py`,
`tests/optim/latent/test_latent_space_strategy.py`.
Phase 9 retains independent state-transition
identities, coordinate correctness and
optimization-quality verification.
