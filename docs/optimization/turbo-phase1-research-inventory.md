# TuRBO / Trust Region Phase 1 research and inventory

## Scope

This document records the Phase 1 source-of-truth audit for the TuRBO and
trust-region development flow. The audit is based on the current `main`
implementation, the BoTorch TuRBO-1 tutorial, and the original TuRBO reference
implementation and paper.

Phase 1 does not redefine TuRBO as an acquisition function or optimizer backend.
TuRBO remains a stateful search strategy that restricts candidate generation to
an adaptive local region while reusing robotorchan model, acquisition,
initialization, sampling, and optimizer primitives.

## Existing robotorchan implementation

The current repository already contains:

- `SearchStrategy` and `SearchResult` as the search-strategy boundary.
- `TuRBOState`, `update_turbo_state`, and `TuRBOStrategy`.
- public-space trust-region bounds and incumbent tracking.
- q-batch forwarding to BoTorch `optimize_acqf`.
- deterministic state-transition and integration tests.
- TuRBO documentation and a high-dimensional sequential benchmark.
- adjacent stateful high-dimensional strategies, including BAxUS.
- optimizer dispatch with linear, equality, and nonlinear candidate constraints.
- batch, pending-point, and fantasization infrastructure elsewhere in the
  optimization stack.

This means the new development flow is an enhancement and correctness audit of
an existing baseline, not a green-field implementation.

## Source-of-truth findings

### State semantics

The current `TuRBOState` has the standard length, counters, tolerances,
best-value, and restart fields. However, its default `failure_tolerance=4` is
independent of input dimension and batch size.

The BoTorch TuRBO-1 tutorial derives failure tolerance from both quantities:

`ceil(max(4 / batch_size, dim / batch_size))`.

The original TuRBO-1 implementation also uses dimension and batch size when
setting its failure tolerance. Phase 2 should therefore make dimension and batch
semantics explicit in the state contract rather than retaining a universal
default.

The current success tolerance of 3 matches the original TuRBO implementation.
BoTorch's current tutorial uses a different tutorial default while explicitly
noting the original value. robotorchan should keep the choice configurable and
document which baseline it follows.

### Improvement semantics

The current standalone state update applies a relative numerical tolerance, but
`TuRBOStrategy.update_state` moves the center whenever the raw candidate value
exceeds the previous best. This can move the incumbent for an improvement that
the state machine classified as a failure.

Phase 2 must define one improvement predicate and use it consistently for the
counter transition, best-value update semantics, and incumbent movement.

The baseline remains observed-best maximization. Minimization and transformed
objectives must be integrated through explicit objective semantics rather than
hidden sign conventions.

### Trust-region geometry

The current trust region is an isotropic box expressed as a fraction of each
public bound range. This is scale-aware with respect to the global box but is
not the ARD-shaped TuRBO geometry used by the reference implementations.

The reference TuRBO geometry normalizes lengthscale weights to unit geometric
mean and uses those weights to stretch or contract each dimension. Phase 3
should add this as an explicit geometry component with an isotropic fallback
when meaningful ARD lengthscales are unavailable.

The canonical TuRBO derivation assumes normalized coordinates in `[0, 1]^d`.
robotorchan must not silently double-normalize models that already own an input
transform. Trust-region geometry should therefore have explicit public-space
normalization semantics independent of the surrogate's internal transform.

### Candidate optimization

The current `TuRBOStrategy.optimize` imports BoTorch `optimize_acqf`
directly. robotorchan now has a richer optimizer dispatch layer supporting
structured constraints, initialization behavior, multiple backends, and
cross-cutting validation.

The target architecture should route acquisition-optimization candidate
generation through robotorchan's public optimizer contract where compatible.
Direct BoTorch calls should only remain where a deliberate TuRBO-specific
semantic requires them.

### Thompson sampling

The current TuRBO strategy has no TuRBO Thompson-sampling path. The BoTorch
tutorial and original implementation use trust-region candidate sampling,
dimension-wise perturbation, and posterior sampling. The perturbation
probability is dimension dependent and candidates with an empty perturbation
mask are forced to perturb at least one dimension.

Phase 5 should implement this as a separate candidate-generation path, reusing
robotorchan posterior/sampling primitives rather than treating TS as acquisition
optimization.

### Restart

The current implementation detects convergence and blocks further optimization,
but does not execute a restart policy. This is a useful low-level contract and
should remain distinguishable from restart execution.

Phase 7 should introduce an explicit restart policy. Restart must not silently
fall back to a random point. The policy, seed, new initial design, and whether
old local observations remain active must be observable.

### Constraints and async

Candidate constraints already exist in the optimizer layer, including linear
inequality, equality, and nonlinear inequality constraints. TuRBO should
intersect these with local bounds instead of translating the trust region into a
second constraint system.

Batch / async / fantasization support also already exists elsewhere. TuRBO must
reuse pending-point acquisition semantics and must update state only from
completed evaluations. Pending candidates are not successes or failures.

### High-dimensional and reduced-space models

TuRBO already coexists with high-dimensional search strategies and is exercised
with a reduced-space model in integration tests. This does not establish full
semantic compatibility.

The default contract for TuRBO remains a trust region in the public/raw input
space. Reduced-space trust regions are model-specific behavior and must not be
inferred merely because a surrogate internally performs PCA, PLS, random
projection, or another representation transform.

SAAS and MAP-SAAS require explicit validation of lengthscale extraction and ARD
weight semantics before they are advertised as ARD-aware TuRBO combinations.

## Architecture decision

The target dependency direction is:

```text
model / posterior
      |
acquisition or posterior sampling
      |
TuRBO search strategy
  |-- immutable state value
  |-- improvement/state transition
  |-- trust-region geometry
  |-- restart policy
  |-- candidate-generation policy
      |
robotorchan optimizer / initializer / sampler
      |
candidate in public input space
```

The following responsibilities stay separate:

1. `TuRBOState` stores serializable algorithm state.
2. A pure state-update function performs success/failure transitions.
3. Trust-region geometry computes local bounds from center, global bounds,
   state, and optional dimension weights.
4. Candidate generation consumes the local region and delegates to either the
   optimizer path or the Thompson-sampling path.
5. Restart policy chooses how a converged local run is re-entered.
6. A strategy object may coordinate these components but must not hide model
   fitting, objective evaluation, or asynchronous completion.

No new generic strategy framework is justified in Phase 1. The existing
`SearchStrategy` abstraction is sufficient as the public family boundary.
Stateful-specific interfaces should be introduced only where TuRBO requirements
demonstrate a concrete need.

## Compatibility baseline

| Area | Phase 1 status | Required action |
| --- | --- | --- |
| Continuous single-objective | baseline exists | strengthen |
| q=1 acquisition optimization | baseline exists | route through robotorchan |
| q-batch | partial | state semantics and E2E |
| State transition | partial | dimension/batch-aware contract |
| Restart trigger | exists | preserve |
| Restart execution | missing | implement later |
| ARD geometry | missing | implement |
| Isotropic geometry | exists | retain fallback |
| Thompson sampling | missing in TuRBO | implement |
| Constraints | optimizer primitive exists | integrate |
| Async / X_pending | external primitive exists | integrate |
| Fantasization | external primitive exists | integrate |
| SAAS / MAP-SAAS | not established | validate |
| Reduced-space models | partial evidence | define per-model semantics |
| Mixed / discrete | not established | feasibility study |
| Multi-objective | not established | feasibility study |
| MultiFidelity | not established | feasibility study |
| Robust / input perturbation | not established | integrate after baseline |

## Phase 2 entry criteria

Phase 2 should proceed without another architecture redesign. Its concrete
targets are:

- make `dim` and `batch_size` explicit state semantics;
- derive or explicitly override failure tolerance;
- centralize the improvement predicate;
- preserve immutable/pure state transitions;
- test success, failure, expansion, shrink, threshold behavior, and restart;
- keep state serialization independent from model objects;
- avoid introducing restart execution or ARD geometry prematurely.

The existing TuRBO implementation is therefore retained as the migration
baseline, but it is not yet considered the completed TuRBO-1 implementation for
this development flow.
