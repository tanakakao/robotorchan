# Cross-cutting acquisition optimizer behavior

Phase 14 audits behavior that cuts across numerical optimizer families rather
than adding another algorithm.

## q-batch

BoTorch, Torch, DE, CMA-ES, GA, PSO, and the global-to-local hybrid operate on
a joint q x d candidate. Derivative-free backends flatten q x d internally but
evaluate the acquisition on the original q-batch shape.

NSGA-II is intentionally outside this contract because it optimizes a
vector-valued target rather than a scalar acquisition function.

## Sequential generation and pending points

`optimize_acqf_sequential` provides an explicit backend-independent sequential
adapter. It repeatedly requests q=1, updates the acquisition's `X_pending`,
and restores any pre-existing pending points when complete, including when an
optimizer call raises.

This is an adapter, not a claim that every numerical backend natively
implements BoTorch's `sequential=True` semantics.

## Fixed features

BoTorch and the Torch backend already support fixed features. Phase 14 adds
fixed-feature projection to DE, GA, and PSO. Fixed values are applied before
acquisition and constraint evaluation, and the returned candidate is evaluated
again to return the raw acquisition value.

CMA-ES and the hybrid backend remain unsupported for fixed features in this
phase. Their capability metadata stays false rather than silently approximating
the behavior.

## Deferred items

Acquisition-specific pending-point behavior remains the responsibility of the
BoTorch acquisition object. Phase 14 does not synthesize pending-point support
for acquisitions that do not expose `set_X_pending`.

Mixed GA structured-variable repair and the earlier constrained-return
correctness gaps are separate correctness items and remain subject to the later
E2E/final audit phases.
