# Phase 13 optimizer candidate assessment

Phase 13 compares the remaining optimizer candidates by whether they add a
distinct capability to robotorchan rather than by algorithm count.

## PSO

Implemented as `optimize_acqf_pso`.

PSO is useful as a gradient-free scalar acquisition optimizer with a different
search dynamic from DE, CMA-ES, and GA. The implementation is native PyTorch,
evaluates the swarm in a batch, supports q-batches by flattening q x d per
particle, preserves dtype/device, and uses the common CandidateConstraints
penalty adapter.

The returned acquisition value is always the raw acquisition value, not the
penalized ranking score.

## NSGA-III

Not implemented in this phase. NSGA-III mainly adds value for genuinely
many-objective vector optimization. Phase 12 already separates vector
optimization from scalar BoTorch acquisition optimization and provides
NSGA-II. Adding NSGA-III now would increase implementation and maintenance
surface without improving normal qEHVI/qNEHVI optimization.

It remains a candidate when robotorchan has a concrete many-objective direct
surrogate or multi-criterion workflow and can benchmark reference-direction
behavior against NSGA-II.

## MOEA/D

Not implemented in this phase. MOEA/D also targets vector-valued
multi-objective optimization and overlaps the current NSGA-II role. It should
be added only with a concrete decomposition-based use case or benchmark showing
a material advantage for the library's target workflows.

## Decision

The scalar acquisition optimizer set now has materially different local and
global choices: BoTorch, Torch gradient optimization, sampling, DE, CMA-ES,
GA, hybrid global-to-local, and PSO. NSGA-II remains a separate vector-target
optimizer. NSGA-III and MOEA/D are intentionally deferred rather than exposed
as redundant scalar acquisition backends.
