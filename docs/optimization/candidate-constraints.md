# Candidate constraints across optimizer backends

Phase 10 extends the existing `CandidateConstraints` contract to derivative-free
candidate optimizers without introducing a second constraint language.

BoTorch and BoTorch-mixed backends continue to forward linear equality,
linear inequality, and nonlinear inequality constraints to BoTorch itself.

Differential Evolution, CMA-ES, continuous GA, and mixed-variable GA evaluate the
same `CandidateConstraints` contract through a common violation function. Their
search objective is augmented by a configurable positive `constraint_penalty`.
Equality constraints use `equality_tolerance` (default `1e-6`).

Both intra-point and inter-point q-batch constraints are supported. Nonlinear
callables need to be differentiable only for gradient-based BoTorch optimization;
derivative-free backends only require valid tensor outputs.

This penalty implementation deliberately preserves the native evolutionary search
operators. It does not claim exact feasibility: callers should use a sufficiently
large penalty and verify the returned candidate when feasibility is safety-critical
or mathematically strict. Benchmarking feasibility rates is part of the optimizer
benchmark phase.

The PyTorch `gen_candidates_torch` backend remains constraint-disabled because
that BoTorch generator does not provide the same constrained optimization contract
as the SciPy-backed BoTorch candidate generator. Constraints are not silently
approximated there.
