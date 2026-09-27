# Optimizer expansion final audit

Phase 20 closes the optimizer expansion workstream by checking the public scalar dispatcher,
backend capability metadata, runtime contracts, candidate constraints, structured variables,
and regression coverage as one system.

## Public scalar backends

The public `robotorchan.optim.optimize_acqf` dispatcher supports BoTorch, Torch Adam/AdamW/SGD,
random and Sobol sampling, differential evolution, CMA-ES, genetic algorithm, PSO, and the
global-to-local hybrid backend. Mixed-variable GA and vector-valued NSGA-II remain specialist
backend APIs because their contracts differ from scalar continuous acquisition optimization.

## Cross-cutting contracts

- Bounds are finite floating-point tensors with shape `[2, d]`, `d >= 1`, and strict lower/upper ordering.
- Seeded stochastic backends use local generators. `seed=None` preserves ordinary PyTorch RNG behavior.
- Capability metadata is descriptive and the public dispatcher rejects unsupported requested combinations.
- Fixed features are supported by the public BoTorch, Torch, sampling, DE, GA, and PSO paths where advertised.
- Derivative-free constraint support uses feasibility penalties; the returned acquisition value is always the raw acquisition value.
- Equality tolerance must be non-negative.
- Mixed GA validates legal integer domains and finite, unique categorical values. Categorical genes are sampled during initialization and inherited across generations instead of being unconditionally resampled after crossover.

## Deliberate boundaries

Constraint-penalty global optimizers do not provide the same exact-feasibility guarantee as BoTorch's
native constrained local optimization. The hybrid backend therefore remains a global exploration plus
BoTorch local refinement mechanism rather than a replacement for BoTorch constraint semantics.

NSGA-II is retained for vector-valued optimization and is not required for scalar qEHVI/qNEHVI
acquisition optimization.

Tree-specific constraint handling is outside this workstream and can reuse the common derivative-free
constraint contract in a separate implementation flow.
