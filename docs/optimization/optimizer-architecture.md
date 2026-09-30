# Acquisition optimizer architecture

robotorchan extends BoTorch acquisition optimization. It does not define a
second acquisition-function abstraction.

## Design goal

A user who already knows BoTorch should be able to use robotorchan without
learning a replacement optimization API. BoTorch types and conventions remain
the default vocabulary:

- `AcquisitionFunction`
- tensor `bounds`
- candidate batch size `q`
- `fixed_features`
- BoTorch-style linear and nonlinear candidate constraints

robotorchan adds alternative numerical optimizer backends and convenience
selection on top of that contract.

## Responsibility boundary

The optimization stack has three responsibilities.

1. **Search strategy** defines where optimization occurs. Examples include the
   original space, a trust region, an embedding, or a latent space.
2. **Candidate optimizer backend** defines how an acquisition function is
   numerically optimized in that space. BoTorch/SciPy, Torch optimizers,
   differential evolution, CMA-ES, and evolutionary algorithms belong here.
3. **Convenience dispatch** may select a backend from an argument, but it must
   preserve direct access to the underlying BoTorch-compatible behavior.

The numerical backends must not become surrogate-model abstractions. They
consume an already constructed BoTorch acquisition function.

## Package layout

Concrete public-input-space strategies are implemented under
`robotorchan.optim.strategies`. The canonical user-facing imports remain available from
`robotorchan.optim`. Embedding, latent-space, and trust-region strategies keep separate
packages because they own additional state and coordinate transformations.

Candidate-space constraint types and constraint evaluation helpers are owned by
`robotorchan.optim.constraints`. These remain separate from probabilistic output constraints
used by constrained acquisition functions.

## BoTorch-first rule

When BoTorch already provides the required behavior, robotorchan delegates to
BoTorch instead of reimplementing it. In particular, the baseline original-space
and mixed-space paths remain thin adapters over `botorch.optim.optimize_acqf`
and `botorch.optim.optimize_acqf_mixed`.

Future alternative backends should expose BoTorch-like arguments where those
arguments have the same semantics. Backend-specific options belong in an
explicit options/configuration surface rather than changing the meaning of a
BoTorch argument.

Unsupported behavior must raise an explicit validation error. A backend must
not silently approximate a requested constraint, q-batch behavior, discrete
domain, or fixed feature.

## Capability contract

`OptimizerCapabilities` describes numerical-backend capabilities separately
from model and acquisition capabilities. It records support for:

- continuous, integer, categorical, and mixed domains;
- candidate gradients;
- linear inequality and equality constraints;
- nonlinear and inter-point nonlinear constraints;
- joint q-batches and sequential optimization;
- fixed features;
- GPU execution and batched acquisition evaluation.

These flags are descriptive runtime contracts. They are not a scoring or
automatic-selection policy.

## Public API direction

The intended convenience API is conceptually:

```python
from robotorchan.optim import optimize_acqf

candidates, value = optimize_acqf(
    acq_function=acq_function,
    bounds=bounds,
    q=3,
    optimizer="de",
)
```

The exact dispatcher is introduced only after backend contracts are stable.
Direct BoTorch usage remains valid, and existing search strategies continue to
accept normal BoTorch acquisition functions.

This gives robotorchan two properties at the same time:

- BoTorch users can use the library naturally without relearning core concepts.
- users who want more choices can select additional optimization methods through
  a convenience argument and receive capability validation before optimization.
