# MAP-SAAS × multi-fidelity composition audit

## Decision

Phase 22 does **not** add a public MAP-SAAS multi-fidelity model.

The current BoTorch-backed contracts make a thin wrapper insufficient. robotorchan can build a
sound model later, but it must explicitly separate the SAAS design kernel from the fidelity
covariance instead of applying a SAAS prior to the complete multi-fidelity covariance.

## Current source-of-truth

robotorchan currently exposes:

- `SingleTaskMultiFidelityGP`, which delegates the multi-fidelity covariance to BoTorch;
- `AdditiveMapSaasSingleTaskGP` and `EnsembleMapSaasSingleTaskGP`, which wrap BoTorch MAP-SAAS;
- PCA, PLS, and random-projection multi-fidelity models, where reduction is restricted to design
  columns and fidelity columns remain structural inputs.

BoTorch's MAP-SAAS utility is reusable at the kernel level, but there is no dedicated upstream
MAP-SAAS multi-fidelity model for robotorchan to wrap.

## Statistical boundary

For raw input `X = [x_design, x_fidelity]`, the intended covariance is conceptually:

```text
K(X, X') = compose(
    K_design_saas(x_design, x_design'),
    K_fidelity(x_fidelity, x_fidelity'),
)
```

SAAS shrinkage belongs to design lengthscales only. Fidelity-kernel parameters must not be
interpreted as sparse design relevance parameters.

Rejected shortcuts:

1. pass the complete multi-fidelity covariance to `add_saas_prior`;
2. subclass a MAP-SAAS single-task model and merely add fidelity arguments;
3. claim MAP-SAAS by adding an unrelated prior to `SingleTaskMultiFidelityGP`;
4. treat fidelity columns as ordinary ARD design dimensions;
5. advertise fully Bayesian SAAS multi-fidelity without a fidelity-aware Pyro model.

## Feasible implementation path

A future exact MAP-SAAS multi-fidelity implementation must prove:

1. design and fidelity dimensions are resolved in raw space;
2. the design kernel has ARD only over design dimensions;
3. `add_saas_prior` applies only to that design kernel;
4. fidelity covariance uses BoTorch-native fidelity components;
5. composition preserves BoTorch multi-fidelity semantics;
6. `iteration_fidelity` and `data_fidelities` remain distinct;
7. raw training tensors are preserved;
8. exact-GP MLL and raw-space posterior remain valid;
9. posterior sampling and a true fidelity-aware acquisition path execute.

Prefer upstream kernel-building utilities over copied BoTorch internals. If stable public APIs
cannot express the required composition, the model remains research-gated.

## Additive versus ensemble MAP-SAAS

`AdditiveMapSaasSingleTaskGP` and `EnsembleMapSaasSingleTaskGP` are not interchangeable.
An additive variant needs an explicit design-component/fidelity composition. An ensemble variant
also needs ensemble posterior semantics to remain valid after that composition.

The first implementation candidate should therefore establish one exact MAP-SAAS multi-fidelity
contract. The ensemble variant should follow only after independent posterior and acquisition
validation.

## Fully Bayesian boundary

Fully Bayesian SAAS × multi-fidelity remains separate. It requires a probabilistic model whose
sampled design lengthscales and fidelity covariance coexist in the same Pyro model. A MAP
implementation is not evidence that the fully Bayesian variant is supported.

## Phase 22 outcome

| Combination | Status |
| --- | --- |
| PCA × MultiFidelity | implemented |
| PLS × MultiFidelity | implemented |
| RandomProjection × MultiFidelity | implemented |
| MAP-SAAS × MultiFidelity | feasible, implementation-gated |
| Ensemble MAP-SAAS × MultiFidelity | feasible after single-model contract |
| Fully Bayesian SAAS × MultiFidelity | research-gated |

The next implementation phase should inspect BoTorch's exact fidelity covariance construction and
prototype a design-only SAAS prior composition. Public export should occur only after proving that
fidelity parameters are excluded from SAAS shrinkage.

## Phase 23 follow-up

The covariance prototype is documented in `map_saas_multifidelity_prototype.md`. The audit found
a sound implementation seam for `linear_truncated=False`: attach SAAS only to the non-fidelity
design covariance passed through BoTorch's `covar_module` argument, while leaving native fidelity
kernels untouched. The linear-truncated path remains outside the first implementation contract.

## Phase 34 follow-up

The specialized additive and ensemble families were re-audited after the reference
`MapSaasMultiFidelityGP` implementation and runtime stabilization. The reference implementation
does not collapse the remaining distinction: additive covariance composition and ensemble
posterior semantics still require independent evidence. See
`additive_ensemble_map_saas_multifidelity_audit.md` for the implementation gates.
