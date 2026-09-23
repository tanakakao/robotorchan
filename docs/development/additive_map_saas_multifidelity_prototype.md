# Additive MAP-SAAS × multi-fidelity covariance prototype

## Result

Phase 35 narrows the additive candidate to a covariance-composition prototype. It does not add a
public model.

BoTorch's additive MAP-SAAS model is not equivalent to the design-only Matern + SAAS construction
used by `MapSaasMultiFidelityGP`. The additive design covariance must remain a first-class object
and must be composed with native fidelity kernels without allowing fidelity coordinates to enter
the additive SAAS relevance mechanism.

## Proposed composition

```text
raw X
  |
  +-- design dims ------> additive MAP-SAAS design covariance
  |
  +-- data fidelity ----> native DownsamplingKernel
  |
  +-- iteration fidelity -> native ExponentialDecayKernel
  |
  +---------------------> exact multi-fidelity covariance
```

The prototype must preserve the following ownership rule:

```text
SAAS / additive parameters -> design dimensions only
fidelity parameters        -> structural fidelity dimensions only
```

## Why direct subclass composition is not yet justified

`AdditiveMapSaasSingleTaskGP` is a complete exact GP model rather than a public design-kernel
factory. Treating the model itself as a covariance component would conflate model, likelihood, and
posterior ownership. Copying its internal kernel construction would instead couple robotorchan to
BoTorch internals.

Therefore Phase 35 does not introduce a source implementation that depends on private upstream
construction details.

## Implementation seam to prove

A future executable prototype may proceed only if one of these seams is available and stable:

1. a public BoTorch utility that builds the additive MAP-SAAS design covariance independently of
   the GP model; or
2. a small robotorchan-owned additive covariance whose statistical contract is documented and
   tested independently, with a clear reason why BoTorch's public API cannot express it.

The second option is not a wrapper and must meet the project's custom-implementation bar.

## Required executable evidence

Before public export, tests must prove:

- fidelity dimensions never appear in additive component `active_dims`;
- SAAS relevance parameters cover design dimensions only;
- data-only, iteration-only, and combined fidelity configurations are valid;
- negative fidelity indices normalize correctly;
- overlapping fidelity roles are rejected;
- raw training tensors remain unchanged;
- posterior, sampling, conditioning, and fantasize are valid;
- native cost-aware MF acquisition executes;
- the covariance is observably additive rather than merely a renamed single design kernel.

## Ensemble boundary

This prototype does not unlock ensemble MAP-SAAS × MF. Ensemble support remains gated on a
separate posterior audit because BoTorch's ensemble model has mixture/ensemble semantics that a
single exact multi-fidelity GP does not inherit automatically.

## Phase 35 decision

Do not add `AdditiveMapSaasMultiFidelityGP` yet. The statistical target is clear, but the current
public wrapper surface does not provide a justified thin-wrapper implementation seam. The next
phase should inspect the installed/upstream BoTorch public MAP-SAAS utilities specifically for a
stable additive covariance factory. If none exists, decide explicitly whether a robotorchan-owned
kernel implementation is worth the maintenance cost.
